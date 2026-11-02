from time import perf_counter, time
from typing import Annotated
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from app.api.schemas import ContentContext, ContentType, ModerationRequest, ModerationResponse, QueuedResponse
from app.cache.idempotency import IdempotencyStore, UploadKey, get_idempotency_store
from app.core.config import settings
from app.core.logging import log_problem, log_result
from app.core.versioning import current_model_version
from app.jobs.queue import Job, JobQueue, QueueFull
from app.pipeline.context import ModerationContext
from app.pipeline.orchestrator import finish, run_pipeline
from app.validation.csam import CsamCheckError
from app.validation.errors import ProcessingTimeout, ValidationFailure
from app.validation.intake import remove_temp_file, save_upload


router = APIRouter()


def get_job_queue(store: Annotated[IdempotencyStore, Depends(get_idempotency_store)]) -> JobQueue:
    return JobQueue(store.redis)


def queued(content_id: UUID) -> JSONResponse:
    return JSONResponse(status_code=202, content=QueuedResponse(content_id=content_id).model_dump(mode="json"))


def allowed_link(url: str) -> bool:
    """Only the app's own storage: the service must never fetch an address a caller picks."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return False
    media = settings.media
    scheme_ok = parts.scheme == "https" or (not media.require_https and parts.scheme == "http")
    return scheme_ok and host is not None and host.lower() in media.allowed_hosts


@router.post(
    "/api/v1/moderate",
    status_code=202,
    response_model=QueuedResponse,
)
async def moderate_content(request: ModerationRequest, queue: Annotated[JobQueue, Depends(get_job_queue)]):
    """Accept the upload and answer at once. A worker checks it and updates the database."""
    if not allowed_link(request.media_url):
        raise HTTPException(status_code=422, detail="media_url must be an HTTPS link to the app's storage.")
    content_id = uuid4()
    job = Job(
        str(content_id), str(request.account_id), str(request.pet_id or ""), request.entity_type,
        str(request.entity_id), request.media_url, request.content_type, request.context,
        current_model_version(), time(), str(request.declared_breed_id or ""),
        "" if request.declared_is_cross_breed is None else str(request.declared_is_cross_breed).lower(),
    )
    try:
        await queue.enqueue(job)
    except QueueFull:
        log_problem(503, content_id, f"{request.content_type} queue is full")
        raise HTTPException(
            status_code=503,
            detail="Too many uploads are waiting. Try again shortly.",
            headers={"Retry-After": str(settings.queue.retry_after_seconds)},
        )
    except RedisError as exc:
        log_problem(503, content_id, f"redis unavailable: {exc}")
        raise HTTPException(status_code=503, detail="Queue unavailable. Try again shortly.")
    return QueuedResponse(content_id=content_id)


@router.get(
    "/api/v1/moderate/{content_id}",
    response_model=ModerationResponse,
    responses={202: {"model": QueuedResponse}},
)
async def moderation_result(content_id: UUID, queue: Annotated[JobQueue, Depends(get_job_queue)]):
    """For debugging and support: the database holds the real result. Kept for queue.result_ttl_seconds."""
    try:
        state = await queue.state(content_id)
    except RedisError as exc:
        log_problem(503, content_id, f"redis unavailable: {exc}")
        raise HTTPException(status_code=503, detail="Result store unavailable. Try again shortly.")
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown content_id.")
    return state.response if state.status == "done" else queued(content_id)


@router.post(
    "/api/v1/moderate/upload",
    response_model=ModerationResponse,
)
async def moderate_upload(
    file: Annotated[UploadFile, File()],
    content_type: Annotated[ContentType, Form()],
    user_id: Annotated[UUID, Form()],
    store: Annotated[IdempotencyStore, Depends(get_idempotency_store)],
    context: Annotated[ContentContext, Form()] = "post",
) -> ModerationResponse:
    """Test page and quick local runs only (QUEUE_ENABLED=false): answers in this request, writes nothing."""
    if settings.queue.enabled:
        raise HTTPException(status_code=404, detail="Not available while the queue is on.")
    ctx = ModerationContext(
        content_id=uuid4(),
        user_id=user_id,
        content_type=content_type,
        context=context,
        started_at=perf_counter(),
    )

    # Step 4 runs here, not in the orchestrator: it needs the upload stream, and
    # its SHA-256 is the idempotency key we must check before any heavy work.
    try:
        intake = await save_upload(file, content_type, ctx.content_id)
    except ValidationFailure as failure:
        # We stopped reading, so there is no full hash: this answer is not cached.
        ctx.reject("MOD_008", failure.flag)
        return finish(ctx)

    ctx.path = intake.path
    ctx.size_bytes = intake.size_bytes
    ctx.sha256 = intake.sha256
    key = UploadKey(intake.sha256, user_id, content_type, context, current_model_version())

    try:
        cached = await store.get_cached(key)
        if cached is not None:
            log_result(cached, cached=True)
            return cached

        if not await store.lock(key):
            cached = await store.wait_for_result(key)
            if cached is not None:
                log_result(cached, cached=True)
                return cached
            log_problem(409, ctx.content_id, "same upload is already being processed")
            raise HTTPException(
                status_code=409,
                detail="This upload is already being processed. Try again shortly.",
            )

        try:
            try:
                response = await run_in_threadpool(run_pipeline, ctx)
            except CsamCheckError as exc:
                # The safety check could not run: no decision, nothing stored, retry later.
                log_problem(503, ctx.content_id, str(exc))
                raise HTTPException(status_code=503, detail="Safety check unavailable. Try again shortly.")
            except ProcessingTimeout as exc:
                # Not a decision: skip storing, release the lock, let the caller retry.
                log_problem(503, ctx.content_id, str(exc))
                raise HTTPException(
                    status_code=503,
                    detail="Processing timed out. Try again shortly.",
                )

            # Only final decisions reach this line; errors skip it.
            await store.store(key, response)
            return response
        finally:
            await store.release(key)
    except RedisError as exc:
        # Redis is down: no duplicate protection and no blocklist, so no decision.
        log_problem(503, ctx.content_id, f"redis unavailable: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Duplicate-protection store unavailable. Try again shortly.",
        )
    finally:
        remove_temp_file(intake.path)
