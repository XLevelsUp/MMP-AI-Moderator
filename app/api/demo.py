"""Test page only: run one file through the real checks and the real database writes on a dummy
database, then show the stored rows. Off unless DEMO_DATABASE_URL is set, and it refuses any
database that isn't on this machine, so it can never reach the real one.
"""
from time import perf_counter, time
from typing import Annotated
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import psycopg
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api.schemas import ContentType, MediaType, MEDIA_CONTEXT
from app.core.config import settings
from app.core.versioning import current_model_version
from app.jobs.queue import Job
from app.jobs.results import DecisionStore, failure_fallback
from app.pipeline.context import ModerationContext
from app.pipeline.orchestrator import finish, run_pipeline
from app.validation.errors import ValidationFailure
from app.validation.intake import remove_temp_file, save_upload


router = APIRouter()

DEMO_ACCOUNT = UUID("00000000-0000-4000-8000-000000000001")
DEMO_PET = UUID("00000000-0000-4000-8000-000000000002")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def demo_url() -> str:
    url = settings.database.demo_url
    if not url:
        raise HTTPException(status_code=404, detail="The database preview is off (DEMO_DATABASE_URL is not set).")
    if urlsplit(url).hostname not in LOCAL_HOSTS:
        raise HTTPException(status_code=403, detail="The database preview only works with a database on this machine.")
    return url


def new_row(url: str, media_type: str, name: str) -> UUID:
    """A pending row, as the app would create before sending the request."""
    media_id = DEMO_PET if media_type == "pet_profile_photo" else uuid4()
    with psycopg.connect(url) as db:
        db.execute("insert into identity.accounts (id, auth_user_id) values (%s, %s) on conflict do nothing", (DEMO_ACCOUNT, DEMO_ACCOUNT))
        db.execute("insert into pets.pets (id, owner_account_id, name, status) values (%s, %s, 'Demo pet', 'active') on conflict do nothing", (DEMO_PET, DEMO_ACCOUNT))
        if media_type == "pet_media":
            db.execute("insert into pets.pet_media (id, pet_id, kind, url, storage_path, display_order, moderation_status) "
                       "values (%s, %s, 'image', %s, %s, (select coalesce(max(display_order), 0) + 1 from pets.pet_media), 'pending')",
                       (media_id, DEMO_PET, name, name))
            db.execute("update social.posts set moderation_status = 'pending' where id = %s", (media_id,))
        elif media_type == "post_media":
            post = uuid4()
            db.execute("insert into social.posts (id, pet_id, moderation_status) values (%s, %s, 'pending')", (post, DEMO_PET))
            db.execute("insert into social.post_media (id, post_id, storage_path, media_type, sort_order, moderation_status) "
                       "values (%s, %s, %s, 'photo', 0, 'pending')", (media_id, post, name))
        else:
            db.execute("update pets.pets set profile_photo_path = %s, profile_photo_moderation_status = 'pending', "
                       "profile_photo_moderation_rejection_code = null where id = %s", (name, DEMO_PET))
    return media_id


def stored_rows(url: str, media_type: str, media_id: UUID, content_id: UUID) -> dict:
    def row(db, sql, value):
        found = db.execute(sql, (value,)).fetchone()
        return found[0] if found else None

    with psycopg.connect(url) as db:
        rows = {"moderation.decisions": row(db, "select to_jsonb(d) from moderation.decisions d where id = %s", content_id)}
        if media_type == "pet_media":
            rows["pets.pet_media"] = row(db, "select to_jsonb(m) from pets.pet_media m where id = %s", media_id)
            rows["social.posts"] = row(db, "select to_jsonb(p) from social.posts p where id = %s", media_id)
        elif media_type == "post_media":
            media = row(db, "select to_jsonb(m) from social.post_media m where id = %s", media_id)
            rows["social.post_media"] = media
            rows["social.posts"] = row(db, "select to_jsonb(p) from social.posts p where id = %s", media["post_id"])
        else:
            rows["pets.pets"] = row(db, "select to_jsonb(p) from pets.pets p where id = %s", media_id)
    return rows


@router.post("/api/v1/demo/moderate")
async def demo_moderate(
    file: Annotated[UploadFile, File()],
    content_type: Annotated[ContentType, Form()],
    media_type: Annotated[MediaType, Form()] = "pet_media",
    simulate_failure: Annotated[bool, Form()] = False,
    declared_breed_id: Annotated[UUID | None, Form()] = None,
):
    url = demo_url()
    context = MEDIA_CONTEXT[media_type]
    ctx = ModerationContext(content_id=uuid4(), user_id=DEMO_ACCOUNT, content_type=content_type,
                            context=context, started_at=perf_counter())
    name = f"pet-images/demo/{file.filename}"
    media_id = new_row(url, media_type, name)
    sha256, size = None, None
    try:
        intake = await save_upload(file, content_type, ctx.content_id)
    except ValidationFailure as failure:
        ctx.reject("MOD_008", failure.flag)
        response = finish(ctx)
    else:
        ctx.path, ctx.size_bytes, ctx.sha256 = intake.path, intake.size_bytes, intake.sha256
        sha256, size = intake.sha256, intake.size_bytes
        try:
            if simulate_failure:  # what the worker writes once all 3 tries have failed
                ctx.flags.append("check_incomplete")
                response = failure_fallback(ctx.to_response())
                sha256, size = None, 0
            else:
                response = await run_in_threadpool(run_pipeline, ctx)
        finally:
            remove_temp_file(intake.path)

    job = Job(str(ctx.content_id), str(DEMO_ACCOUNT), str(DEMO_PET), media_type, str(media_id),
              f"https://demo.local/storage/v1/object/public/{name}", content_type, context,
              current_model_version(), time(), str(declared_breed_id or ""), "")
    await run_in_threadpool(DecisionStore(url).save, job, response, sha256, size)
    rows = await run_in_threadpool(stored_rows, url, media_type, media_id, ctx.content_id)
    return {"media_type": media_type, "media_id": media_id, "rows": rows}


@router.get("/api/v1/demo/decisions")
def demo_decisions(limit: int = 20):
    with psycopg.connect(demo_url()) as db:
        found = db.execute("select to_jsonb(d) from moderation.decisions d order by created_at desc limit %s", (limit,)).fetchall()
    return [row[0] for row in found]
