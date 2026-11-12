"""GPU worker: holds the models, takes uploads from the queue one at a time, updates the database.

    uv run python -m app.worker

Run one per GPU. Each loads all the models, so several on one GPU need several times the memory.
The API and the workers must share Redis and the same settings and environment.
"""
import asyncio
import logging
import os
import socket
from pathlib import Path
from time import perf_counter, time
from uuid import UUID

from redis.exceptions import RedisError

from app.cache.idempotency import IdempotencyStore, UploadKey, make_redis
from app.core.config import settings
from app.core.logging import setup_logging
from app.jobs.queue import Delivery, JobQueue
from app.jobs.results import DecisionStore, failure_fallback
from app.ml.registry import load_models
from app.pipeline.context import ModerationContext
from app.pipeline.orchestrator import finish, run_pipeline
from app.validation.csam import get_csam_checker
from app.validation.errors import ValidationFailure
from app.validation.intake import download_media, remove_temp_file


logger = logging.getLogger("mmp.worker")


class Worker:
    def __init__(self, queue: JobQueue, store: IdempotencyStore, decisions: DecisionStore, consumer: str, transport=None):
        self.queue = queue
        self.store = store
        self.decisions = decisions
        self.consumer = consumer
        self.transport = transport  # tests only: stands in for the network
        self.images_in_a_row = 0
        self.last_reclaim = 0.0

    def context_for(self, delivery: Delivery) -> ModerationContext:
        job = delivery.job
        return ModerationContext(
            content_id=UUID(job.content_id), user_id=UUID(job.account_id), content_type=job.content_type,
            context=job.context, started_at=perf_counter(),
        )

    async def answer(self, delivery: Delivery, ctx: ModerationContext):
        """Download and check the file. Returns the answer and the file's fingerprint and size."""
        job = delivery.job
        ctx.path = settings.temp_dir / f"{job.content_id}.upload"
        try:
            intake = await download_media(job.media_url, job.content_type, ctx.path, self.transport)
        except ValidationFailure as failure:
            ctx.reject("MOD_008", failure.flag)
            return finish(ctx), None, None
        ctx.size_bytes, ctx.sha256 = intake.size_bytes, intake.sha256

        key = UploadKey(intake.sha256, UUID(job.account_id), job.content_type, job.context, job.model_version, job.media_id)
        cached = await self.store.get_cached(key)
        if cached is not None:  # sent again: same file, same row, same models
            return cached.model_copy(update={"content_id": ctx.content_id}), intake.sha256, intake.size_bytes
        response = await asyncio.to_thread(run_pipeline, ctx)
        await self.store.store(key, response)
        return response, intake.sha256, intake.size_bytes

    async def process(self, delivery: Delivery) -> None:
        """A job leaves the queue only once its result is in the database. On any failure it stays
        there: reclaim() retries it, then give_up() sends it to a moderator."""
        job = delivery.job
        ctx = self.context_for(delivery)
        logger.info("job %s %s waited %.1f s", job.content_id, job.content_type, time() - job.queued_at)
        try:
            response, sha256, size = await self.answer(delivery, ctx)
            if "gate1_unavailable" in response.flags:  # a model failed to load: no real scores
                response = failure_fallback(response)
            await asyncio.to_thread(self.decisions.save, job, response, sha256, size)
            await self.queue.remove(delivery)
        except Exception:
            logger.exception("job %s failed, will be retried", job.content_id)
        finally:
            # This path, not ctx.path: a quarantined file has already moved out of it.
            remove_temp_file(settings.temp_dir / f"{job.content_id}.upload")

    async def give_up(self, delivery: Delivery) -> None:
        """No decision is possible: zeros are written, the post stays hidden and a moderator looks at it."""
        ctx = self.context_for(delivery)
        ctx.flags.append("check_incomplete")
        response = failure_fallback(ctx.to_response())
        await asyncio.to_thread(self.decisions.save, delivery.job, response, None, 0)
        await self.queue.remove(delivery)

    async def reclaim(self) -> None:
        for delivery, attempts in await self.queue.reclaim(self.consumer):
            if attempts > settings.queue.max_attempts:
                logger.error("job %s sent to a moderator after %d attempts", delivery.job.content_id, attempts - 1)
                try:
                    await self.give_up(delivery)
                except Exception:
                    logger.exception("job %s could not be recorded, will be retried", delivery.job.content_id)
            else:
                logger.warning("job %s retried, attempt %d", delivery.job.content_id, attempts)
                await self.process(delivery)

    async def step(self) -> bool:
        """Do one job if there is one. False when nothing is waiting."""
        if time() - self.last_reclaim >= settings.queue.reclaim_every_seconds:
            self.last_reclaim = time()
            await self.reclaim()

        video_first = self.images_in_a_row >= settings.queue.images_per_video
        delivery = await self.queue.next_job(self.consumer, video_first=video_first)
        if delivery is None:
            return False
        self.images_in_a_row = self.images_in_a_row + 1 if delivery.job.content_type == "image" else 0
        await self.process(delivery)
        return True

    async def run(self) -> None:
        await self.queue.ensure_groups()
        logger.info("worker %s ready", self.consumer)
        while True:
            try:
                if not await self.step():
                    await asyncio.sleep(settings.queue.poll_seconds)
            except RedisError as exc:
                logger.error("redis unavailable, retrying: %s", exc)
                await asyncio.sleep(settings.queue.redis_retry_seconds)


async def serve() -> None:
    redis = make_redis()
    worker = Worker(JobQueue(redis), IdempotencyStore(redis), DecisionStore(settings.database.url),
                    f"{socket.gethostname()}-{os.getpid()}")
    await worker.run()


def main() -> None:
    setup_logging()
    if not settings.database.url:
        raise SystemExit("DATABASE_URL is not set: the worker has nowhere to write its results.")
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    get_csam_checker()  # logs a loud warning while the placeholder is in use
    load_models()  # before the loop starts: nothing else to do until they are ready
    asyncio.run(serve())


if __name__ == "__main__":
    main()
