"""Uploads waiting for the AI: Redis Streams, shared by the API and the GPU workers.

A job stays in the stream until a worker has written its result to the database, so a worker
that dies loses nothing: another worker takes the job over.
"""
import logging
from dataclasses import asdict, dataclass, fields

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from app.core.config import settings


logger = logging.getLogger("mmp.queue")


class QueueFull(Exception):
    """Too many uploads are waiting: the caller should retry later."""


@dataclass(frozen=True)
class Job:
    content_id: str
    account_id: str
    pet_id: str  # empty when the app did not send one
    media_type: str
    media_id: str
    media_url: str
    content_type: str
    context: str
    model_version: str
    queued_at: float
    declared_breed_id: str = ""        # empty when the app did not send one
    declared_is_cross_breed: str = ""  # true, false or empty


@dataclass(frozen=True)
class Delivery:
    stream: str
    message_id: str
    job: Job


def stream_for(content_type: str) -> str:
    q = settings.queue
    return q.video_stream if content_type == "video" else q.image_stream


JOB_FIELDS = {f.name for f in fields(Job)}


class BadJob(Exception):
    """A stream entry that isn't a valid job: written by an older or newer API, or by hand."""


def job_from(raw: dict) -> Job:
    """Unknown fields are ignored (a newer API may add some); a missing one is a BadJob."""
    values = {decoded(k): decoded(v) for k, v in raw.items()}
    try:
        values = {k: v for k, v in values.items() if k in JOB_FIELDS}
        return Job(**{**values, "queued_at": float(values["queued_at"])})
    except (KeyError, TypeError, ValueError) as exc:
        raise BadJob(f"not a valid job: {exc}") from exc


def decoded(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


class JobQueue:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def ensure_groups(self) -> None:
        q = settings.queue
        for stream in (q.image_stream, q.video_stream):
            try:
                await self.redis.xgroup_create(stream, q.group, id="0", mkstream=True)
            except ResponseError as exc:
                if "BUSYGROUP" not in str(exc):  # already there: another worker made it
                    raise

    async def enqueue(self, job: Job) -> None:
        q = settings.queue
        limit = q.video_max_waiting if job.content_type == "video" else q.image_max_waiting
        stream = stream_for(job.content_type)
        if await self.redis.xlen(stream) >= limit:
            raise QueueFull()
        await self.redis.xadd(stream, {k: str(v) for k, v in asdict(job).items()})

    async def next_job(self, consumer: str, *, video_first: bool = False) -> Delivery | None:
        """The next waiting job, photos before videos unless video_first. Never blocks."""
        q = settings.queue
        streams = [q.video_stream, q.image_stream] if video_first else [q.image_stream, q.video_stream]
        for stream in streams:
            while True:  # a bad entry is dropped and the next one read
                found = await self.redis.xreadgroup(q.group, consumer, {stream: ">"}, count=1)
                messages = [m for _, batch in found for m in batch]
                if not messages:
                    break
                message_id, raw = messages[0]
                try:
                    return Delivery(stream, decoded(message_id), job_from(raw))
                except BadJob as exc:
                    await self.drop(stream, decoded(message_id), exc)
        return None

    async def drop(self, stream: str, message_id: str, reason: Exception) -> None:
        """A bad entry would be retried forever: log it and take it out of the stream."""
        logger.error("dropped queue entry %s from %s: %s", message_id, stream, reason)
        await self.redis.xack(stream, settings.queue.group, message_id)
        await self.redis.xdel(stream, message_id)

    async def pending(self) -> int:
        """Jobs handed to a worker but not finished: in progress, or left behind by a dead worker."""
        q = settings.queue
        total = 0
        for stream in (q.image_stream, q.video_stream):
            try:
                summary = await self.redis.xpending(stream, q.group)
            except ResponseError:  # no group yet: nothing pending
                continue
            total += summary["pending"] if isinstance(summary, dict) else summary[0]
        return total

    async def reclaim(self, consumer: str) -> list[tuple[Delivery, int]]:
        """Jobs a dead worker left behind, each with how many times it has been handed out."""
        q = settings.queue
        taken = []
        for stream in (q.image_stream, q.video_stream):
            claimed = await self.redis.xautoclaim(
                stream, q.group, consumer, min_idle_time=q.reclaim_idle_seconds * 1000, count=q.reclaim_batch,
            )
            for message_id, raw in claimed[1]:
                if not raw:  # removed from the stream while it was waiting
                    continue
                try:
                    job = job_from(raw)
                except BadJob as exc:
                    await self.drop(stream, decoded(message_id), exc)
                    continue
                pending = await self.redis.xpending_range(stream, q.group, message_id, message_id, 1)
                attempts = pending[0]["times_delivered"] if pending else 1
                taken.append((Delivery(stream, decoded(message_id), job), attempts))
        return taken

    async def remove(self, delivery: Delivery) -> None:
        await self.redis.xack(delivery.stream, settings.queue.group, delivery.message_id)
        await self.redis.xdel(delivery.stream, delivery.message_id)

    async def waiting(self) -> dict[str, int]:
        q = settings.queue
        return {"image": await self.redis.xlen(q.image_stream), "video": await self.redis.xlen(q.video_stream)}
