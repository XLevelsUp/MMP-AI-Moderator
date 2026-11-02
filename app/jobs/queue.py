"""Uploads waiting for the AI: Redis Streams, shared by the API and the GPU workers.

A job stays in the stream until a worker has stored its answer, so a worker that dies loses
nothing: another worker takes the job over. The state of each job (queued / done / failed) is
kept separately for GET /api/v1/moderate/{content_id}. The database holds the real result.
"""
import json
from dataclasses import asdict, dataclass
from time import time
from uuid import UUID

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from app.api.schemas import ModerationResponse
from app.core.config import settings


class QueueFull(Exception):
    """Too many uploads are waiting: the caller should retry later."""


@dataclass(frozen=True)
class Job:
    content_id: str
    account_id: str
    pet_id: str  # empty when the app did not send one
    entity_type: str
    entity_id: str
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


@dataclass(frozen=True)
class JobState:
    status: str  # queued or done
    response: ModerationResponse | None = None


def state_key(content_id: UUID | str) -> str:
    return f"mod:job:{content_id}"


def stream_for(content_type: str) -> str:
    q = settings.queue
    return q.video_stream if content_type == "video" else q.image_stream


def job_from(fields: dict) -> Job:
    values = {k.decode(): v.decode() for k, v in fields.items()}
    return Job(**{**values, "queued_at": float(values["queued_at"])})


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
        await self.redis.set(state_key(job.content_id), json.dumps({"status": "queued"}), ex=q.result_ttl_seconds)
        await self.redis.xadd(stream, {k: str(v) for k, v in asdict(job).items()})

    async def state(self, content_id: UUID | str) -> JobState | None:
        raw = await self.redis.get(state_key(content_id))
        if raw is None:
            return None
        data = json.loads(raw)
        response = ModerationResponse.model_validate(data["response"]) if "response" in data else None
        return JobState(data["status"], response)

    async def next_job(self, consumer: str, *, video_first: bool = False) -> Delivery | None:
        """The next waiting job, photos before videos unless video_first. Never blocks."""
        q = settings.queue
        streams = [q.video_stream, q.image_stream] if video_first else [q.image_stream, q.video_stream]
        for stream in streams:
            found = await self.redis.xreadgroup(q.group, consumer, {stream: ">"}, count=1)
            for _, messages in found:
                for message_id, fields in messages:
                    return Delivery(stream, decoded(message_id), job_from(fields))
        return None

    async def reclaim(self, consumer: str) -> list[tuple[Delivery, int]]:
        """Jobs a dead worker left behind, each with how many times it has been handed out."""
        q = settings.queue
        taken = []
        for stream in (q.image_stream, q.video_stream):
            claimed = await self.redis.xautoclaim(
                stream, q.group, consumer, min_idle_time=q.reclaim_idle_seconds * 1000, count=q.reclaim_batch,
            )
            for message_id, fields in claimed[1]:
                if not fields:  # removed from the stream while it was waiting
                    continue
                pending = await self.redis.xpending_range(stream, q.group, message_id, message_id, 1)
                attempts = pending[0]["times_delivered"] if pending else 1
                taken.append((Delivery(stream, decoded(message_id), job_from(fields)), attempts))
        return taken

    async def complete(self, delivery: Delivery, response: ModerationResponse) -> None:
        state = {"status": "done", "response": response.model_dump(mode="json")}
        await self.redis.set(state_key(delivery.job.content_id), json.dumps(state), ex=settings.queue.result_ttl_seconds)
        await self.remove(delivery)

    async def remove(self, delivery: Delivery) -> None:
        await self.redis.xack(delivery.stream, settings.queue.group, delivery.message_id)
        await self.redis.xdel(delivery.stream, delivery.message_id)

    async def waiting(self) -> dict[str, int]:
        q = settings.queue
        return {"image": await self.redis.xlen(q.image_stream), "video": await self.redis.xlen(q.video_stream)}
