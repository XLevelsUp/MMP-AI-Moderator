"""Production build (12 Nov 2026): API keys, start-up checks, links, health, logs, bad jobs,
the part-time worker, and the audit fixes."""
import asyncio
import json
import logging
from time import time
from unittest.mock import patch

import fakeredis
import pytest

from app.api import security
from app.cache.idempotency import UploadKey
from app.core.config import settings
from app.core.logging import JsonFormatter
from app.jobs.queue import BadJob, job_from
from app.validation.errors import ValidationFailure
from app.validation.intake import DownloadError, allowed_link, download_media
from tests.test_queue import (HOST, URL, FakeDecisions, body, enqueue_all, jpeg_bytes, make_job, parts,  # noqa: F401
                              queue_on, run, serving, worker)

KEY = "k" * 40


# ---------- API key ----------

def test_no_keys_configured_lets_requests_through(client, queue_on, parts):
    run(parts[0].ensure_groups())
    assert client.post("/api/v1/moderate", json=body()).status_code == 202


@pytest.mark.parametrize("header, status", [(None, 401), ("wrong", 401), (KEY, 202)])
def test_api_key_is_required_once_set(client, queue_on, parts, monkeypatch, header, status):
    monkeypatch.setattr(settings.api, "keys", [KEY])
    run(parts[0].ensure_groups())
    headers = {settings.api.key_header: header} if header else {}
    assert client.post("/api/v1/moderate", json=body(), headers=headers).status_code == status


# ---------- production start-up checks ----------

def test_development_has_no_start_up_problems():
    assert security.production_problems(worker=False) == []


def test_production_refuses_to_start_without_keys_hosts_or_database(monkeypatch):
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings.api, "keys", [])
    monkeypatch.setattr(settings.media, "allowed_hosts", [])
    monkeypatch.setattr(settings.database, "url", None)
    api = " ".join(security.production_problems(worker=False))
    assert "API_KEYS" in api and "MEDIA_ALLOWED_HOSTS" in api
    assert "DATABASE_URL" in " ".join(security.production_problems(worker=True))
    with pytest.raises(SystemExit):
        security.check_production(worker=False)


def test_production_rejects_short_keys_and_the_demo_database(monkeypatch):
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings.api, "keys", ["short"])
    monkeypatch.setattr(settings.media, "allowed_hosts", [HOST])
    monkeypatch.setattr(settings.queue, "enabled", True)
    monkeypatch.setattr(settings.database, "demo_url", "postgresql://u:p@127.0.0.1/demo")
    problems = " ".join(security.production_problems(worker=False))
    assert "32 characters" in problems and "DEMO_DATABASE_URL" in problems


def test_production_with_everything_set_starts(monkeypatch):
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings.api, "keys", [KEY])
    monkeypatch.setattr(settings.media, "allowed_hosts", [HOST])
    monkeypatch.setattr(settings.queue, "enabled", True)
    monkeypatch.setattr(settings.database, "url", "postgresql://u:p@db.example.com/postgres")
    assert security.production_problems(worker=False) == []
    assert security.production_problems(worker=True) == []


# ---------- links ----------

def test_only_supabase_storage_paths_are_allowed(queue_on):
    assert allowed_link(URL)
    assert not allowed_link(f"https://{HOST}/auth/v1/user")
    assert not allowed_link(f"https://{HOST}/rest/v1/accounts")


def test_worker_refuses_a_link_the_api_would_not_accept(queue_on, tmp_path):
    with pytest.raises(DownloadError):
        run(download_media("https://evil.example.com/storage/v1/object/public/a.jpg", "image",
                           tmp_path / "x", serving()))


# ---------- health ----------

def test_health_live(client):
    assert client.get("/health/live").json() == {"status": "ok"}


def test_health_ready_reports_queue_and_workers(client, redis_server, monkeypatch):
    fake = fakeredis.FakeAsyncRedis(server=redis_server)
    monkeypatch.setattr("app.main.make_redis", lambda: fake)
    fakeredis.FakeRedis(server=redis_server).set(f"{settings.worker.heartbeat_key}w1", "1")
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["workers_running"] == 1 and r.json()["waiting"] == {"image": 0, "video": 0}


def test_health_ready_is_503_without_redis(client, monkeypatch):
    from redis.asyncio import Redis
    monkeypatch.setattr("app.main.make_redis", lambda: Redis.from_url("redis://127.0.0.1:1/0", socket_connect_timeout=0.2))
    assert client.get("/health/ready").status_code == 503


# ---------- logs ----------

def test_json_log_line_parses():
    record = logging.LogRecord("mmp.test", logging.INFO, __file__, 1, "job %s done", ("abc",), None)
    entry = json.loads(JsonFormatter().format(record))
    assert entry["message"] == "job abc done" and entry["level"] == "INFO" and entry["logger"] == "mmp.test"


# ---------- bad jobs (audit) ----------

def test_unknown_fields_are_ignored():
    job = make_job()
    raw = {k.encode(): str(v).encode() for k, v in job.__dict__.items()} | {b"added_later": b"x"}
    assert job_from(raw).content_id == job.content_id


def test_missing_field_is_a_bad_job():
    with pytest.raises(BadJob):
        job_from({b"content_id": b"x"})


def test_a_bad_entry_is_dropped_and_the_next_job_still_runs(queue_on, parts):
    queue = parts[0]
    good = make_job()

    async def go():
        await queue.ensure_groups()
        await queue.redis.xadd(settings.queue.image_stream, {"content_id": "broken"})
        await queue.enqueue(good)
        delivery = await queue.next_job("w1")
        return delivery, await queue.redis.xlen(settings.queue.image_stream)

    delivery, left = run(go())
    assert delivery.job.content_id == good.content_id and left == 1


# ---------- answers made with a missing model aren't cached (audit) ----------

def test_incomplete_answer_is_not_cached(parts):
    from app.api.schemas import ModerationResponse
    from uuid import uuid4
    store = parts[1]
    key = UploadKey("a" * 64, uuid4(), "image", "post", "v1")
    answer = ModerationResponse(content_id=uuid4(), decision="HUMAN_REVIEW", flags=["guard_unavailable"],
                                processing_ms=1, model_version="v1")

    async def go():
        await store.store(key, answer)
        return await store.get_cached(key)

    assert run(go()) is None


# ---------- part-time worker ----------

def test_drain_mode_stops_when_the_queue_is_empty(queue_on, parts, monkeypatch):
    monkeypatch.setattr(settings.worker, "exit_when_empty", True)
    monkeypatch.setattr(settings.worker, "idle_exit_seconds", 0.05)
    decisions = FakeDecisions()

    async def go():
        await enqueue_all(parts[0], make_job(), make_job())
        await asyncio.wait_for(worker(parts, decisions).run(), timeout=10)
        return await parts[0].redis.keys(f"{settings.worker.heartbeat_key}*")

    beats = run(go())
    assert len(decisions.saved) == 2 and beats == []


def test_drain_mode_waits_for_a_job_in_progress(queue_on, parts, monkeypatch):
    monkeypatch.setattr(settings.worker, "idle_exit_seconds", 0)
    queue = parts[0]

    async def go():
        await enqueue_all(queue, make_job())
        await queue.next_job("other-worker")   # taken, not finished
        w = worker(parts)
        return await w.finished(time() - 1)

    assert run(go()) is False


def test_normal_mode_keeps_running(queue_on, parts, monkeypatch):
    monkeypatch.setattr(settings.worker, "exit_when_empty", False)

    async def go():
        await enqueue_all(parts[0])
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(worker(parts).run(), timeout=0.3)

    run(go())


# ---------- out of memory isn't a bad file (audit) ----------

def test_out_of_memory_is_not_called_a_bad_file(tmp_path):
    from app.validation.image_check import check_image
    path = tmp_path / "a.jpg"
    path.write_bytes(jpeg_bytes())
    with patch("app.validation.image_check.Image.open", side_effect=MemoryError):
        with pytest.raises(MemoryError):
            check_image(path)


# ---------- truncated video (audit) ----------

def test_video_that_decodes_too_few_frames_is_broken(tmp_path, monkeypatch):
    from app.prepare import video
    monkeypatch.setattr(video, "decode", lambda path, duration, times, size, pieces: [])
    with pytest.raises(ValidationFailure):
        video.prepare_video(tmp_path / "v.mp4", 30.0)


# ---------- the link must be the row's own file ----------

@pytest.mark.parametrize("link, row, ok", [
    ("pet-images/3f1c/buddy.jpg", "3f1c/buddy.jpg", True),             # pet_media keeps the path without the bucket
    ("pet-images/3f1c/buddy.jpg", "pet-images/3f1c/buddy.jpg", True),  # or with it
    ("pet-images/3f1c/buddy.jpg", "/3f1c/buddy.jpg", True),
    ("post-media/a b.mp4", "a%20b.mp4", True),                         # stored URL-encoded
    ("pet-images/3f1c/buddy.jpg", None, True),                         # nothing recorded: can't compare
    ("pet-images/3f1c/buddy.jpg", "3f1c/other.jpg", False),
    ("pet-images/x3f1c/buddy.jpg", "3f1c/buddy.jpg", False),           # a suffix of a folder name isn't a match
])
def test_paths_match(link, row, ok):
    from app.jobs.results import paths_match
    assert paths_match(link, row) is ok


def test_a_link_for_another_row_is_dropped_without_writing(queue_on, parts):
    from app.jobs.results import MediaMismatch

    class Mismatched(FakeDecisions):
        def save(self, job, response, sha256, size_bytes, *, check_path=True):
            raise MediaMismatch("not this row's file")

    decisions = Mismatched()

    async def go():
        await enqueue_all(parts[0], make_job())
        await worker(parts, decisions).step()
        return await parts[0].waiting()

    assert run(go()) == {"image": 0, "video": 0} and decisions.saved == []


def test_giving_up_skips_the_path_check(queue_on, parts):
    seen = []

    class Recording(FakeDecisions):
        def save(self, job, response, sha256, size_bytes, *, check_path=True):
            seen.append(check_path)

    async def go():
        await enqueue_all(parts[0], make_job())
        delivery = await parts[0].next_job("w1")
        await worker(parts, Recording()).give_up(delivery)

    run(go())
    assert seen == [False]
