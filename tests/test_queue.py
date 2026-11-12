import asyncio
import io
import shutil
import subprocess
from time import time
from types import SimpleNamespace
from uuid import uuid4

import fakeredis
import httpx
import pytest
from PIL import Image

from app.api.schemas import BreedCheck, ModerationResponse
from app.cache.idempotency import IdempotencyStore
from app.core.config import settings
from app.jobs.queue import Job, JobQueue, QueueFull
from app.jobs.results import failure_fallback, outcome_for, reason_codes, storage_path_of
from app.pipeline import orchestrator
from app.prepare import prepare_media
from app.prepare.video import prepare_animated
from app.validation.csam import CsamCheckError
from app.validation.errors import ProcessingTimeout, ValidationFailure
from app.validation.image_check import check_image
from app.validation.result import ValidationResult
from app.worker import Worker


USER = "3f2b8c1e-0000-4000-8000-000000000001"
HOST = "media.example.com"
URL = f"https://{HOST}/storage/v1/object/public/pet-images/a/b.jpg"


def jpeg_bytes(color="orange") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buffer, "JPEG")
    return buffer.getvalue()


class FakeDecisions:
    def __init__(self, fail=False):
        self.saved = []
        self.fail = fail

    def save(self, job, response, sha256, size_bytes):
        if self.fail:
            raise RuntimeError("database down")
        self.saved.append(SimpleNamespace(job=job, response=response, sha256=sha256, size=size_bytes))


def serving(content=None, status=200):
    return httpx.MockTransport(lambda request: httpx.Response(status, content=jpeg_bytes() if content is None else content))


@pytest.fixture
def queue_on(monkeypatch):
    monkeypatch.setattr(settings.queue, "enabled", True)
    monkeypatch.setattr(settings.queue, "poll_seconds", 0.01)
    monkeypatch.setattr(settings.media, "allowed_hosts", [HOST])


@pytest.fixture
def parts(redis_server):
    redis = fakeredis.FakeAsyncRedis(server=redis_server)
    return JobQueue(redis), IdempotencyStore(redis)


def worker(parts, decisions=None, transport=None, name="w1"):
    return Worker(parts[0], parts[1], decisions or FakeDecisions(), name, transport or serving())


def make_job(content_type="image", media_type="pet_media", media_id=None) -> Job:
    return Job(str(uuid4()), USER, "", media_type, media_id or str(uuid4()), URL, content_type,
               "profile" if media_type == "pet_profile_photo" else "post", "v1", time())


def body(**changes):
    return {"media_url": URL, "content_type": "image", "account_id": USER, "media_type": "pet_media",
            "media_id": str(uuid4()), **changes}


def run(coro):
    return asyncio.run(coro)


def uploads(temp_dir):
    return list(temp_dir.glob("*.upload"))


async def enqueue_all(queue, *jobs):
    await queue.ensure_groups()
    for job in jobs:
        await queue.enqueue(job)


# ---------- API ----------

def test_request_is_queued_with_202_at_once(client, queue_on, parts):
    run(parts[0].ensure_groups())
    r = client.post("/api/v1/moderate", json=body())
    assert r.status_code == 202
    assert r.content == b""                     # no response data: the result goes to the database
    assert run(parts[0].waiting()) == {"image": 1, "video": 0}


@pytest.mark.parametrize("url", [
    f"http://{HOST}/a.jpg", "https://evil.example.com/a.jpg", "https://169.254.169.254/latest/meta-data",
    f"https://{HOST}.evil.com/a.jpg", "not a link", "ftp://media.example.com/a.jpg",
])
def test_only_links_to_the_apps_storage(client, queue_on, parts, url):
    assert client.post("/api/v1/moderate", json=body(media_url=url)).status_code == 422
    assert run(parts[0].waiting()) == {"image": 0, "video": 0}


@pytest.mark.parametrize("changes", [
    {"context": "profile"},                                        # pet_media goes with post
    {"media_type": "pet_profile_photo", "context": "post"},
    {"context": "story"}, {"media_type": "chat_image"}, {"content_type": "gif"},
    {"account_id": "nope"}, {"media_id": "nope"},
])
def test_bad_request_is_422(client, queue_on, changes):
    assert client.post("/api/v1/moderate", json=body(**changes)).status_code == 422


def test_full_queue_is_503_with_retry_after(client, queue_on, parts, monkeypatch):
    monkeypatch.setattr(settings.queue, "image_max_waiting", 1)
    run(parts[0].ensure_groups())
    assert client.post("/api/v1/moderate", json=body()).status_code == 202
    r = client.post("/api/v1/moderate", json=body())
    assert r.status_code == 503
    assert r.headers["retry-after"] == str(settings.queue.retry_after_seconds)


def test_there_is_no_result_call(client, queue_on):
    assert client.get(f"/api/v1/moderate/{uuid4()}").status_code in (404, 405)


def test_upload_route_is_off_with_the_queue(client, queue_on):
    r = client.post("/api/v1/moderate/upload", files={"file": ("x.jpg", jpeg_bytes(), "image/jpeg")},
                    data={"content_type": "image", "user_id": USER})
    assert r.status_code == 404


# ---------- what gets written ----------

def answer(decision, code=None, flags=()):
    return ModerationResponse(content_id=uuid4(), decision=decision, rejection_code=code, flags=list(flags),
                              processing_ms=1, model_version="v")


@pytest.mark.parametrize("response, status, code, review", [
    (answer("APPROVED"), "approved", None, "done"),
    (answer("HUMAN_REVIEW"), "in_review", None, "pending"),
    (answer("REJECTED", "MOD_001", ["no_pet_detected"]), "in_review", None, "pending"),   # a moderator decides
    (answer("REJECTED", "MOD_002", ["welfare_abuse"]), "in_review", None, "pending"),
    (answer("REJECTED", "MOD_008", ["corrupt_file"]), "in_review", None, "pending"),   # a moderator decides
    (answer("REJECTED", "MOD_004", ["quarantined"]), "in_review", None, "pending"),    # decided, never displayed
])
def test_status_for_each_answer(response, status, code, review):
    outcome = outcome_for(response)
    assert (outcome.moderation_status, outcome.rejection_code, outcome.review_status) == (status, code, review)


def test_breed_result_never_changes_the_status():
    unmatched = answer("APPROVED").model_copy(update={"breed_check": BreedCheck(
        status="unmatched", match_score=12.0, detected_breed_id=uuid4(), detected_breed_name="Beagle", detected_score=91.0)})
    assert outcome_for(unmatched).moderation_status == "approved"


def test_breed_check_says_not_checked_until_it_is_built():
    assert answer("APPROVED").breed_check.status == "not_checked"


def test_request_carries_the_claimed_breed(client, queue_on, parts):
    breed = str(uuid4())
    run(parts[0].ensure_groups())
    r = client.post("/api/v1/moderate", json=body(declared_breed_id=breed, declared_is_cross_breed=False))
    assert r.status_code == 202

    async def read():
        return await parts[0].next_job("w1")

    job = run(read()).job
    assert job.declared_breed_id == breed and job.declared_is_cross_breed == "false"


def test_claimed_breed_is_optional(client, queue_on, parts):
    run(parts[0].ensure_groups())
    assert client.post("/api/v1/moderate", json=body()).status_code == 202

    async def read():
        return await parts[0].next_job("w1")

    job = run(read()).job
    assert job.declared_breed_id == "" and job.declared_is_cross_breed == ""


def test_reason_codes_leave_out_status_flags():
    flags = ["validation_passed", "gate1_passed", "welfare_distress", "gate2_review", "nsfw_unavailable"]
    assert reason_codes(answer("HUMAN_REVIEW", flags=flags)) == ["welfare_distress", "check_incomplete"]


@pytest.mark.parametrize("url, path", [
    (URL, "pet-images/a/b.jpg"),
    (f"https://{HOST}/storage/v1/object/sign/post-media/a%20b.mp4?token=secret", "post-media/a b.mp4"),
    (f"https://{HOST}/other/c.jpg?token=secret", "other/c.jpg"),
])
def test_storage_path_has_no_token(url, path):
    assert storage_path_of(url) == path


# ---------- worker ----------

def test_nothing_waiting(parts):
    run(parts[0].ensure_groups())
    assert run(worker(parts).step()) is False


def test_worker_writes_the_result_and_cleans_up(queue_on, parts, temp_dir):
    decisions = FakeDecisions()
    job = make_job()

    async def go():
        await enqueue_all(parts[0], job)
        await worker(parts, decisions).step()
        return await parts[0].waiting()

    waiting = run(go())
    assert len(decisions.saved) == 1
    saved = decisions.saved[0]
    assert saved.job == job and len(saved.sha256) == 64 and saved.size == len(jpeg_bytes())
    assert waiting == {"image": 0, "video": 0}
    assert uploads(temp_dir) == []


@pytest.mark.parametrize("content, status, flag", [(b"", 200, "empty_file"), (b"not an image", 200, "unsupported_format")])
def test_bad_file_goes_to_a_moderator(queue_on, parts, content, status, flag):
    decisions = FakeDecisions()

    async def go():
        await enqueue_all(parts[0], make_job())
        await worker(parts, decisions, serving(content, status)).step()

    run(go())
    response = decisions.saved[0].response
    assert response.rejection_code == "MOD_008" and flag in response.flags
    assert (outcome_for(response).moderation_status, outcome_for(response).review_status) == ("in_review", "pending")


def test_file_over_the_limit_stops_the_download(queue_on, parts, monkeypatch):
    monkeypatch.setattr(settings.limits, "image_max_bytes", 10)
    decisions = FakeDecisions()

    async def go():
        await enqueue_all(parts[0], make_job())
        await worker(parts, decisions).step()

    run(go())
    assert "file_too_large" in decisions.saved[0].response.flags


def test_sent_again_reuses_the_check_but_still_writes(queue_on, parts, monkeypatch):
    calls = []
    real = orchestrator.run_pipeline
    monkeypatch.setattr("app.worker.run_pipeline", lambda ctx: calls.append(1) or real(ctx))
    decisions = FakeDecisions()
    media = str(uuid4())
    first, second, other = make_job(media_id=media), make_job(media_id=media), make_job()

    async def go():
        await enqueue_all(parts[0], first, second, other)
        w = worker(parts, decisions)
        for _ in range(3):
            await w.step()

    run(go())
    assert len(calls) == 2                     # the third file is the same, but in another row
    assert len(decisions.saved) == 3
    assert str(decisions.saved[1].response.content_id) == second.content_id


@pytest.mark.parametrize("how", ["link", "pipeline", "database"])
def test_failure_keeps_the_job_for_a_retry(queue_on, parts, temp_dir, monkeypatch, how):
    def boom(ctx):
        raise CsamCheckError("down") if how == "pipeline" else ProcessingTimeout("slow")

    if how == "pipeline":
        monkeypatch.setattr("app.worker.run_pipeline", boom)
    decisions = FakeDecisions(fail=how == "database")
    job = make_job()
    transport = serving(status=404) if how == "link" else serving()

    async def go():
        await enqueue_all(parts[0], job)
        await worker(parts, decisions, transport).step()
        return await parts[0].waiting()

    waiting = run(go())
    assert waiting["image"] == 1
    assert decisions.saved == []
    assert uploads(temp_dir) == []


def test_a_failing_job_goes_to_a_moderator_after_the_retries(queue_on, parts, monkeypatch):
    monkeypatch.setattr(settings.queue, "reclaim_idle_seconds", 0)
    monkeypatch.setattr(settings.queue, "reclaim_every_seconds", 1)
    monkeypatch.setattr(settings.queue, "max_attempts", 2)
    decisions = FakeDecisions()
    job = make_job()

    async def go():
        await enqueue_all(parts[0], job)
        w = worker(parts, decisions, serving(status=404))
        await w.step()                         # attempt 1: fails
        for _ in range(2):
            w.last_reclaim = 0
            await w.step()                     # attempt 2 fails, then it gives up
        return await parts[0].waiting()

    waiting = run(go())
    assert waiting == {"image": 0, "video": 0}
    assert len(decisions.saved) == 1
    saved = decisions.saved[0]
    assert saved.sha256 is None and saved.size == 0 and saved.response.decision == "HUMAN_REVIEW"
    assert saved.response.flags == ["check_incomplete"]
    assert reason_codes(saved.response) == ["check_incomplete"]     # how a zero row is told apart
    assert outcome_for(saved.response).moderation_status == "in_review"
    r = saved.response
    assert r.pet_score == 0 and r.composite_score == 0 and r.ai_score == 0 and r.ai_generated is False
    assert set(r.dimension_scores.model_dump().values()) == {0.0}
    assert r.breed_check.match_score == 0 and r.breed_check.detected_score == 0
    assert r.breed_check.status == "not_checked"


def test_a_model_that_failed_to_load_writes_zeros_too(queue_on, parts, monkeypatch):
    monkeypatch.setattr("app.worker.run_pipeline", lambda ctx: answer("HUMAN_REVIEW", flags=["gate1_unavailable"]))
    decisions = FakeDecisions()

    async def go():
        await enqueue_all(parts[0], make_job())
        await worker(parts, decisions).step()

    run(go())
    saved = decisions.saved[0].response
    assert saved.pet_score == 0 and outcome_for(saved).moderation_status == "in_review"
    assert reason_codes(saved) == ["check_incomplete"]


def test_a_real_answer_is_not_zeroed(queue_on, parts):
    decisions = FakeDecisions()

    async def go():
        await enqueue_all(parts[0], make_job())
        await worker(parts, decisions, serving(b"not an image")).step()   # a bad file is a result, not an error

    run(go())
    saved = decisions.saved[0].response
    assert saved.pet_score is None and saved.dimension_scores is None
    assert saved.rejection_code == "MOD_008"


def test_fallback_keeps_who_and_what():
    original = answer("HUMAN_REVIEW", flags=["check_incomplete"])
    fallback = failure_fallback(original)
    assert fallback.content_id == original.content_id and fallback.model_version == original.model_version


def test_a_dead_workers_job_is_taken_over(queue_on, parts, monkeypatch):
    monkeypatch.setattr(settings.queue, "reclaim_idle_seconds", 0)
    decisions = FakeDecisions()
    job = make_job()

    async def go():
        await enqueue_all(parts[0], job)
        await parts[0].next_job("dead")        # handed out, never finished
        assert await worker(parts, decisions, name="alive").step() is False   # reclaim did the job
        return await parts[0].waiting()

    assert run(go()) == {"image": 0, "video": 0}
    assert decisions.saved[0].job == job


def test_nothing_is_lost_while_the_database_is_down(queue_on, parts, monkeypatch):
    monkeypatch.setattr(settings.queue, "reclaim_idle_seconds", 0)
    monkeypatch.setattr(settings.queue, "max_attempts", 1)
    job = make_job()

    async def go():
        await enqueue_all(parts[0], job)
        await parts[0].next_job("dead")
        await worker(parts, FakeDecisions(fail=True), name="alive").step()   # can't even record giving up
        return await parts[0].waiting()

    assert run(go())["image"] == 1


def test_quarantined_evidence_is_not_deleted(queue_on, parts, temp_dir, monkeypatch):
    kept = temp_dir / "evidence.bin"
    real = orchestrator.run_pipeline

    def quarantining(ctx):
        response = real(ctx)
        kept.write_bytes(b"evidence")
        ctx.path = kept
        return response

    monkeypatch.setattr("app.worker.run_pipeline", quarantining)

    async def go():
        await enqueue_all(parts[0], make_job())
        await worker(parts).step()

    run(go())
    assert kept.exists()
    assert uploads(temp_dir) == []


def test_photos_go_before_videos(queue_on, parts):
    video, photo = make_job("video"), make_job()

    async def go():
        await enqueue_all(parts[0], video, photo)
        return await parts[0].next_job("w1")

    assert run(go()).job == photo


def test_a_video_waits_for_a_limited_number_of_photos(queue_on, parts, monkeypatch):
    monkeypatch.setattr(settings.queue, "images_per_video", 2)
    decisions = FakeDecisions()

    async def go():
        await enqueue_all(parts[0], make_job("video"), make_job(), make_job(), make_job())
        w = worker(parts, decisions)
        for _ in range(4):
            await w.step()

    run(go())
    assert [s.job.content_type for s in decisions.saved] == ["image", "image", "video", "image"]


def test_two_workers_never_get_the_same_job(queue_on, parts):
    async def go():
        await enqueue_all(parts[0], make_job(), make_job())
        return await parts[0].next_job("w1"), await parts[0].next_job("w2"), await parts[0].next_job("w3")

    a, b, c = run(go())
    assert a.job != b.job and c is None


def test_enqueue_refuses_when_full(queue_on, parts, monkeypatch):
    monkeypatch.setattr(settings.queue, "video_max_waiting", 1)

    async def go():
        await enqueue_all(parts[0], make_job("video"), make_job())     # photos have their own limit
        with pytest.raises(QueueFull):
            await parts[0].enqueue(make_job("video"))

    run(go())


def demo_post(client):
    return client.post("/api/v1/demo/moderate", files={"file": ("x.jpg", jpeg_bytes(), "image/jpeg")}, data={"content_type": "image"})


def test_database_preview_is_off_by_default(client, monkeypatch):
    monkeypatch.setattr(settings.database, "demo_url", None)
    assert demo_post(client).status_code == 404
    assert client.get("/api/v1/demo/decisions").status_code == 404


@pytest.mark.parametrize("url", [
    "postgresql://u:p@db.abcdefgh.supabase.co:5432/postgres",
    "postgresql://u:p@aws-0-ap.pooler.supabase.com:6543/postgres",
    "postgresql://u:p@127.0.0.1.evil.com/postgres",
])
def test_database_preview_refuses_any_remote_database(client, monkeypatch, url):
    monkeypatch.setattr(settings.database, "demo_url", url)
    assert demo_post(client).status_code == 403
    assert client.get("/api/v1/demo/decisions").status_code == 403


def test_worker_needs_a_database(monkeypatch):
    from app import worker as module

    monkeypatch.setattr(settings.database, "url", None)
    with pytest.raises(SystemExit):
        module.main()

# ---------- the file types the storage buckets accept ----------

def post_file(client, data, content_type="image"):
    return client.post("/api/v1/moderate/upload", files={"file": ("f", data, "application/octet-stream")},
                       data={"content_type": content_type, "user_id": USER})


def gif_bytes(frames=1, transparent=False) -> bytes:
    pictures = [Image.new("P", (64, 64), i) for i in range(frames)]
    buffer = io.BytesIO()
    extra = {"transparency": 0} if transparent else {}
    pictures[0].save(buffer, "GIF", save_all=frames > 1, append_images=pictures[1:], **extra)
    return buffer.getvalue()


@pytest.mark.parametrize("data", [gif_bytes(), gif_bytes(frames=3), gif_bytes(transparent=True)])
def test_gif_is_accepted(client, data):
    r = post_file(client, data)
    assert r.status_code == 200
    assert r.json()["rejection_code"] is None


COLORS = ("red", "lime", "blue")


def animated(path, kind, ms=1000, colors=COLORS):
    pictures = [Image.new("RGB", (64, 64), c) for c in colors]
    pictures[0].save(path, kind, save_all=True, append_images=pictures[1:], duration=ms, **({"lossless": True} if kind == "WEBP" else {}))
    return path


def strongest_channel(frame):
    pixel = frame.detector.convert("RGB").getpixel((32, 32))
    return pixel.index(max(pixel))


@pytest.mark.parametrize("name, kind", [("a.gif", "GIF"), ("a.webp", "WEBP"), ("a.png", "PNG")])
def test_animated_images_are_read_like_a_video(tmp_path, name, kind):
    path = animated(tmp_path / name, kind)
    info = check_image(path)
    assert info.animated and info.duration_seconds == 3.0

    frames = prepare_media(path, ValidationResult(file_format=kind.lower(), image=info, metadata=None, phash=None)).frames
    assert {strongest_channel(f) for f in frames} == {0, 1, 2}   # all three colours, not just the first
    assert frames[0].timestamp == 0.0 and frames[-1].timestamp == 2.0


def test_a_long_gif_keeps_the_frame_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.prepare, "max_video_frames", 2)
    path = animated(tmp_path / "a.gif", "GIF")
    info = check_image(path)
    assert len(prepare_animated(path, info.duration_seconds).frames) == 2


def test_gif_with_no_delay_plays_at_browser_speed(tmp_path):
    path = animated(tmp_path / "a.gif", "GIF", ms=0)
    assert check_image(path).duration_seconds == 0.3


def test_gif_over_the_time_limit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.limits, "max_video_duration_seconds", 2)
    monkeypatch.setattr(settings.limits, "duration_tolerance_seconds", 0)
    with pytest.raises(ValidationFailure) as exc:
        check_image(animated(tmp_path / "a.gif", "GIF"))
    assert exc.value.flag == "duration_exceeded"


def test_gif_with_too_many_frames_is_refused_before_decoding(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.limits, "animated_max_total_pixels", 64 * 64 * 2)
    with pytest.raises(ValidationFailure) as exc:
        check_image(animated(tmp_path / "a.gif", "GIF"))
    assert exc.value.flag == "image_too_large_pixels"


def test_gif_broken_after_the_first_frame_is_corrupt(tmp_path):
    path = animated(tmp_path / "a.gif", "GIF")
    data = path.read_bytes()
    path.write_bytes(data[:len(data) - 12])
    with pytest.raises(ValidationFailure) as exc:
        check_image(path)
    assert exc.value.flag == "corrupt_file"


@pytest.mark.parametrize("content_type, moving, expected", [
    ("video", False, True), ("image", True, True), ("image", False, False),
])
def test_judged_as_video(content_type, moving, expected):
    image = None if content_type == "video" else SimpleNamespace(animated=moving)
    ctx = SimpleNamespace(content_type=content_type, validation=SimpleNamespace(image=image))
    assert orchestrator.judged_as_video(ctx) is expected


def test_heif_is_accepted(client):
    from pillow_heif import from_pillow

    buffer = io.BytesIO()
    from_pillow(Image.new("RGB", (64, 64), "orange")).save(buffer, format="HEIF")
    r = post_file(client, buffer.getvalue())
    assert r.status_code == 200
    assert r.json()["rejection_code"] is None


@pytest.mark.parametrize("name", ["webm", "mpeg"])
def test_new_video_types_are_accepted(client, videos, name):
    r = post_file(client, videos[name].read_bytes(), "video")
    assert r.status_code == 200
    assert r.json()["rejection_code"] is None


def recorded_webm(seconds) -> bytes:
    """Written to a pipe, like a browser's recording: no duration in the header."""
    done = subprocess.run(
        [shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=160x120:rate=5",
         "-pix_fmt", "yuv420p", "-c:v", "libvpx", "-b:v", "100k", "-f", "webm", "-"],
        capture_output=True, check=True, timeout=120,
    )
    return done.stdout


def test_webm_without_a_duration_is_measured(client, videos):
    r = post_file(client, recorded_webm(2), "video")
    assert r.json()["rejection_code"] is None
    assert "corrupt_file" not in r.json()["flags"]


def test_webm_without_a_duration_still_has_the_time_limit(client, videos, monkeypatch):
    monkeypatch.setattr(settings.limits, "max_video_duration_seconds", 1)
    monkeypatch.setattr(settings.limits, "duration_tolerance_seconds", 0)
    r = post_file(client, recorded_webm(3), "video")
    assert r.json()["rejection_code"] == "MOD_008"
    assert "duration_exceeded" in r.json()["flags"]


def test_matroska_is_still_refused(client, videos):
    r = post_file(client, videos["mkv"].read_bytes(), "video")
    assert r.json()["rejection_code"] == "MOD_008"
    assert "unsupported_format" in r.json()["flags"]
