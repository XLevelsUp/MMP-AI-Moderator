"""Production-build test on a GPU machine (used by scripts/colab_build_test.sh).

    python scripts/build_test_run.py --stage-files <test_media dir> <web root>
    python scripts/build_test_run.py --run

--run expects in the environment: MEDIA_HOST, REDIS_URL, DATABASE_URL (moderation_service),
PG_ADMIN_URL (to create the test rows and read the results), HF_HOME, EASYOCR_MODEL_DIR.
It starts the API in production mode, sends every test file, runs the worker until the queue is
empty, then checks the database. Prints a report and saves build_test_report.json.
"""
import json
import os
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

GROUPS = ("pet", "no_pet", "bad")
WEB_PREFIX = Path("storage/v1/object/public/pet-images/buildtest")
API = "http://127.0.0.1:8000"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".webm", ".mpeg", ".mpg"}
STAGED_LIST = Path("/tmp/buildtest_files.json")


def safe_name(group: str, index: int, path: Path) -> str:
    return f"{group}_{index:02d}{path.suffix.lower()}"


def stage_files(src: Path, web_root: Path) -> None:
    """Copy the test files under Supabase-style paths, with plain names (no spaces or commas)."""
    target = web_root / WEB_PREFIX
    target.mkdir(parents=True, exist_ok=True)
    (target / "ping.txt").write_text("ok")
    staged = []
    for group in GROUPS:
        for i, path in enumerate(sorted((src / group).glob("*")), 1):
            if path.is_file():
                name = safe_name(group, i, path)
                shutil.copy(path, target / name)
                staged.append({"group": group, "name": name, "original": path.name,
                               "content_type": "video" if path.suffix.lower() in VIDEO_EXT else "image"})
    STAGED_LIST.write_text(json.dumps(staged, indent=1))
    print(f"staged {len(staged)} files: " + ", ".join(f"{g} {sum(s['group'] == g for s in staged)}" for g in GROUPS))


class GpuSampler(threading.Thread):
    """Highest GPU memory seen while the worker runs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mb = 0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.strip().splitlines()
            if out:
                self.peak_mb = max(self.peak_mb, int(out[0]))
            self.stop.wait(2)


def wait_for(url: str, process: subprocess.Popen, seconds: int, log: Path) -> None:
    import httpx
    deadline = time.time() + seconds
    while time.time() < deadline:
        if process.poll() is not None:
            sys.exit(f"stopped while starting:\n{log.read_text()[-3000:]}")
        try:
            if httpx.get(url, timeout=3).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    sys.exit(f"{url} not ready after {seconds} s")


def run() -> None:
    import httpx
    import psycopg

    host = os.environ["MEDIA_HOST"]
    check_rows = os.environ.get("CHECK_ROW_PATH", "false").lower() == "true"  # the link-to-row check is optional
    staged = json.loads(STAGED_LIST.read_text())
    checks: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")

    # --- test rows: one account and pet, one pet_media row per file, plus one row whose link is wrong
    account, pet = uuid.uuid4(), uuid.uuid4()
    with psycopg.connect(os.environ["PG_ADMIN_URL"], autocommit=True) as db:
        db.execute("insert into identity.accounts (id, auth_user_id) values (%s, %s)", (account, uuid.uuid4()))
        db.execute("insert into pets.pets (id, owner_account_id, name, status) values (%s, %s, 'Build test', 'active')",
                   (pet, account))
        for i, f in enumerate(staged):
            f["media_id"] = str(uuid.uuid4())
            f["url"] = f"https://{host}/{WEB_PREFIX.as_posix()}/{f['name']}"
            db.execute("insert into pets.pet_media (id, pet_id, kind, url, storage_path, display_order) values (%s, %s, %s, %s, %s, %s)",
                       (f["media_id"], pet, f["content_type"], f["url"], f"buildtest/{f['name']}", i))
        mismatch_id = str(uuid.uuid4())
        db.execute("insert into pets.pet_media (id, pet_id, kind, url, storage_path, display_order) values (%s, %s, 'image', 'x', 'buildtest/someone-elses.jpg', 999)",
                   (mismatch_id, pet))

    # --- API in production mode
    key = secrets.token_hex(32)
    api_env = {**os.environ, "APP_ENV": "production", "LOG_FORMAT": "json", "API_KEYS": key,
               "MEDIA_ALLOWED_HOSTS": host, "QUEUE_ENABLED": "true", "DATABASE_URL": ""}
    api_log = Path("/tmp/api.log")
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
                           env=api_env, stdout=api_log.open("w"), stderr=subprocess.STDOUT)
    try:
        wait_for(f"{API}/health/ready", api, 120, api_log)
        check("API starts in production mode", True)

        def send(media_id, url, content_type, headers):
            body = {"media_url": url, "content_type": content_type, "account_id": str(account),
                    "media_type": "pet_media", "media_id": media_id, "pet_id": str(pet)}
            return httpx.post(f"{API}/api/v1/moderate", json=body, headers=headers, timeout=10).status_code

        first = staged[0]
        check("no API key -> 401", send(first["media_id"], first["url"], first["content_type"], {}) == 401)
        check("link on another host -> 422", send(first["media_id"], "https://evil.example.com/storage/v1/object/public/a.jpg",
                                                 "image", {"X-API-Key": key}) == 422)
        codes = [send(f["media_id"], f["url"], f["content_type"], {"X-API-Key": key}) for f in staged]
        check(f"all {len(staged)} uploads queued (202)", all(c == 202 for c in codes), str(sorted(set(codes))))
        # a real file's link sent with another row's id: the worker must drop it
        check("mismatched link queued", send(mismatch_id, staged[0]["url"], staged[0]["content_type"], {"X-API-Key": key}) == 202)
        ready = httpx.get(f"{API}/health/ready", timeout=5).json()
        check("queue shows the waiting uploads", sum(ready["waiting"].values()) == len(staged) + 1, json.dumps(ready["waiting"]))
    finally:
        pass  # the API keeps running while the worker works, as in production

    # --- worker, as on the GPU server
    worker_env = {**os.environ, "APP_ENV": "production", "LOG_FORMAT": "json", "MEDIA_ALLOWED_HOSTS": host,
                  "ML_DEVICE": "cuda", "ML_DTYPE": "float16", "FFMPEG_HWACCEL": "cuda", "GUARD_MODEL": "shieldstral",
                  "WORKER_EXIT_WHEN_EMPTY": "true", "HF_HUB_OFFLINE": "1"}
    worker_log = Path("/tmp/worker.log")
    sampler = GpuSampler()
    sampler.start()
    started = time.time()
    worker = subprocess.Popen([sys.executable, "-m", "app.worker", "--exit-when-empty"], env=worker_env,
                              stdout=worker_log.open("w"), stderr=subprocess.STDOUT)
    try:
        status = worker.wait(timeout=3600)
    except subprocess.TimeoutExpired:
        worker.kill()
        status = "timeout"
    total = time.time() - started
    sampler.stop.set()
    log = worker_log.read_text(errors="replace")
    not_loaded = [line for line in log.splitlines() if "failed to load" in line or "models not loaded" in line]
    check("every model loaded on the GPU", not not_loaded, (not_loaded[0][:300] if not_loaded else ""))
    check("worker stopped by itself when the queue was empty", status == 0, f"status {status}")
    if not_loaded:
        print(log[-3000:])
        sys.exit("A model did not load, so no upload was really checked. See above.")
    ready_line = next((json.loads(line) for line in log.splitlines()
                       if line.startswith("{") and '"logger": "mmp.worker"' in line and " ready" in line), None)
    load_s = None
    if ready_line:
        first_line = next(json.loads(line) for line in log.splitlines() if line.startswith("{"))
        load_s = round(_ts(ready_line["time"]) - _ts(first_line["time"]), 1)
    check("worker logs are JSON", ready_line is not None)
    if check_rows:
        check("mismatched link dropped and logged", "the link is not this row's file" in log)
    ready = httpx.get(f"{API}/health/ready", timeout=5).json()
    check("queue empty afterwards", sum(ready["waiting"].values()) == 0, json.dumps(ready["waiting"]))
    check("worker heartbeat cleared", ready["workers_running"] == 0)
    api.terminate()

    # --- results in the database
    with psycopg.connect(os.environ["PG_ADMIN_URL"]) as db:
        rows = {r[0]: r for r in db.execute(
            "select d.media_id::text, d.decision, d.rejection_code, d.review_status, d.reason_codes, d.processing_ms, "
            "m.moderation_status, d.flags from moderation.decisions d join pets.pet_media m on m.id = d.media_id").fetchall()}
        mismatch = db.execute("select moderation_status, (select count(*) from moderation.decisions where media_id = %s) "
                              "from pets.pet_media where id = %s", (mismatch_id, mismatch_id)).fetchone()
    check(f"a decision row for every upload ({len(staged)})", all(f["media_id"] in rows for f in staged), f"{len(rows)} rows")
    if check_rows:
        check("mismatched row untouched: no decision, status unchanged", mismatch == ("approved", 0), str(mismatch))
    incomplete = [f["name"] for f in staged if f["media_id"] in rows and "check_incomplete" in rows[f["media_id"]][4]]
    check("no check left incomplete", not incomplete, ", ".join(incomplete))
    status_ok = all(rows[f["media_id"]][6] == ("approved" if rows[f["media_id"]][1] == "APPROVED" else "in_review")
                    for f in staged if f["media_id"] in rows)
    check("each media row's status matches its decision", status_ok)
    review_ok = all(rows[f["media_id"]][3] == ("done" if rows[f["media_id"]][1] == "APPROVED" else "pending")
                    for f in staged if f["media_id"] in rows)
    check("review_status: done when approved, pending otherwise", review_ok)

    # --- how the AI did (information, not pass/fail: the benchmark expectation)
    print("\n== decisions")
    print(f"{'file':<14} {'was':<55} {'decision':<13} {'code':<8} {'status':<10} {'ms':>6}")
    for f in staged:
        r = rows.get(f["media_id"])
        if r:
            print(f"{f['name']:<14} {f['original'][:55]:<55} {r[1]:<13} {r[2] or '-':<8} {r[6]:<10} {r[5]:>6}")
    pets = [rows[f["media_id"]] for f in staged if f["group"] == "pet" and f["media_id"] in rows]
    no_pets = [rows[f["media_id"]] for f in staged if f["group"] == "no_pet" and f["media_id"] in rows]
    bads = [rows[f["media_id"]] for f in staged if f["group"] == "bad" and f["media_id"] in rows]
    images = [rows[f["media_id"]][5] for f in staged if f["content_type"] == "image" and f["media_id"] in rows and f["group"] != "bad"]
    videos = [rows[f["media_id"]][5] for f in staged if f["content_type"] == "video" and f["media_id"] in rows and f["group"] != "bad"]

    processed = [r for r in rows.values() if r[5] > 0]
    check("uploads really processed by the models", len(processed) >= len(staged) - 1,
          f"{len(processed)} of {len(staged)} have a processing time")

    def p95(values):
        return sorted(values)[max(0, int(round(0.95 * len(values))) - 1)] if values else None

    report = {
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        "passed": sum(ok for _, ok, _ in checks), "failed": sum(not ok for _, ok, _ in checks),
        "pets_approved": f"{sum(r[1] == 'APPROVED' for r in pets)} of {len(pets)}",
        "no_pets_not_approved": f"{sum(r[1] != 'APPROVED' for r in no_pets)} of {len(no_pets)}",
        "bad_files_mod_008": f"{sum(r[2] == 'MOD_008' for r in bads)} of {len(bads)}",
        "image_ms_p50": statistics.median(images) if images else None, "image_ms_p95": p95(images),
        "video_ms": sorted(videos),
        "model_load_seconds": load_s, "worker_total_seconds": round(total, 1),
        "gpu_peak_memory_mb": sampler.peak_mb,
    }
    Path("/content/build_test_report.json").write_text(json.dumps(report, indent=2))
    print("\n== summary")
    for k, v in report.items():
        if k != "checks":
            print(f"{k}: {v}")
    print(f"\n{report['passed']} checks passed, {report['failed']} failed. Report: /content/build_test_report.json")
    print("Logs: /tmp/api.log, /tmp/worker.log")
    sys.exit(1 if report["failed"] else 0)


def _ts(value: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(value).timestamp()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--stage-files":
        stage_files(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1:] == ["--run"]:
        run()
    else:
        sys.exit(__doc__)
