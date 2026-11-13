import tempfile

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.api.demo import router as demo_router
from app.api.routes import router
from app.core.config import settings
from app.core.versioning import current_model_version
from app.core.logging import setup_logging
from app.validation.csam import get_csam_checker

from contextlib import asynccontextmanager

from fastapi.concurrency import run_in_threadpool

from redis.exceptions import RedisError

from app.api.security import check_production
from app.cache.idempotency import make_redis
from app.ml.registry import load_models, registry


setup_logging()
check_production(worker=False)
get_csam_checker()  # logs a loud warning while the placeholder is in use


# Keep FastAPI's own upload buffers on D: (rule 9).
settings.temp_dir.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(settings.temp_dir)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.queue.enabled:  # with the queue, the GPU workers hold the models
        await run_in_threadpool(load_models)  # once, before the first request
    yield

app = FastAPI(title="MeetMyPets Moderation", lifespan=lifespan,
              docs_url=None if settings.environment == "production" else "/docs",
              redoc_url=None, openapi_url=None if settings.environment == "production" else "/openapi.json")
app.include_router(router)
if settings.environment != "production":
    app.include_router(demo_router)  # the local test page's database preview


@app.get("/health/live")
def live():
    """The process answers. For the load balancer and the container restart policy."""
    return {"status": "ok"}


@app.get("/health/ready")
async def ready():
    """Ready to take uploads: Redis answers. Also reports the queue and the workers running."""
    redis = make_redis()
    try:
        await redis.ping()
        q = settings.queue
        waiting = {"image": await redis.xlen(q.image_stream), "video": await redis.xlen(q.video_stream)}
        workers = len([key async for key in redis.scan_iter(f"{settings.worker.heartbeat_key}*")])
    except RedisError as exc:
        return JSONResponse(status_code=503, content={"status": "unavailable", "redis": str(exc)})
    finally:
        await redis.aclose()
    return {"status": "ok", "queue_enabled": q.enabled, "waiting": waiting, "workers_running": workers}



@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": settings.ml.device,
        "models": dict(registry.status),
        "model_version": current_model_version(),
        "gate1_enforce": settings.gate1.enforce,
    }

