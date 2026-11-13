"""API keys and the start-up checks a production server must pass."""
import hmac
import logging

from fastapi import HTTPException, Request

from app.core.config import settings


logger = logging.getLogger("mmp.security")


def key_is_valid(given: str | None) -> bool:
    """Constant-time comparison against every configured key, so timing can't reveal one."""
    if not given:
        return False
    return any(hmac.compare_digest(given.encode(), key.encode()) for key in settings.api.keys)


async def require_api_key(request: Request) -> None:
    """FastAPI dependency for the moderation routes. With no keys configured (development
    only: production refuses to start that way), every request is let through."""
    if not settings.api.keys:
        return
    if not key_is_valid(request.headers.get(settings.api.key_header)):
        raise HTTPException(status_code=401, detail=f"Missing or wrong {settings.api.key_header} header.")


def production_problems(*, worker: bool) -> list[str]:
    """What stops this process from running safely in production. Empty in development."""
    if settings.environment != "production":
        return []
    problems = []
    if not settings.media.allowed_hosts:
        problems.append("MEDIA_ALLOWED_HOSTS is empty: no media link would be accepted")
    if not settings.media.require_https:
        problems.append("media.require_https is off")
    if worker:
        if not settings.database.url:
            problems.append("DATABASE_URL is not set")
    else:
        if not settings.api.keys:
            problems.append("API_KEYS is not set: anyone could send jobs")
        if any(len(key) < 32 for key in settings.api.keys):
            problems.append("every API key must be at least 32 characters")
        if not settings.queue.enabled:
            problems.append("QUEUE_ENABLED must be true in production")
    if settings.database.demo_url:
        problems.append("DEMO_DATABASE_URL must not be set in production")
    return problems


def check_production(*, worker: bool) -> None:
    problems = production_problems(worker=worker)
    if problems:
        raise SystemExit("Refusing to start in production: " + "; ".join(problems))
    if settings.environment == "development" and not worker and not settings.api.keys:
        logger.warning("No API_KEYS set: the moderation API accepts every request (development only)")
