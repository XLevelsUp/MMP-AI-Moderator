import json
import logging
from datetime import datetime, timezone
from uuid import UUID

from app.api.schemas import ModerationResponse
from app.core.config import settings


logger = logging.getLogger("mmp.moderation")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for CloudWatch and other log tools."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_logging() -> None:
    handler = logging.StreamHandler()
    if settings.logging.format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.logging.level)


def log_result(
    response: ModerationResponse,
    *,
    cached: bool = False,
    timings: dict[str, int] | None = None,
) -> None:
    """One line per moderated request: id, decision, reason, time (and time per stage)."""
    reason = "-"
    if response.rejection_code:
        reason = f"{response.rejection_code}:{','.join(response.flags)}"
    stages = ",".join(f"{stage}:{ms}" for stage, ms in (timings or {}).items()) or "-"
    logger.info(
        "content_id=%s decision=%s reason=%s ms=%d cached=%s timings_ms=%s",
        response.content_id,
        response.decision,
        reason,
        response.processing_ms,
        cached,
        stages,
    )


def log_problem(status: int, content_id: UUID, detail: str) -> None:
    """One line for requests that end without a decision (409, 503)."""
    logger.warning("content_id=%s status=%d detail=%s", content_id, status, detail)
