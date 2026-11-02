import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import httpx
from fastapi import UploadFile

from app.core.config import settings
from app.validation.errors import ValidationFailure


@dataclass
class IntakeResult:
    path: Path
    size_bytes: int
    sha256: str


def max_bytes_for(content_type: str) -> int:
    if content_type == "video":
        return settings.limits.video_max_bytes
    return settings.limits.image_max_bytes


async def save_upload(
    file: UploadFile,
    content_type: str,
    content_id: UUID,
) -> IntakeResult:
    """Stream the upload to disk in chunks, hashing and counting as we go."""
    temp_dir = settings.temp_dir
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{content_id}.upload"

    limit = max_bytes_for(content_type)
    chunk_size = settings.intake.chunk_size_bytes
    digest = hashlib.sha256()
    size = 0

    try:
        with path.open("wb") as out:
            while chunk := await file.read(chunk_size):
                size += len(chunk)
                if size > limit:
                    raise ValidationFailure("file_too_large")
                digest.update(chunk)
                out.write(chunk)

        if size == 0:
            raise ValidationFailure("empty_file")
    except BaseException:
        remove_temp_file(path)
        raise

    return IntakeResult(path=path, size_bytes=size, sha256=digest.hexdigest())


class DownloadError(Exception):
    """The link didn't deliver the file. Not a decision: the job is retried."""


async def download_media(url: str, content_type: str, path: Path, transport=None) -> IntakeResult:
    """Stream the app's file to disk, hashing and counting as we go; stop as soon as it is too big."""
    m = settings.media
    limit = max_bytes_for(content_type)
    digest = hashlib.sha256()
    size = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(m.download_timeout_seconds, connect=m.connect_timeout_seconds)
    # Redirects stay off: a link on an allowed host must not lead somewhere else.
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, transport=transport) as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise DownloadError(f"media link answered {response.status_code}")
            with path.open("wb") as out:
                async for chunk in response.aiter_bytes(settings.intake.chunk_size_bytes):
                    size += len(chunk)
                    if size > limit:
                        raise ValidationFailure("file_too_large")
                    digest.update(chunk)
                    out.write(chunk)
    if size == 0:
        raise ValidationFailure("empty_file")
    return IntakeResult(path=path, size_bytes=size, sha256=digest.hexdigest())


def remove_temp_file(path: Path) -> None:
    path.unlink(missing_ok=True)
