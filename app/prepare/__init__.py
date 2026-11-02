from pathlib import Path

from app.prepare.images import prepare_image
from app.prepare.media import PreparedMedia
from app.prepare.video import prepare_animated, prepare_video
from app.validation.result import ValidationResult


def prepare_media(path: Path, validation: ValidationResult) -> PreparedMedia:
    """Graph Step 2: turn a validated file into upright RGB pictures at model sizes."""
    if validation.image is not None:
        if validation.image.animated:
            return prepare_animated(path, validation.image.duration_seconds)
        return prepare_image(path)
    video = validation.video
    return prepare_video(path, video.duration_seconds, video.width, video.height)
