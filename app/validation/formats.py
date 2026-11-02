from pathlib import Path

from app.core.config import settings
from app.validation.errors import ValidationFailure


# All signatures below fit in the first 64 bytes of a file (WebM's doc type sits furthest in).
HEADER_BYTES = 64

HEIC_BRANDS = {b"heic", b"heix", b"mif1", b"msf1"}
HEIF_BRANDS = {b"heif", b"heim", b"heis", b"hevc", b"hevx"}
MP4_BRANDS = {b"isom", b"iso2", b"iso3", b"iso4", b"iso5", b"iso6", b"mp41", b"mp42", b"avc1", b"M4V ", b"dash"}
MOV_BRANDS = {b"qt  "}
EBML = b"\x1a\x45\xdf\xa3"              # WebM and Matroska; only WebM names itself in the header
MPEG_STARTS = (b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3")  # program stream, video stream


def detect_format(header: bytes) -> str | None:
    """Return the real format from the file's first bytes, or None if unknown."""
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if header.startswith(EBML) and b"webm" in header:
        return "webm"
    if header.startswith(MPEG_STARTS):
        return "mpeg"
    if header[:4] == b"RIFF":
        if header[8:12] == b"WEBP":
            return "webp"
        if header[8:12] == b"AVI ":
            return "avi"
    if header[4:8] == b"ftyp":
        brand = header[8:12]
        if brand in HEIC_BRANDS:
            return "heic"
        if brand in HEIF_BRANDS:
            return "heif"
        if brand in MP4_BRANDS:
            return "mp4"
        if brand in MOV_BRANDS:
            return "mov"
    return None


def check_format(path: Path, content_type: str) -> str:
    """Detect the real format and make sure it matches the declared content_type."""
    with path.open("rb") as file:
        header = file.read(HEADER_BYTES)

    file_format = detect_format(header)
    allowed = settings.allowed_formats

    if file_format in allowed.image:
        real_type = "image"
    elif file_format in allowed.video:
        real_type = "video"
    else:
        raise ValidationFailure("unsupported_format")

    if real_type != content_type:
        raise ValidationFailure("type_mismatch")

    return file_format
