"""The service's one job: write each result to the database. Moderators and the app do the rest."""
import json
import logging
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

import psycopg

from app.api.schemas import BreedCheck, DimensionScores, ModerationResponse
from app.core.config import settings
from app.jobs.queue import Job


logger = logging.getLogger("mmp.results")

# Flags that explain a decision and don't come from gate2.signals. The rest only say what ran.
FILE_REASONS = {
    "unsupported_format", "type_mismatch", "file_too_large", "empty_file", "image_too_small",
    "image_too_large_pixels", "corrupt_file", "no_video_stream", "duration_exceeded",
}
OTHER_REASONS = {"quarantined", "known_bad_image", "no_pet_detected", "gate1_borderline", "gate1_head_doubt",
                 "wild_context", "check_incomplete"}
INCOMPLETE_FLAGS = {"gate1_unavailable", "nsfw_unavailable", "guard_unavailable", "gate2_incomplete"}


@dataclass(frozen=True)
class Outcome:
    moderation_status: str
    rejection_code: str | None  # only when the status is final
    review_status: str          # pending = in the moderators' queue; done = nothing for a moderator to do


def outcome_for(response: ModerationResponse) -> Outcome:
    """Only an AI approval is final. Everything else waits for a moderator (decided 12 Nov 2026),
    bad files (MOD_008) and hash-list matches included. A hash-list match's file must never be
    displayed: the panel shows its details only."""
    if response.decision == "APPROVED":
        return Outcome("approved", None, "done")
    return Outcome("in_review", None, "pending")


def failure_fallback(response: ModerationResponse) -> ModerationResponse:
    """Decided 2026-11-03: a check that failed writes 0 in every score. These zeros are NOT results:
    the reason code check_incomplete marks such a row, so filter on it before using scores."""
    zero_dimensions = DimensionScores(**dict.fromkeys(DimensionScores.model_fields, 0.0))
    return response.model_copy(update={
        "pet_score": 0.0, "composite_score": 0.0, "ai_generated": False, "ai_score": 0.0,
        "dimension_scores": zero_dimensions, "breed_check": BreedCheck(match_score=0.0, detected_score=0.0),
    })


def reason_codes(response: ModerationResponse) -> list[str]:
    known = FILE_REASONS | OTHER_REASONS | set(settings.gate2.signals)
    codes = [flag for flag in response.flags if flag in known]
    if INCOMPLETE_FLAGS.intersection(response.flags) and "check_incomplete" not in codes:
        codes.append("check_incomplete")
    if response.rejection_code == settings.gate2.low_score_code and not codes:
        codes.append("low_composite")
    return codes


def storage_path_of(media_url: str) -> str:
    """Bucket and path, without the host or any token. Supabase: /storage/v1/object/<kind>/<bucket>/<path>."""
    parts = unquote(urlsplit(media_url).path).strip("/").split("/")
    if parts[:3] == ["storage", "v1", "object"] and len(parts) > 4:
        return "/".join(parts[4:])
    return "/".join(parts)


INSERT_DECISION = """
insert into moderation.decisions (
  id, account_id, pet_id, media_type, media_id, storage_path, media_sha256, media_size_bytes,
  content_type, context, decision, rejection_code, hard_reject, pet_score, composite_score,
  dimension_scores, ai_generated, ai_score, reason_codes, flags, species_detected, detected_objects,
  processing_ms, model_version, review_status, declared_breed_id, declared_is_cross_breed, breed_status,
  breed_match_score, detected_breed_id, detected_breed_name, detected_breed_score
) values (
  %(id)s, %(account_id)s, %(pet_id)s, %(media_type)s, %(media_id)s, %(storage_path)s, %(sha256)s, %(size_bytes)s,
  %(content_type)s, %(context)s, %(decision)s, %(rejection_code)s, %(hard_reject)s, %(pet_score)s, %(composite_score)s,
  %(dimension_scores)s, %(ai_generated)s, %(ai_score)s, %(reason_codes)s, %(flags)s, %(species)s, %(objects)s,
  %(processing_ms)s, %(model_version)s, %(review_status)s, %(declared_breed_id)s, %(declared_is_cross_breed)s,
  %(breed_status)s, %(breed_match_score)s, %(detected_breed_id)s, %(detected_breed_name)s, %(detected_breed_score)s
) on conflict (id) do nothing
"""  # a job retried after a crash mustn't write a second record

# The file each row points to, to check the job's link belongs to that row.
ROW_PATH = {
    "pet_media": "select storage_path from pets.pet_media where id = %(id)s",
    "post_media": "select storage_path from social.post_media where id = %(id)s",
    "pet_profile_photo": "select profile_photo_path from pets.pets where id = %(id)s",
}


class MediaMismatch(Exception):
    """The link is not this row's file: someone sent another file's link with this row's id."""


def paths_match(link_path: str, row_path: str | None) -> bool:
    """The link's 'bucket/path' must be the row's path, which is stored with or without the bucket.
    A row with no path recorded can't be compared, so it passes."""
    if not row_path:
        return True
    link_path, row_path = link_path.strip("/"), unquote(row_path).strip("/")
    return link_path == row_path or link_path.endswith("/" + row_path)


UPDATE_PET_MEDIA = """
update pets.pet_media set moderation_status = %(status)s, moderation_rejection_code = %(code)s where id = %(id)s
"""
UPDATE_POST_WITH_SAME_ID = "update social.posts set moderation_status = %(status)s where id = %(id)s"
UPDATE_POST_MEDIA = """
update social.post_media set moderation_status = %(status)s, moderation_rejection_code = %(code)s
where id = %(id)s returning post_id
"""
LOCK_POST = "select id from social.posts where id = %(post_id)s for update"
UPDATE_POST_FROM_FILES = """
update social.posts p set moderation_status = (
  select case
    when bool_or(m.moderation_status = 'rejected') then 'rejected'
    when bool_and(m.moderation_status = 'approved') then 'approved'
    when bool_or(m.moderation_status = 'in_review') then 'in_review'
    else 'pending'
  end
  from social.post_media m where m.post_id = p.id
) where p.id = %(post_id)s
"""
UPDATE_PET_PHOTO = """
update pets.pets set profile_photo_moderation_status = %(status)s, profile_photo_moderation_rejection_code = %(code)s
where id = %(id)s
"""


class DecisionStore:
    """One transaction per result: the record and the visible status change together."""

    def __init__(self, url: str):
        self.url = url

    def save(self, job: Job, response: ModerationResponse, sha256: str | None, size_bytes: int | None,
             *, check_path: bool = True) -> None:
        """check_path=False only when giving up on a job: no content was judged, so only the
        status goes to in_review, and a missing grant or path must not block that forever."""
        outcome = outcome_for(response)  # the breed check is recorded only: it plays no part here
        breed = response.breed_check
        scores = response.dimension_scores.model_dump() if response.dimension_scores else None
        objects = [o.model_dump() for o in response.objects_detected] or None
        row = {
            "id": job.content_id, "account_id": job.account_id, "pet_id": job.pet_id or None,
            "media_type": job.media_type, "media_id": job.media_id, "storage_path": storage_path_of(job.media_url),
            "sha256": sha256, "size_bytes": size_bytes, "content_type": job.content_type, "context": job.context,
            "decision": response.decision, "rejection_code": response.rejection_code, "hard_reject": response.hard_reject,
            "pet_score": response.pet_score, "composite_score": response.composite_score,
            "dimension_scores": json.dumps(scores) if scores else None,
            "ai_generated": response.ai_generated, "ai_score": response.ai_score,
            "reason_codes": reason_codes(response), "flags": response.flags, "species": response.species_detected,
            "objects": json.dumps(objects) if objects else None, "processing_ms": response.processing_ms,
            "model_version": response.model_version, "review_status": outcome.review_status,
            "declared_breed_id": job.declared_breed_id or None,
            "declared_is_cross_breed": {"true": True, "false": False}.get(job.declared_is_cross_breed),
            "breed_status": breed.status, "breed_match_score": breed.match_score,
            "detected_breed_id": breed.detected_breed_id, "detected_breed_name": breed.detected_breed_name,
            "detected_breed_score": breed.detected_score,
        }
        target = {"status": outcome.moderation_status, "code": outcome.rejection_code, "id": job.media_id}
        with psycopg.connect(self.url, connect_timeout=settings.database.connect_timeout_seconds) as db:
            if check_path and settings.media.check_row_path:
                found_row = db.execute(ROW_PATH[job.media_type], {"id": job.media_id}).fetchone()
                if found_row and not paths_match(row["storage_path"], found_row[0]):
                    raise MediaMismatch(f"{job.media_type} {job.media_id} is {found_row[0]!r}, "
                                        f"the link is {row['storage_path']!r}")
            db.execute(INSERT_DECISION, row)
            if job.media_type == "pet_media":
                found = db.execute(UPDATE_PET_MEDIA, target).rowcount
                db.execute(UPDATE_POST_WITH_SAME_ID, target)
            elif job.media_type == "post_media":
                post = db.execute(UPDATE_POST_MEDIA, target).fetchone()
                found = 1 if post else 0
                if post:
                    db.execute(LOCK_POST, {"post_id": post[0]})
                    db.execute(UPDATE_POST_FROM_FILES, {"post_id": post[0]})
            else:
                found = db.execute(UPDATE_PET_PHOTO, target).rowcount
        if not found:  # deleted while it waited: the record is still kept
            logger.warning("job %s: no %s row %s to update", job.content_id, job.media_type, job.media_id)
