from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


ContentType = Literal["image", "video"]
ContentContext = Literal["post", "profile", "breeding", "story"]
Decision = Literal["APPROVED", "REJECTED", "HUMAN_REVIEW"]
RejectionCode = Literal[
    "MOD_001", "MOD_002", "MOD_003", "MOD_004", "MOD_005",
    "MOD_006", "MOD_007", "MOD_008", "MOD_009",
]

Score = Field(default=None, ge=0, le=100)


class DimensionScores(BaseModel):
    D1_welfare: float = Field(ge=0, le=100)
    D2_nsfw: float = Field(ge=0, le=100)
    D3_violence: float = Field(ge=0, le=100)
    D4_prohibited: float = Field(ge=0, le=100)
    D5_quality: float = Field(ge=0, le=100)
    D6_relevance: float = Field(ge=0, le=100)


class DetectedObject(BaseModel):
    """One thing the object detector found (not in the spec; for analysing decisions)."""

    label: str                  # the detector's name, e.g. "dog", "statue", "pistol"
    kind: Literal["animal", "lookalike", "item"]
    score: float = Field(ge=0, le=1)   # the detector's confidence for its best box


EntityType = Literal["pet_media", "post_media", "pet_profile_photo"]
ENTITY_CONTEXT = {"pet_media": "post", "post_media": "post", "pet_profile_photo": "profile"}


class ModerationRequest(BaseModel):
    media_url: str
    content_type: ContentType
    account_id: UUID                # identity.accounts.id
    entity_type: EntityType         # which table holds this file
    entity_id: UUID                 # the row whose moderation_status gets updated
    pet_id: UUID | None = None
    context: Literal["post", "profile"] = "post"
    declared_breed_id: UUID | None = None           # pets.breeds.id: the breed the user claimed
    declared_is_cross_breed: bool | None = None     # a cross breed has no single breed to match

    @model_validator(mode="after")
    def context_matches_entity(self):
        if ENTITY_CONTEXT[self.entity_type] != self.context:
            raise ValueError(f"context {self.context} does not go with entity_type {self.entity_type}")
        return self


class QueuedResponse(BaseModel):
    """HTTP 202: accepted. The worker writes the result to the database."""

    content_id: UUID
    status: Literal["queued"] = "queued"


class BreedCheck(BaseModel):
    """Claimed breed vs the photo. Recorded only: it never changes the decision. Not built yet, so
    today every answer says not_checked."""

    status: Literal["matched", "unmatched", "not_checked"] = "not_checked"
    declared_breed_id: UUID | None = None
    match_score: float | None = Score         # how well the photo fits the claimed breed
    detected_breed_id: UUID | None = None     # the breed the AI found: set when unmatched
    detected_breed_name: str | None = None
    detected_score: float | None = Score


class ModerationResponse(BaseModel):
    content_id: UUID
    decision: Decision

    pet_score: float | None = Score
    composite_score: float | None = Score

    ai_generated: bool | None = None
    ai_score: float | None = Score

    dimension_scores: DimensionScores | None = None

    hard_reject: bool = False
    rejection_code: RejectionCode | None = None
    flags: list[str] = Field(default_factory=list)
    species_detected: list[str] = Field(default_factory=list)
    breed_check: BreedCheck = Field(default_factory=BreedCheck)
    objects_detected: list[DetectedObject] = Field(default_factory=list)

    processing_ms: int = Field(ge=0)
    model_version: str
