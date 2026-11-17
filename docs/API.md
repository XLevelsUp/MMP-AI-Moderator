# Moderation API: what the app backend sends

The backend calls the API **once per uploaded file, after the file is in Supabase storage**. The
API answers at once with `202` and no body. A GPU worker checks the file later and writes the
result **directly into the database**, so the backend never waits for or reads an API answer.

## Request

`POST /api/v1/moderate` · `Content-Type: application/json` · header `X-API-Key: <key>`

```json
{
  "media_url": "https://<project-ref>.supabase.co/storage/v1/object/public/pet-images/abc/dog.jpg",
  "content_type": "image",
  "account_id": "8d1c...-uuid",
  "media_type": "pet_media",
  "media_id": "4f2a...-uuid",
  "pet_id": "b71e...-uuid",
  "context": "post",
  "declared_breed_id": null,
  "declared_is_cross_breed": null
}
```

| Field | Required | Values |
| --- | --- | --- |
| `media_url` | yes | HTTPS link to the file in the app's Supabase storage (`/storage/v1/object/...`). The worker downloads it later, so a signed link must stay valid for **at least a day** |
| `content_type` | yes | `image` or `video` |
| `account_id` | yes | `identity.accounts.id` of the uploader |
| `media_type` | yes | Which table holds the file: `pet_media`, `post_media` or `pet_profile_photo` |
| `media_id` | yes | The row whose status gets updated: `pets.pet_media.id`, `social.post_media.id`, or `pets.pets.id` for a profile photo |
| `pet_id` | no | `pets.pets.id`, when known |
| `context` | no | `post` (default) or `profile` |
| `declared_breed_id` | no | `pets.breeds.id` the user claimed; the AI checks the photo against it |
| `declared_is_cross_breed` | no | `true` for a cross breed (no single breed to match) |

Files: images up to 50 MB, videos up to 200 MB and 60 seconds (limits in `config/settings.yaml`).

## Answers

| Status | Meaning | What the backend does |
| --- | --- | --- |
| `202` | Queued. No body | Nothing more. The result appears in the database |
| `401` | Missing or wrong `X-API-Key` | Fix the key |
| `422` | Bad request body, or `media_url` is not an HTTPS link to the app's storage | Fix the request; don't retry as is |
| `503` | Queue full or unavailable | **Retry later**, after the `Retry-After` seconds when given |

Sending the same file for the same row again is safe: it is not checked twice.

## Where the result appears

The worker updates the media row and adds one row to `moderation.decisions`.

| `media_type` | Status columns updated |
| --- | --- |
| `pet_media` | `pets.pet_media.moderation_status`, `moderation_rejection_code` |
| `post_media` | `social.post_media.moderation_status`, `moderation_rejection_code`, and the post's `social.posts.moderation_status` |
| `pet_profile_photo` | `pets.pets.profile_photo_moderation_status`, `profile_photo_moderation_rejection_code` |

| AI decision | `moderation_status` | `moderation.decisions.review_status` |
| --- | --- | --- |
| `APPROVED` | `approved` | `done` |
| `REJECTED` or `HUMAN_REVIEW` | `in_review` | `pending`: a moderator decides in the admin panel |

Rejection codes: `MOD_001` no pet · `MOD_002` animal welfare · `MOD_003` animal fighting ·
`MOD_004` nudity · `MOD_005` violence · `MOD_006` prohibited items · `MOD_007` low quality ·
`MOD_008` bad or unreadable file · `MOD_009` animal trafficking.

Main columns of `moderation.decisions`: `decision`, `rejection_code`, `reason_codes`, `flags`,
`pet_score`, `composite_score`, `dimension_scores` (D1-D6), `ai_generated`, `ai_score`,
`species_detected`, the breed check (`breed_match_score`, `detected_breed_id`, ...),
`processing_ms`, `model_version`, and the moderator's review fields (`review_status`,
`final_decision`, `missing_reason_codes`, `reviewed_by`, `reviewed_at`). The AI's own columns are
read-only; only the review columns can be changed. Schema: `db/migrations/`.

## Health

| | |
| --- | --- |
| `GET /health/live` | The API process is up |
| `GET /health/ready` | Redis reachable; shows `waiting` (photos and videos in the queue) and `workers_running` |
