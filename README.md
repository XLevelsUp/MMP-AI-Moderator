# MeetMyPets AI Moderation Service

Checks every photo and video uploaded to MeetMyPets and writes the result straight into the app's
Supabase database. Two questions per upload:

- **Gate 1:** is there a real pet in it?
- **Gate 2:** is it safe and suitable (animal welfare, nudity, violence, prohibited items, quality)?

All models are self-hosted on our own GPU. No commercial moderation APIs are used.

- **What is built, measured results, known gaps:** [docs/STATUS.md](docs/STATUS.md)
- **Request format for the app backend:** [docs/API.md](docs/API.md)
- **Server setup:** [docs/DEPLOY.md](docs/DEPLOY.md)

## How it works

```
App backend ──POST /api/v1/moderate──► API ──► Redis queue
     ▲            (answers 202 at once)            │
     │                                             ▼
     └──── reads the result ◄── Supabase ◄── GPU worker (downloads the file, runs the models,
                                Postgres       writes moderation.decisions + the media status)
```

1. The backend sends the file's **storage link** and the row it belongs to. The API checks the
   API key and the link, puts a job in the queue and answers **202** with no body.
2. A **worker** on the GPU takes jobs one at a time: downloads the file, validates it, runs
   Gate 1 and Gate 2, and writes the decision.
3. The result lands in `moderation.decisions`, and the media row's `moderation_status` becomes
   `approved` or `in_review`. Everything not approved goes to the moderators' review queue
   (`review_status = 'pending'`).

| Decision | Media status | Review |
| --- | --- | --- |
| `APPROVED` | `approved` | `done` (no person needed) |
| `REJECTED` / `HUMAN_REVIEW` | `in_review` | `pending`: a moderator confirms or corrects it |

Rejection codes (one per rejection): `MOD_001` no pet · `MOD_002` animal welfare ·
`MOD_003` animal fighting · `MOD_004` nudity · `MOD_005` violence · `MOD_006` prohibited items ·
`MOD_007` low quality · `MOD_008` bad or unreadable file · `MOD_009` animal trafficking.

## Repository layout

| Folder | What is in it |
| --- | --- |
| `app/api` | FastAPI routes, request/response schemas, API-key check, production start-up guard |
| `app/validation` | Download, file checks (format, size, length), CSAM check (placeholder) |
| `app/prepare` | Turning images and videos into frames (GPU video decoding with ffmpeg) |
| `app/gates` | Gate 1 (pet) and Gate 2 (safety dimensions D1-D6) |
| `app/ml` | Model loading and inference (SigLIP 2, OWLv2, BioCLIP 2, Shieldstral, NSFW, AI-image, audio, OCR) |
| `app/pipeline` | Runs the stages for one upload |
| `app/jobs` | Redis queue and writing results to the database |
| `app/worker.py` | The GPU worker process |
| `config/settings.yaml` | **Every threshold, limit and model version.** Nothing is hard-coded |
| `db/migrations` | SQL for the `moderation` schema (with rollback files) |
| `deploy` | Dockerfile, compose file, Redis config, worker start script, env templates |
| `scripts` | Model download, benchmarks, evaluation, Colab test kit |
| `tests` | pytest suite |

## Local development

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) (never pip), ffmpeg and Docker (for Redis).

```bash
uv sync                          # install exactly what uv.lock pins
cp .env.example .env             # local settings; never commit .env
docker compose up -d             # Redis on localhost:6379
uv run python -m scripts.download_models --models siglip2_base owlv2 bioclip2 nsfw_vit ai_smaug ast_audioset whisper_base
uv run python -m scripts.download_models --ocr
```

| Task | Command |
| --- | --- |
| Run the API | `uv run uvicorn app.main:app --reload` |
| Run a worker (second terminal) | `uv run python -m app.worker` |
| Run the tests | `uv run pytest` |
| Check one file by hand | `uv run python -m scripts.inspect_file <path> image\|video` |

On a laptop without a GPU everything runs on the CPU (`ML_DEVICE=cpu`), slowly. Shieldstral
(about 8 GB) is off by default there; set `GUARD_MODEL=shieldstral` on a machine with enough memory.

## Production

Two Docker images from `deploy/Dockerfile`:

| Image | Target | Size | Runs |
| --- | --- | --- | --- |
| `mmp-moderation-api` | `api` | about 2.6 GB | The API (CPU only) |
| `mmp-moderation-worker` | `worker` | about 12.6 GB | The GPU worker (CUDA 12.6). Models are **not** inside: they live in `/models` on the server |

Settings come from environment files, one per program, so the internet-facing API never holds the
database password:

| File | Template | Holds |
| --- | --- | --- |
| `api.env` | [deploy/api.env.example](deploy/api.env.example) | API keys, Redis password, allowed media host |
| `worker.env` | [deploy/worker.env.example](deploy/worker.env.example) | Redis link, database login, allowed media host |

In production (`APP_ENV=production`) the API and the worker refuse to start if a required setting
is missing or unsafe, and the worker refuses to start if any model failed to load. Health checks:
`GET /health/live` and `GET /health/ready` (Redis, queue length, running workers).

Full steps: [docs/DEPLOY.md](docs/DEPLOY.md).

## Rules for this repository

- **Never commit secrets:** `.env`, `api.env`, `worker.env`, passwords, API keys, database links.
- **Never commit `data/`** (test media and datasets) or model weights (except `models/gate1_head.npz`).
- **Limits and thresholds go in `config/settings.yaml`**, not in code.
- **Database changes go in `db/migrations`** with a rollback file, staging first.
- **Use uv**, not pip.
- Never download, store or use abusive, sexual or child-abuse material for testing or training.

## Known limits

- The CSAM hash check is a placeholder until a provider is connected; the worker logs a warning.
- One worker handles one file at a time: a photo takes about 1.5-3.5 s of GPU time on a T4.
- Videos with a pet take about 35-40 s on a T4 (target 15 s); Shieldstral and video decoding are
  the slow parts. Speed work is planned for the next version.
