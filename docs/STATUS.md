# MeetMyPets AI Moderation — POC status

**As of 30 Oct 2026 · code `gate2-0.19.2` / `gate1-fused-0.7.1` · 497 tests passing**

The moderation logic is close to its first production version: every upload goes through
validation, Gate 1 (is there a pet?) and Gate 2 (is it safe and suitable?), and gets
APPROVED, HUMAN_REVIEW or REJECTED with scores, flags and a reason code. What is not built
yet is the service around it (deployment, review queue, monitoring) — see
[Not built yet](#not-built-yet).

## What a request goes through

| Step | What it does | Models |
|---|---|---|
| Queue | `POST /api/v1/moderate` takes the file's link as JSON, queues it in Redis Streams and answers 202 at once. GPU workers (`python -m app.worker`) download and check one job at a time and update the database directly: `moderation.decisions` and `moderation_status`. Photos first, but a video waits for at most 8 photos. A failed job is retried, then goes to a moderator. A full queue answers 503 | — |
| Validation | format, size, duration, decode, metadata; duplicate cache and perceptual-hash blocklist in Redis | — |
| Prepare | resize; videos sampled at 1 frame per second (0.5 s under 10 s), GPU decoding when available | — |
| Gate 1 | pet score from SigLIP + the OWLv2 animal detector + BioCLIP species; no pet = answer at once (MOD_001); wild / marine animals go to review when 2 of 3 views agree. A trained head (pet / wild / no_pet, logistic regression on the SigLIP + BioCLIP embeddings) double-checks every pass: if it is ≥ 0.8 sure there is no pet, a human looks (`gate1.decider: check`) | SigLIP 2, OWLv2, BioCLIP 2, Gate 1 head |
| Gate 2 | D1 welfare, D2 adult, D3 violence, D4 items, D5 quality, D6 relevance; composite score and decision rules | all below |
| — safety questions | 11 fixed yes / no questions about each picture (full frame + a crop around the pet); a strong "yes" fires the spec signal, a medium score sends the post to a human | Shieldstral 1.0 |
| — scenes and contexts | harm / safe sentences, and the spec 7 contexts that soften deductions (medical, raw feeding, rescue, memorial) | SigLIP 2 |
| — nudity, items, text, sound | human nudity, weapons / drugs / alcohol / tobacco boxes, text on images (English + Hindi word lists), video sounds and speech | NSFW ViT, OWLv2, EasyOCR, AST, Whisper |
| AI label | `ai_generated` / `ai_score` from metadata first, then a detector; never changes the decision | Smaug |

Videos: Shieldstral looks at 12 frames — SigLIP's 3 most suspicious moments (each with the
next frame, these also get the pet crop) and the rest spread over the video. A medium score
must show in at least 2 frames.

## Models

| Model | Job | Licence | Download | GPU memory (fp16, approx.) |
|---|---|---|---|---|
| Shieldstral 1.0 (`mistralai/Shieldstral-1.0-3B`) | D1–D4 safety questions | Apache-2.0 | 7.73 GB | ~7.7 GB |
| SigLIP 2 base | pet score, scenes, contexts, relevance | Apache-2.0 | 1.54 GB | ~0.4–0.75 GB |
| BioCLIP 2 | species, wild-animal check | MIT | 1.71 GB | ~0.85 GB |
| Smaug AI-image detector | AI label | Apache-2.0 | 0.91 GB | ~0.9 GB |
| OWLv2 base | animal / look-alike / item boxes, pet crop | Apache-2.0 | 0.62 GB | ~0.3 GB |
| AST AudioSet | video sounds | BSD-3-Clause | 0.35 GB | ~0.2 GB |
| NSFW ViT | human nudity | Apache-2.0 | 0.34 GB | ~0.2 GB |
| EasyOCR (en + hi) | text on images | Apache-2.0 | 0.30 GB | ~0.2–0.3 GB |
| Whisper base | video speech | Apache-2.0 | 0.29 GB | ~0.15 GB |

All self-hosted, pinned to an exact revision in `config/settings.yaml`. Total about 13.8 GB
on disk and 11–11.5 GB of GPU memory: fits a T4 (15 GB) tightly, an L4 (24 GB) comfortably.
Shieldstral was chosen on 29 Oct 2026 over ShieldGemma 2: Apache-2.0 and ungated, about 13x
faster on the same GPU, fewer false alarms on normal pet photos, and it reads behaviour
(distress, the moment of a bite) where ShieldGemma did not.

## Measured results

**Benchmark (Colab T4, 30 Oct, `gate2-0.19`): 33 / 33 decisions correct** — 13 no-pet files
rejected, 19 pet photos and videos approved (including a bird cage and two app-promo videos
that earlier went to review), and a grooming video where the dog bites the groomer sent to
human review.

| Measure (T4, fp16) | Result | Spec target |
|---|---|---|
| Photo, p50 / p95 | 1.95 s / 3.7 s | < 3 s p95 |
| 60 s video, p95 | 26.2 s (before the 0.19.2 crop change, about a quarter less guard work since) | < 15 s p95 |
| Shieldstral per photo | 1.9 s (was 4.9 s before batching) | — |
| Models loaded and warm | 375 s | — |

An L4 is expected to be about 2x faster than a T4 for this model; this has to be measured.

**False alarms on 264 safe photos** (pets, grooming, vet visits, rope toys, nursing mothers,
raw food, toys, play fighting): Shieldstral thresholds keep at most about 1% of normal photos
in review. Play fighting goes to review by design; nothing safe is rejected.

## Thresholds

`config/settings.yaml` → `gate2.guard`: the 11 questions, their wording and, per question,
`review_from` (a human looks) and `act_from` (the spec signal fires). `review_from` was set
from the safe photos with `scripts/guard_calibrate.py` (rerun it whenever the questions, the
wording, the image size or the model change). `act_from` stays at the model card's 0.5 until
real harmful examples set it from the spec's recall targets.

## Known limitations

- **Breed check:** the fields are in the request, result and database, but the check isn't built, so every result says `not_checked`. A result stays in the database even if the user swaps the file while the job waits; a swap made while a job runs can be overwritten by that job's status.

- **The queue doesn't make one GPU faster.** Requests no longer pile up on the GPU, but a worker does
  one job at a time, so a T4 still does about 0.5 photos per second. 50 per second needs batching
  across requests and more GPUs. The 2,000 / 200 waiting limits can overshoot slightly during a burst. Redis isn't set up to survive a restart without losing waiting jobs. Nothing yet alerts
  when no worker is running.
- **Animated GIF, WebP and PNG** are judged like a video (2 Nov 2026, replacing decision D-11, first
  frame only): frames at the video times, the same 60 s limit, the same Gate 1 and Gate 2 rules. The
  blocklist still hashes only the first frame.
- **Catch rate is not measured.** No real harmful examples have been tested yet beyond one
  video; they must come from the trust-and-safety team (never collected by us).
- **Blood:** Shieldstral did not see a one-second bleed after a bite (scored 0.09).
- **Split-second snaps** without visible injury are missed by every model tried.
- **Hard rejects from Shieldstral at 0.5 are not calibrated:** a pet monkey photo scored
  "hurt" 0.49. Per the 28 Oct decision every rejection still goes to a human.
- **Gate 1 look-alikes:** the rules approved 6 no-pet look-alikes on the hard-case set; with the
  trained head checking (2 Nov 2026) it is 2 (an empty planted aquarium, a closed carrier),
  approval precision 90.6% → 96.4%, no extra pets rejected, 4 more pets to review (two cartoons,
  a meme, a can label). The head alone is not ready to decide: it calls aquarium fish with glare,
  a hamster in hands and a blurred running dog "wild", and rejects cartoons and memes. Round 2
  (2 Nov, `gate1-fused-0.7.1`): 198 more web photos of exactly those kinds (1,896 in all,
  held-back accuracy 98.9 %) did **not** fix those hard-case photos — same scores — so more web
  photos will not get the head to decide alone; real app photos from the review queue are needed.
  The round 2 head runs as the check at 0.7: same hard-case result as round 1 at 0.8 (2 / 3 / 6).
  Retrain on Colab: `bash scripts/colab_head.sh` (≈ 15 min on a T4).

The fix for the first three is a fine-tune of Shieldstral on moderator labels from the review queue.

## How to run

**Laptop (CPU, Shieldstral off — it needs ~8 GB more than the laptop has):**
see the commands in the README (Redis, API, Streamlit test page, tests).

**Colab T4 benchmark:** upload the code and the benchmark files, then
`bash scripts/colab_t4.sh` (installs, downloads the models, runs the benchmark with
Shieldstral on).

**Colab T4 API + local Streamlit:** download the models, start Redis and
`GUARD_MODEL=shieldstral ML_DEVICE=cuda ML_DTYPE=float16 FFMPEG_HWACCEL=cuda uvicorn app.main:app`,
open a tunnel to port 8000, and point the test page's API URL at it. Add `QUEUE_ENABLED=false`:
the test page uses `POST /api/v1/moderate/upload`, which answers in the same request and writes nothing.

Switches: `GUARD_MODEL` (Shieldstral on / off), `ML_DEVICE`, `ML_DTYPE`, `FFMPEG_HWACCEL`.

## Not built yet

| Item | Why it is needed |
|---|---|
| Dockerfile, production settings (GPU, Shieldstral on), `/health/live` + `/health/ready`, JSON logs, `/metrics`, decoder warm-up | DevOps cannot run it without these |
| Batching across requests in the GPU workers | the queue is built; this is what raises throughput toward 50 photos per second |
| Background workers for the AI label and speech-to-text | speed; they never change the decision |
| Review queue + moderator correction form (`moderation` schema in the existing database) | every review / rejection must reach a person; their corrections are the training data |
| Speed test on an L4, video tuning to < 15 s | spec target |
| Catch-rate test on trust-and-safety examples, then `act_from` thresholds | spec recall targets |
| Shadow mode, then live | prove agreement with moderators before auto-approval |
| Gate 1 head retrained on real app photos (web round 2 did not help) so it can decide alone; Shieldstral fine-tune | the known limitations above |
