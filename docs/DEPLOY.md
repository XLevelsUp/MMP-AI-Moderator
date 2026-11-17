# Deploying the AI moderation service

Two servers in AWS Mumbai (`ap-south-1`, the same region as the production Supabase database; staging is in Seoul, which works from Mumbai, just a little slower): an always-on `t3.small` for the API and the Redis queue,
and a `g4dn.xlarge` GPU server that starts once a day, clears the queue and shuts itself down.
Results go straight into the Supabase database.

```
App backend ──HTTPS──► API (t3.small) ──► Redis queue (same server)
                                              ▲
                     GPU worker (g4dn.xlarge) ┘ ──► Supabase Postgres (moderation.decisions + status columns)
                     starts daily, stops itself when the queue is empty
```

Everything here was built and tested on 12 Nov 2026 except what is marked **not tested yet**.
The full flow (API → queue → GPU worker → database) passed on a Colab T4 on 17 Nov 2026.

**First deployment: one machine.** To start, everything can run on a single always-on
`g4dn.xlarge`. Follow sections 2, 3, 5 and 6 on that one machine, with these differences:

| | One machine |
| --- | --- |
| Security group | 22 (SSH) and 8000 (API) from your IPs only. No 6379 |
| `REDIS_BIND_IP` in `api.env` | The machine's own private IP (`hostname -I`) |
| `REDIS_URL` in `worker.env` | `redis://:<REDIS_PASSWORD>@<the same private IP>:6379/0` |
| `WORKER_EXIT_WHEN_EMPTY` in `worker.env` | `false`: the worker keeps running and the machine stays on |
| Start the worker | `docker run -d --name mmp-worker --restart unless-stopped --gpus all --env-file /opt/mmp/worker.env -v /models:/models:ro --shm-size 2g <worker image>` instead of `run-worker.sh` and the service of 6.7 (those shut the server down) |
| Images | Can be built on the machine itself (section 3) instead of pushed to ECR |
| Cost | About $0.53 an hour while running. Stop the instance when not testing; it comes back up with everything started |

Later, split it into the two-server setup below to save cost.

## 1. Before you start

| Item | Where |
| --- | --- |
| GPU quota: "Running On-Demand G and VT instances", at least 4 vCPUs in `ap-south-1` | AWS Service Quotas. Request it first: approval can take days |
| A domain name for the API, such as `moderation.example.com` | Your DNS provider |
| The database migrations run on the target database | Section 2 |
| A password for the `moderation_service` role | Section 2 |

## 2. Database

Run in the Supabase SQL Editor, in this order, on staging first and production only after staging is tested:

1. `db/migrations/20261104100000_moderation_schema.sql`
2. `db/migrations/20261112100000_review_status_done.sql`
3. Optional, only if you turn on `CHECK_ROW_PATH=true`: `db/migrations/20261113100000_media_path_read.sql`
   (lets the worker check a link is the row's own file; off by default)
4. Give the worker its login. Keep the password out of chat and the repo:

   ```sql
   alter role moderation_service login password '<strong password>';
   ```

Do **not** run `20261104100100_moderation_go_live.sql` yet: it makes new uploads start as `pending` and is for go-live.
Each migration has an undo file in `db/migrations/rollback/`.

The worker's `DATABASE_URL` uses the Supabase pooler in session mode:

```
postgresql://moderation_service.<project ref>:<password>@<pooler host>:5432/postgres?sslmode=require
```

The pooler host is in the Supabase dashboard under Connect. **Not tested yet** against the real pooler.

## 3. Build the images

On any machine with Docker (the GPU is not needed to build):

```bash
docker build -f deploy/Dockerfile --target api    -t mmp-moderation-api:1.0 .
docker build -f deploy/Dockerfile --target worker -t mmp-moderation-worker:1.0 .
```

Push both to Amazon ECR (create two repositories named `mmp-moderation-api` and `mmp-moderation-worker` first):

```bash
aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.ap-south-1.amazonaws.com
for name in api worker; do
  docker tag mmp-moderation-$name:1.0 <account>.dkr.ecr.ap-south-1.amazonaws.com/mmp-moderation-$name:1.0
  docker push <account>.dkr.ecr.ap-south-1.amazonaws.com/mmp-moderation-$name:1.0
done
```

| Image | Size (approx.) | Holds |
| --- | --- | --- |
| `mmp-moderation-api` | about 2.6 GB | The API, CPU PyTorch (imported but no models loaded) |
| `mmp-moderation-worker` | about 12.6 GB | The worker, CUDA 12.6 PyTorch, ffmpeg. **No models**: they're on the GPU server's disk |

## 4. Network

| Security group | Inbound | From |
| --- | --- | --- |
| `mmp-api` | 443 (HTTPS) | The app backend's addresses, or anywhere if it runs outside AWS |
| `mmp-api` | 6379 (Redis) | The `mmp-worker` security group only |
| `mmp-api` | 22 (SSH) | Your office IP only |
| `mmp-worker` | 22 (SSH) | Your office IP only |

The worker needs outbound access to Redis and to Supabase (database and storage). Never open 6379 to the internet.

## 5. API server (`t3.small`, always on)

1. Launch Ubuntu 24.04 LTS, `t3.small`, 30 GB gp3, security group `mmp-api`. Give it an Elastic IP and point the domain at it.
2. Install Docker: `curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker ubuntu`, then log in again.
3. Copy the repo's `deploy/` folder to `/opt/mmp/deploy/`.
4. Create `/opt/mmp/api.env` from `deploy/api.env.example`. `chmod 600` it.

   | Setting | Value |
   | --- | --- |
   | `API_KEYS` | `openssl rand -hex 32`. Give the same key to the backend team |
   | `REDIS_PASSWORD` | `openssl rand -hex 24` |
   | `REDIS_BIND_IP` | This server's private IP, such as `10.0.1.25` |
   | `MEDIA_ALLOWED_HOSTS` | `<project ref>.supabase.co` |

5. Pull the image and start:

   ```bash
   cd /opt/mmp/deploy
   API_IMAGE=<account>.dkr.ecr.ap-south-1.amazonaws.com/mmp-moderation-api:1.0 \
     docker compose -f docker-compose.api.yml --env-file /opt/mmp/api.env up -d
   curl -s localhost:8000/health/ready
   ```

   The API refuses to start in production if `API_KEYS`, `MEDIA_ALLOWED_HOSTS` or the queue setting is missing, and says which.

6. **HTTPS.** The API listens on port 8000 without TLS. Put HTTPS in front of it, for example Caddy on the same server, which gets a certificate on its own:

   ```bash
   sudo apt install -y caddy
   echo 'moderation.example.com { reverse_proxy 127.0.0.1:8000 }' | sudo tee /etc/caddy/Caddyfile
   sudo systemctl reload caddy
   ```

   **Not tested yet.** An AWS load balancer with an ACM certificate works too, at extra monthly cost.

## 6. GPU server (`g4dn.xlarge`, part-time)

1. Launch the AMI **"Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"**. It has the NVIDIA driver, Docker and the NVIDIA container toolkit. Use `g4dn.xlarge`, a 100 GB gp3 disk and security group `mmp-worker`.
2. Check the GPU: `nvidia-smi` should show a Tesla T4.
3. Copy `deploy/` to `/opt/mmp/deploy/` and `chmod +x /opt/mmp/deploy/run-worker.sh`.
4. Create `/opt/mmp/worker.env` from `deploy/worker.env.example`, with `chmod 600`:
   `REDIS_URL` points to the API server's **private** IP; `DATABASE_URL` is from section 2.
5. Pull the worker image and download the models once (about 13.8 GB, 10 to 20 minutes):

   ```bash
   sudo mkdir -p /models && sudo chown 10001:10001 /models
   docker run --rm -v /models:/models <worker image> python -m scripts.download_models --models siglip2_base owlv2 bioclip2 nsfw_vit ai_smaug ast_audioset whisper_base shieldstral
   docker run --rm -v /models:/models <worker image> python -m scripts.download_models --ocr
   ```

   The first command fetches the 8 models production uses (settings.yaml also lists 3 unused ones, so name them). The second fetches the text-reading (OCR) models.

6. Test by hand first, keeping the server on:

   ```bash
   touch /opt/mmp/STAY_ON
   WORKER_IMAGE=<worker image> /opt/mmp/deploy/run-worker.sh
   ```

   It should log the models loading (about 6 minutes on a T4), "worker ... ready", the jobs, then "queue empty, stopping".
7. Turn on start-at-boot. Edit `deploy/mmp-worker.service` if the image name is not the default, then:

   ```bash
   sudo cp /opt/mmp/deploy/mmp-worker.service /etc/systemd/system/
   echo 'WORKER_IMAGE=<worker image>' | sudo tee /etc/default/mmp-worker
   sudo systemctl daemon-reload && sudo systemctl enable mmp-worker
   rm /opt/mmp/STAY_ON
   ```

   Allow `ubuntu` to shut the server down without a password: `echo 'ubuntu ALL=(root) NOPASSWD: /sbin/shutdown' | sudo tee /etc/sudoers.d/mmp-shutdown`.

**Not tested yet:** the worker image on a real GPU. It was built and checked on a laptop without one.

## 7. Daily schedule and alerts

| What | How |
| --- | --- |
| Start the GPU server daily | EventBridge Scheduler, a daily schedule with target "EC2 StartInstances" and the instance id. Pick a quiet hour, for example 02:00 IST |
| It stops itself | `run-worker.sh` shuts the server down once the queue is empty. If the worker fails to start, it waits 15 minutes for SSH, then shuts down anyway |
| Alert if it stays on | A CloudWatch alarm on the instance's `CPUUtilization` (any value) for more than 3 hours, or a Lambda that checks its state hourly |
| Alert if the API is down | Route 53 health check, or any uptime monitor, on `https://<domain>/health/live` |
| Watch the backlog | `GET /health/ready` shows `waiting` (photos and videos in the queue) and `workers_running` |

Keep the server on for maintenance: `touch /opt/mmp/STAY_ON`. Remove it when done.

## 8. Check it works

1. `curl https://<domain>/health/ready` returns `"status":"ok"`.
2. Send a test request with the key (use a real file link from staging storage):

   ```bash
   curl -i https://<domain>/api/v1/moderate -H "X-API-Key: <key>" -H "Content-Type: application/json" -d '{
     "media_url": "https://<ref>.supabase.co/storage/v1/object/public/pet-images/mock-moderation/dog-bath.png",
     "content_type": "image", "account_id": "<account id>", "media_type": "pet_media", "media_id": "<pet_media id>"}'
   ```

   Expect `202`. `waiting.image` in `/health/ready` goes up by one.
3. Start the GPU server (or run the worker by hand). When it has stopped, check the row:

   ```sql
   select decision, review_status, reason_codes, created_at from moderation.decisions order by created_at desc limit 5;
   ```

## 9. Updating

- **Code change:** rebuild and push the images with a new tag, then on the API server `docker compose ... pull && up -d`, and on the GPU server update `WORKER_IMAGE`. The worker picks it up on its next daily start.
- **Model change:** update the revision in `config/settings.yaml`, rebuild, and rerun the model download on the GPU server.
- **Settings:** edit `api.env` or `worker.env` and restart that service.

## 10. Settings the servers read

| Setting | Server | Meaning |
| --- | --- | --- |
| `APP_ENV` | Both | `production` refuses to start without the settings below. Set in the images |
| `API_KEYS` | API | Comma-separated keys, each at least 32 characters. Callers send one in `X-API-Key` |
| `MEDIA_ALLOWED_HOSTS` | Both | The only hosts media is downloaded from, and only under `/storage/v1/object/` |
| `REDIS_URL` / `REDIS_PASSWORD` / `REDIS_BIND_IP` | API, worker | The queue |
| `DATABASE_URL` | Worker | The `moderation_service` login |
| `LOG_FORMAT`, `LOG_LEVEL` | Both | `json` lines in production |
| `WORKER_EXIT_WHEN_EMPTY` | Worker | `true` in the image: stop after the queue has been empty for 60 seconds |
| `ML_DEVICE`, `ML_DTYPE`, `FFMPEG_HWACCEL`, `GUARD_MODEL`, `HF_HOME`, `EASYOCR_MODEL_DIR` | Worker | Set in the image for the T4 |

## 11. Known limits of this build

- The illegal-content hash check is a placeholder until a provider is connected (`safety.csam_provider`); the worker logs a warning at start.
- One worker handles one file at a time: about 0.5 photos a second on a T4, roughly 3,400 photos in a 2-hour run.
- A video with several video tracks is judged on the first one only.

## 12. Troubleshooting

| Symptom | Look at |
| --- | --- |
| API won't start | `docker compose logs api`: the first line says which production setting is missing |
| `401` from the API | The caller's `X-API-Key` header |
| `422` "media_url must be an HTTPS link to the app's storage" | `MEDIA_ALLOWED_HOSTS`, and the link must contain `/storage/v1/object/` |
| `503` from the API | Redis down, or the queue full (2,000 photos or 200 videos waiting) |
| GPU server stays on | `journalctl -t mmp-worker`; a job still in progress counts as not empty; is `/opt/mmp/STAY_ON` there? |
| Rows with `check_incomplete` | The worker could not finish that file 3 times; its logs say why |
| Log says "dropped: the link is not this row's file" | The backend sent one file's link with another row's id. Nothing was written; fix the request |
