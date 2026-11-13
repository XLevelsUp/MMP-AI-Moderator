#!/usr/bin/env bash
# Test the production build on a Google Colab T4 before deploying to AWS.
#
# Colab has no Docker, so this runs the same code and settings the images use, directly:
# CUDA 12.6 PyTorch from the lock file (as in the worker image), every model on the T4, the API in
# production mode with an API key, Redis with a password, a local Postgres with the real
# migrations, and the worker as moderation_service with --exit-when-empty.
#
#   1. Runtime -> Change runtime type -> T4 GPU
#   2. Upload mmp_colab_build_test.zip and unzip it into /content
#   3. !bash /content/mmp-moderator-poc/scripts/colab_build_test.sh
#
# About 25-35 minutes, most of it downloading models (about 14 GB). The test files are served
# through a temporary public link while it runs; they are benchmark pictures, nothing private.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
VENV=/content/venv
MODELS=/content/models

step() { echo; echo "== $*"; }

if ! nvidia-smi >/dev/null 2>&1; then
    echo "NO GPU: Runtime -> Change runtime type -> T4 GPU -> Save, upload the zip again, rerun."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

step "1/6 system packages: Redis, Postgres, ffmpeg"
apt-get update -qq
apt-get install -y -qq redis-server postgresql ffmpeg >/dev/null

step "2/6 Python, exactly as in the worker image (lock file + CUDA 12.6 PyTorch)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12 -q
UV_PROJECT_ENVIRONMENT=$VENV uv sync --frozen --no-dev --no-install-project --python 3.12 \
    --no-install-package torch --no-install-package torchvision -q
uv pip install --python $VENV/bin/python -q --index-url https://download.pytorch.org/whl/cu126 \
    "torch==2.14.1+cu126" "torchvision==0.29.1+cu126"
export PATH="$VENV/bin:$PATH"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"
ffmpeg -hide_banner -hwaccels | grep -q cuda && echo "ffmpeg: NVIDIA decoding available" || echo "ffmpeg: no NVIDIA decoding (falls back to the CPU)"

step "3/6 models into $MODELS (about 14 GB, the long part)"
export HF_HOME=$MODELS/hf EASYOCR_MODEL_DIR=$MODELS/easyocr HF_HUB_DISABLE_TELEMETRY=1
python -m scripts.download_models --models siglip2_base owlv2 bioclip2 nsfw_vit ai_smaug ast_audioset whisper_base shieldstral
python -m scripts.download_models --ocr
export HF_HUB_OFFLINE=1

step "4/6 Redis (password, append-only) and Postgres (stand-in tables + the real migrations)"
# A second run in the same notebook: stop what the last run left behind.
pkill -f "redis-server" 2>/dev/null || true; pkill -f "cloudflared tunnel" 2>/dev/null || true
pkill -f "http.server" 2>/dev/null || true; pkill -f "uvicorn app.main" 2>/dev/null || true
sleep 1
REDIS_PASSWORD=$(python -c "import secrets; print(secrets.token_hex(16))")
mkdir -p /tmp/redis
redis-server --daemonize yes --requirepass "$REDIS_PASSWORD" --appendonly yes --dir /tmp/redis >/dev/null
export REDIS_URL="redis://:$REDIS_PASSWORD@127.0.0.1:6379/0"

service postgresql start >/dev/null
DB_PASSWORD=$(python -c "import secrets; print(secrets.token_hex(16))")
su postgres -c "psql -qc \"alter user postgres password '$DB_PASSWORD'\""
su postgres -c "dropdb --if-exists buildtest"
# Roles outlive a dropped database: remove the ones a previous run created.
su postgres -c "psql -q -c 'drop role if exists moderation_service, service_role, authenticated, anon'"
su postgres -c "createdb buildtest"
cp data/demo_stub.sql db/migrations/*.sql /tmp/ && chmod a+r /tmp/*.sql
cat > /tmp/service_role.sql <<'SQL'
do $$ begin
  if not exists (select from pg_roles where rolname = 'service_role') then create role service_role nologin bypassrls; end if;
end $$;
SQL
chmod a+r /tmp/service_role.sql
su postgres -c "psql -q -v ON_ERROR_STOP=1 -d buildtest -f /tmp/service_role.sql"
su postgres -c "psql -q -v ON_ERROR_STOP=1 -d buildtest -f /tmp/demo_stub.sql -f /tmp/20261104100000_moderation_schema.sql -f /tmp/20261112100000_review_status_done.sql -f /tmp/20261113100000_media_path_read.sql"
WORKER_PASSWORD=$(python -c "import secrets; print(secrets.token_hex(16))")
su postgres -c "psql -q -d buildtest -c \"alter role moderation_service login password '$WORKER_PASSWORD'\""
export PG_ADMIN_URL="postgresql://postgres:$DB_PASSWORD@127.0.0.1:5432/buildtest"
export DATABASE_URL="postgresql://moderation_service:$WORKER_PASSWORD@127.0.0.1:5432/buildtest"

step "5/6 serve the test files over HTTPS at Supabase-style paths"
if [ ! -x /tmp/cloudflared ]; then
    curl -fsSL -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x /tmp/cloudflared
fi
MEDIA_ROOT=/tmp/media
python scripts/build_test_run.py --stage-files "$ROOT/test_media" "$MEDIA_ROOT"
# Colab already runs its own server on 8080, so the files go on a port nobody uses.
MEDIA_PORT=8765
python -m http.server $MEDIA_PORT --bind 127.0.0.1 --directory "$MEDIA_ROOT" >/tmp/media_server.log 2>&1 &
sleep 2
curl -sf -o /dev/null "http://127.0.0.1:$MEDIA_PORT/storage/v1/object/public/pet-images/buildtest/ping.txt" \
    || { echo "The local file server did not start:"; cat /tmp/media_server.log; exit 1; }
/tmp/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:$MEDIA_PORT >/tmp/tunnel.log 2>&1 &
MEDIA_HOST=""
SERVED=""
for _ in $(seq 1 60); do
    MEDIA_HOST=$(grep -oE "[a-z0-9-]+\.trycloudflare\.com" /tmp/tunnel.log | head -1 || true)
    if [ -n "$MEDIA_HOST" ] && [ "$(curl -s "https://$MEDIA_HOST/storage/v1/object/public/pet-images/buildtest/ping.txt")" = "ok" ]; then
        SERVED=yes; break
    fi
    sleep 2
done
[ -n "$SERVED" ] || { echo "The temporary link does not serve the test files:"; tail -20 /tmp/tunnel.log; exit 1; }
echo "files at https://$MEDIA_HOST/storage/v1/object/public/pet-images/buildtest/"
export MEDIA_HOST

step "6/6 run: API in production mode, send the uploads, worker until the queue is empty"
python scripts/build_test_run.py --run
