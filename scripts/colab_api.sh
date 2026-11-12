#!/usr/bin/env bash
# The API on a Google Colab T4 for the local test page (Runtime -> Change runtime type -> T4 GPU).
# Unzip mmp_backend.zip into /content, then:  !bash /content/mmp-moderator-poc/scripts/colab_api.sh
# Prints a public link: put <link>/api/v1/moderate/upload in the test page's "API URL" box.
# Uses the upload route, which answers in the same request with every model on. No queue, no database.
# Anyone holding the link can use the API while this cell runs: don't share it, stop the cell when done.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi >/dev/null 2>&1; then
    echo "NO GPU: this notebook runs on a CPU. Runtime -> Change runtime type -> T4 GPU -> Save,"
    echo "then upload the zip again (changing the runtime wipes /content) and rerun this cell."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv export --no-hashes --no-dev --format requirements-txt | grep -v -E "^(torch|torchvision|torchaudio)" > /tmp/req.txt
uv pip install --system -q -r /tmp/req.txt
python -c "import torchvision" 2>/dev/null || uv pip uninstall --system torchvision
python -c "import torchaudio" 2>/dev/null || uv pip uninstall --system torchaudio

command -v redis-server >/dev/null || (apt-get update -qq && apt-get install -y -qq redis-server >/dev/null)
command -v ffmpeg >/dev/null || apt-get install -y -qq ffmpeg >/dev/null
redis-server --daemonize yes >/dev/null

# The dummy database for the test page's "Database preview" tab: stub tables plus our real SQL.
command -v psql >/dev/null || (apt-get update -qq && apt-get install -y -qq postgresql >/dev/null)
service postgresql start >/dev/null
su postgres -c "psql -qc \"alter user postgres password 'demo'\""
if ! su postgres -c "psql -lqt" | cut -d'|' -f1 | grep -qw moderation_demo; then
    su postgres -c "createdb moderation_demo"
    cp data/demo_stub.sql data/migrations/20261104100000_moderation_schema.sql /tmp/ && chmod a+r /tmp/demo_stub.sql /tmp/20261104100000_moderation_schema.sql
    su postgres -c "psql -q -v ON_ERROR_STOP=1 -d moderation_demo -f /tmp/demo_stub.sql -f /tmp/20261104100000_moderation_schema.sql"
fi
export DEMO_DATABASE_URL=postgresql://postgres:demo@127.0.0.1:5432/moderation_demo

python -m scripts.download_models --models siglip2_base owlv2 bioclip2 nsfw_vit ai_smaug ast_audioset whisper_base shieldstral
python -m scripts.download_models --ocr

if [ ! -x /tmp/cloudflared ]; then
    curl -fsSL -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x /tmp/cloudflared
fi
/tmp/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 > /tmp/tunnel.log 2>&1 &

export QUEUE_ENABLED=false ML_DEVICE=cuda ML_DTYPE=float16 FFMPEG_HWACCEL=cuda GUARD_MODEL=shieldstral
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 > /tmp/api.log 2>&1 &
API_PID=$!

echo "== loading the models, about 6 minutes"
until curl -sf http://127.0.0.1:8000/health >/dev/null; do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "The API stopped:"; tail -n 40 /tmp/api.log; exit 1
    fi
    sleep 5
done

LINK=""
for _ in $(seq 1 30); do
    LINK=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /tmp/tunnel.log | head -n 1 || true)
    [ -n "$LINK" ] && break
    sleep 2
done
[ -n "$LINK" ] || { echo "No public link:"; tail -n 20 /tmp/tunnel.log; exit 1; }

echo
echo "== READY. Put this in the test page's API URL box:"
echo "   $LINK/api/v1/moderate/upload"
echo
echo "== server log (stop this cell to shut everything down)"
tail -f /tmp/api.log
