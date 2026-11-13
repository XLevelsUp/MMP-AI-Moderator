# MeetMyPets AI moderation: two images from one file.
#
#   docker build -f deploy/Dockerfile --target api    -t mmp-moderation-api .
#   docker build -f deploy/Dockerfile --target worker -t mmp-moderation-worker .
#
# api    = the HTTP API that queues uploads (small server, no GPU, no models).
# worker = the GPU worker that holds the models and writes results to the database.
# Models are NOT in the image: they live on the GPU server's disk and are mounted at /models
# (see docs/DEPLOY.md). That keeps the image small and a model change needs no rebuild.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.0

FROM ${UV_IMAGE} AS uv

# ---------- shared base ----------
FROM ${PYTHON_IMAGE} AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    APP_ENV=production \
    LOG_FORMAT=json \
    TEMP_DIR=/tmp/mmp \
    QUARANTINE_DIR=/app/data/quarantine
COPY --from=uv /uv /usr/local/bin/uv
# Fixed ids, so the model disk on the GPU server can be given to this user: chown -R 10001:10001 /models
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app \
    && mkdir -p /app/data /tmp/mmp \
    && chown app:app /app/data /tmp/mmp
WORKDIR /app

# ---------- API: CPU PyTorch from the lock file ----------
FROM base AS api
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY config ./config
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-server-header"]

# ---------- worker: CUDA PyTorch + ffmpeg ----------
FROM base AS worker
# ffmpeg from Debian includes NVIDIA decoding (nvdec/cuvid); FFMPEG_HWACCEL=cuda uses it,
# and any GPU decode failure falls back to the CPU.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
# Everything from the lock file except PyTorch, which is swapped for the CUDA 12.6 build of the
# same version (the lock pins the CPU build for the laptop).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --no-install-package torch --no-install-package torchvision \
    && uv pip install --index-url https://download.pytorch.org/whl/cu126 \
       "torch==2.14.1+cu126" "torchvision==0.29.1+cu126"
COPY app ./app
COPY config ./config
COPY models/gate1_head.npz ./models/gate1_head.npz
# Only for the one-off model download onto /models (see docs/DEPLOY.md)
COPY scripts/download_models.py ./scripts/
ENV ML_DEVICE=cuda \
    ML_DTYPE=float16 \
    FFMPEG_HWACCEL=cuda \
    GUARD_MODEL=shieldstral \
    HF_HOME=/models/hf \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    EASYOCR_MODEL_DIR=/models/easyocr \
    WORKER_EXIT_WHEN_EMPTY=true \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
USER app
CMD ["python", "-m", "app.worker"]
