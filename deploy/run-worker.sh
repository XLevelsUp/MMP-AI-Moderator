#!/usr/bin/env bash
# Runs on the GPU server at boot (deploy/mmp-worker.service). Clears the queue, then shuts the
# server down so the GPU stops costing money. AWS starts the server again on its schedule.
#
# Settings: /opt/mmp/worker.env (see .env.example). Models: /models (see docs/DEPLOY.md).
# To keep the server on for maintenance, create /opt/mmp/STAY_ON before it boots (or right after).
set -uo pipefail

IMAGE="${WORKER_IMAGE:-mmp-moderation-worker:latest}"
LOG_TAG="mmp-worker"

log() { logger -t "$LOG_TAG" "$*"; echo "$(date -Is) $*"; }

log "starting worker ($IMAGE)"
docker run --rm --name mmp-worker \
    --gpus all \
    --env-file /opt/mmp/worker.env \
    -v /models:/models:ro \
    --shm-size 2g \
    --log-driver journald --log-opt tag="$LOG_TAG" \
    "$IMAGE"
status=$?
log "worker stopped with status $status"

if [ -e /opt/mmp/STAY_ON ]; then
    log "/opt/mmp/STAY_ON exists: leaving the server on"
    exit "$status"
fi

if [ "$status" -ne 0 ]; then
    # Failed at start-up (bad settings, models missing, Redis or database down). Wait a little so
    # the logs can be read over SSH, then stop anyway: an idle GPU must not run all day.
    log "worker failed: shutting down in 15 minutes (touch /opt/mmp/STAY_ON to keep the server on)"
    sleep 900
    [ -e /opt/mmp/STAY_ON ] && exit "$status"
fi

log "queue empty: shutting the server down"
sudo /sbin/shutdown -h now
