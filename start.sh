#!/usr/bin/env bash
set -euo pipefail

: "${DEEPINFER_ENV:?set DEEPINFER_ENV to the DeepInfer virtualenv activate script}"
: "${GATEWAY_MODEL:?set GATEWAY_MODEL to the DeepSeek checkpoint directory}"
: "${DEEPINFER_REPO:?set DEEPINFER_REPO to the DeepInfer checkout}"

source "$DEEPINFER_ENV"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$DEEPINFER_REPO:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MASTER_PORT="${MASTER_PORT:-24555}"
export GATEWAY_GPUS="${GATEWAY_GPUS:-0,1,2,3}"
export PYTHONUNBUFFERED=1

exec python -m uvicorn server:app \
  --app-dir "$REPO_DIR" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --log-level info
