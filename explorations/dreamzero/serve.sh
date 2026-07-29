#!/usr/bin/env bash
# Launch the DreamZero single-GPU inference WebSocket server.
#   Usage: bash serve.sh [GPU] [PORT]
set -euo pipefail
GPU="${1:-0}"; PORT="${2:-8901}"
HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT="${CKPT:-/home/tiger/proj/staging/vla/models/DreamZero-DROID}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TMPDIR=/home/tiger/proj/staging/vla/tmp
export TOKENIZERS_PARALLELISM=false
cd "$HERE/repo"
exec "$HERE/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 \
  socket_test_optimized_AR.py --port "$PORT" --enable-dit-cache --model-path "$CKPT"
