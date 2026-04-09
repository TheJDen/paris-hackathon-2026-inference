#!/usr/bin/env bash
#
# Download Qwen3.5-35B-A3B weights to a node-local cache.
#
# Run this on the H200 node (`ssh server`), not the laptop.
# Override HF_HOME to point at fast local SSD if /root/.cache is on a slow disk.
#
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
LOCAL_DIR="${LOCAL_DIR:-${HOME}/models/$(basename "$MODEL")}"

mkdir -p "$LOCAL_DIR"

echo "[download] model=${MODEL} dst=${LOCAL_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Use the project venv's python so we don't fight with system pip.
PYTHON="${PYTHON:-${PROJECT_DIR}/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python3"
fi

# Enable hf_transfer if available — multi-stream downloads, much faster on
# fast networks. Install it once via: uv pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

"$PYTHON" - "$MODEL" "$LOCAL_DIR" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

model, local_dir = sys.argv[1], sys.argv[2]
print(f"[download] python={sys.executable}")
print(f"[download] hf_transfer={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
path = snapshot_download(
    repo_id=model,
    local_dir=local_dir,
    # Skip vision/video preprocessor configs we don't need for text inference,
    # but keep all weight shards and the chat template.
    ignore_patterns=[
        "*.bin",            # we want safetensors only
        "*.gguf",
        "*.msgpack",
        "*.h5",
        "*.onnx*",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ],
    max_workers=8,
)
print(f"[download] cached at: {path}")
PY

echo "[download] done — set MODEL_PATH=${LOCAL_DIR} when launching the server"
