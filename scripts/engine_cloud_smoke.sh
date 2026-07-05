#!/usr/bin/env bash
# Cloud smoke for OUR engine: install the stack on the (x86) instance, start the
# server, and verify it produces a correct answer. Fail-fast on env/kernels
# BEFORE the ~70GB model download so a broken stack is cheap to discover.
#
# Runs from the repo root (cloudbench rsyncs source there). ngpus=1 is enough:
# the 35B (~70GB) fits on one H200 (141GB).
set -euo pipefail
MODEL="Qwen/Qwen3.5-35B-A3B"
PORT=8080
ART="${ART_DIR:-artifacts}"; mkdir -p "$ART"

log() { echo "[smoke $(date +%H:%M:%S)] $*"; }

# ---------- 1. env: uv + torch + engine deps + fast kernels ----------
log "installing uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
log "creating venv + installing torch and engine deps"
uv venv /tmp/engine-venv --python 3.12
# shellcheck disable=SC1091
source /tmp/engine-venv/bin/activate
uv pip install torch                       # platform torch (x86 cuda wheel)
uv pip install -e ".[engine]"              # fastapi/uvicorn/transformers/pydantic/hf_hub/safetensors
# fast kernels (x86 = prebuilt/fast; FLA from git; defensive kernels dance for causal_conv1d)
uv pip install --no-deps "git+https://github.com/fla-org/flash-linear-attention" einops
uv pip install --no-build-isolation flash-attn
uv pip install kernels \
  && uv pip install --no-build-isolation causal_conv1d \
  && uv pip uninstall kernels kernels-data

# ---------- 2. FAIL-FAST: model class imports + all kernel flags green (no download) ----------
log "verifying model import + kernel availability (pre-download)"
python - <<'PY'
import transformers
transformers.Qwen3_5MoeForConditionalGeneration  # triggers modeling import
from transformers.utils.import_utils import (
    is_flash_linear_attention_available as fla,
    is_causal_conv1d_available as cc,
    is_flash_attn_2_available as fa,
)
assert fla() and cc() and fa(), f"kernel flags: fla={fla()} causal_conv1d={cc()} flash_attn2={fa()}"
print("ENV OK -> fla:", fla(), "causal_conv1d:", cc(), "flash_attn2:", fa())
PY

# ---------- 3. start the server (this downloads + loads the 35B) ----------
log "starting server (downloads + loads ${MODEL})"
python -m uvicorn engine.server:app --host 0.0.0.0 --port "$PORT" > "$ART/server.log" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# ---------- 4. wait for /health (download + load can take ~20 min) ----------
log "waiting for /health"
READY=0
for _ in $(seq 1 360); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" || true)
  if [ "$code" = "200" ]; then READY=1; log "server ready"; break; fi
  sleep 5
done
if [ "$READY" != "1" ]; then log "server never became ready"; tail -40 "$ART/server.log" || true; exit 1; fi

# ---------- 5. correctness check: '2+2' -> contains 4, no <think> ----------
log "sending completion"
REQ='{"model":"'"$MODEL"'","messages":[{"role":"user","content":"What is 2+2? Reply with only the number."}],"max_tokens":16,"temperature":0}'
RESP=$(curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ")
echo "$RESP" | tee "$ART/response.json"
CONTENT=$(echo "$RESP" | python -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])")
log "content: [$CONTENT]"

if echo "$CONTENT" | grep -q "4" && ! echo "$CONTENT" | grep -qi "<think>"; then
  log "SMOKE PASS"; exit 0
else
  log "SMOKE FAIL (want a '4', no <think> tags)"; exit 1
fi
