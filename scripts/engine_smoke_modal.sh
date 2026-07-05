#!/usr/bin/env bash
# Modal engine smoke. Deps + kernels are baked into the Modal image, so this just
# fail-fast checks the kernels, starts the server, and verifies correct output.
# Runs inside the Modal GPU function (cwd=/workspace, HF_HOME=/models Volume).
set -uo pipefail
MODEL="Qwen/Qwen3.5-35B-A3B"; PORT=8080

echo "[smoke] kernel/env check (pre-download)"
python - <<'PY'
import torch, triton, transformers
print("versions -> torch:", torch.__version__, "triton:", triton.__version__, "transformers:", transformers.__version__)
import fla; print("fla:", getattr(fla, "__version__", "?"))          # this import is what crashed on old triton
transformers.Qwen3_5MoeForConditionalGeneration
from transformers.utils.import_utils import (
    is_flash_linear_attention_available as a, is_causal_conv1d_available as b, is_flash_attn_2_available as c)
assert a() and b() and c(), f"kernels fla={a()} causal_conv1d={b()} flash_attn2={c()}"
print("ENV OK -> fla:", a(), "causal_conv1d:", b(), "flash_attn2:", c())
PY

echo "[smoke] starting server (downloads 35B to /models on first run)"
python -m uvicorn engine.server:app --host 0.0.0.0 --port "$PORT" > /tmp/server.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT

echo "[smoke] waiting for /health"
for _ in $(seq 1 360); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health")" = "200" ] && { echo ready; break; }
  sleep 5
done

RESP=$(curl -s "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"What is 2+2? Reply with only the number."}],"max_tokens":16,"temperature":0}')
echo "RESPONSE: $RESP"
CONTENT=$(echo "$RESP" | python -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "")
echo "CONTENT: [$CONTENT]"
if echo "$CONTENT" | grep -q "4" && ! echo "$CONTENT" | grep -qi "<think>"; then
  echo "SMOKE PASS"; exit 0
else
  echo "SMOKE FAIL"; tail -30 /tmp/server.log; exit 1
fi
