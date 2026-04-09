#!/usr/bin/env bash
#
# Submission entrypoint for DP=8 mode.
#
# Launches 8 independent engine instances (one per GPU), waits for all to
# come up healthy, then launches the round-robin proxy in front of them.
# Exits 0 leaving every process running — matches the start.sh contract.
#
# Env vars (all optional):
#   PORT            front-end proxy port           (default: 8765)
#   MODEL           HF model id or local path       (default: Qwen/Qwen3.5-35B-A3B)
#   MAX_BATCH       --max-batch per backend          (default: 32)
#   MAX_MODEL_LEN   --max-model-len per backend      (default: 4096)
#   HEALTH_TIMEOUT  seconds to wait per backend      (default: 600)
#   PYTHON          python interpreter to use        (default: python)
#   PROFILE_RANK    if set (0-7), pass one-shot torch profile flags to that rank
#
# Usage:
#   ./scripts/start_dp.sh
#   PORT=8765 MAX_BATCH=64 ./scripts/start_dp.sh
#   PROFILE_RANK=0 ./scripts/start_dp.sh   # profile rank 0's first big batch
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PORT="${PORT:-8765}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
MAX_BATCH="${MAX_BATCH:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
PYTHON="${PYTHON:-python}"

NUM_RANKS=8
BASE_BACKEND_PORT=7001

echo "[start_dp.sh] launching DP=${NUM_RANKS} engine — model=${MODEL} proxy_port=${PORT} max_batch=${MAX_BATCH} max_model_len=${MAX_MODEL_LEN}"

# ---------------------------------------------------------------------------
# 1. Launch one engine instance per GPU
# ---------------------------------------------------------------------------
for i in $(seq 0 $((NUM_RANKS - 1))); do
    BACKEND_PORT=$((BASE_BACKEND_PORT + i))

    EXTRA_ARGS=""
    if [ -n "${PROFILE_RANK:-}" ] && [ "$i" -eq "$PROFILE_RANK" ]; then
        EXTRA_ARGS="--profile-torch-after-batches 30 --profile-torch-min-batch-size 4 --profile-tag dp_rank${PROFILE_RANK}"
        echo "[start_dp.sh] rank ${i}: profiling enabled (${EXTRA_ARGS})"
    fi

    echo "[start_dp.sh] rank ${i}: CUDA_VISIBLE_DEVICES=${i} port=${BACKEND_PORT}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$i nohup "$PYTHON" -m server.main \
        --model "$MODEL" \
        --port "$BACKEND_PORT" \
        --device cuda:0 \
        --max-batch "$MAX_BATCH" \
        --max-model-len "$MAX_MODEL_LEN" \
        $EXTRA_ARGS \
        >> server.dp.log 2>&1 &
    disown $! 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# 2. Wait for ALL 8 backends to become healthy
# ---------------------------------------------------------------------------
echo "[start_dp.sh] waiting for all ${NUM_RANKS} backends to come up (timeout ${HEALTH_TIMEOUT}s each) ..."

for i in $(seq 0 $((NUM_RANKS - 1))); do
    BACKEND_PORT=$((BASE_BACKEND_PORT + i))
    deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
    echo -n "[start_dp.sh] rank ${i} (port ${BACKEND_PORT}): "
    while true; do
        if curl -fsS "http://localhost:${BACKEND_PORT}/health" >/dev/null 2>&1; then
            echo "up"
            break
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo ""
            echo "[start_dp.sh] ERROR: rank ${i} did not come up within ${HEALTH_TIMEOUT}s" >&2
            echo "[start_dp.sh] last 50 lines of server.dp.log:" >&2
            tail -n 50 server.dp.log >&2 || true
            exit 1
        fi
        sleep 2
    done
done

echo "[start_dp.sh] all ${NUM_RANKS} backends healthy"

# ---------------------------------------------------------------------------
# 3. Build the --backends argument list for the proxy
# ---------------------------------------------------------------------------
BACKEND_URLS=""
for i in $(seq 0 $((NUM_RANKS - 1))); do
    BACKEND_PORT=$((BASE_BACKEND_PORT + i))
    BACKEND_URLS="${BACKEND_URLS} http://localhost:${BACKEND_PORT}"
done
# trim leading space
BACKEND_URLS="${BACKEND_URLS# }"

# ---------------------------------------------------------------------------
# 4. Launch the proxy
# ---------------------------------------------------------------------------
echo "[start_dp.sh] launching proxy on port ${PORT} → backends: ${BACKEND_URLS}"
# shellcheck disable=SC2086
nohup "$PYTHON" -m server.dp_proxy \
    --port "$PORT" \
    --backends $BACKEND_URLS \
    >> proxy.log 2>&1 &
PROXY_PID=$!
disown "$PROXY_PID" 2>/dev/null || true
echo "[start_dp.sh] proxy pid=${PROXY_PID}"

# ---------------------------------------------------------------------------
# 5. Wait for the proxy's /health to return 200
# ---------------------------------------------------------------------------
PROXY_DEADLINE=$(( $(date +%s) + 30 ))
while true; do
    if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[start_dp.sh] proxy /health is up — DP=8 server ready on port ${PORT}"
        exit 0
    fi
    if [ "$(date +%s)" -ge "$PROXY_DEADLINE" ]; then
        echo "[start_dp.sh] ERROR: proxy did not come up within 30s" >&2
        tail -n 30 proxy.log >&2 || true
        exit 1
    fi
    sleep 1
done
