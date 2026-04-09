#!/usr/bin/env bash
#
# Submission entrypoint.
#
# Boots the engine server, waits for /health to return 200, then exits 0
# leaving the server running. The submission rules require this script
# to "exit cleanly and leave the server running" — that's exactly the
# pattern below.
#
# Usage:
#   ./scripts/start.sh                          # default model, port 8000
#   PORT=9000 ./scripts/start.sh --stub         # Phase 0 smoke run
#   ./scripts/start.sh --tp 8                   # Phase 1 TP (unused)
#   EP=8 ./scripts/start.sh                     # Expert Parallelism across 8 GPUs
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PORT="${PORT:-8888}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/server.log}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"

PYTHON="${PYTHON:-python}"

# EP=8 uses torchrun to spawn one process per GPU; each rank loads the model
# with only its shard of expert weights, cutting GPU memory from ~70 GB to
# ~17.5 GB per rank and parallelising the MoE forward across 8 GPUs.
EP="${EP:-8}"

if [ "$EP" -gt 1 ]; then
    echo "[start.sh] launching EP=${EP} server: model=${MODEL} port=${PORT} log=${LOG_FILE}"
    nohup torchrun \
        --standalone \
        --nproc_per_node="$EP" \
        -m server.main \
        --model "$MODEL" \
        --port "$PORT" \
        "$@" \
        >"$LOG_FILE" 2>&1 &
else
    echo "[start.sh] launching server: model=${MODEL} port=${PORT} log=${LOG_FILE}"
    nohup "$PYTHON" -m server.main \
        --model "$MODEL" \
        --port "$PORT" \
        "$@" \
        >"$LOG_FILE" 2>&1 &
fi

SERVER_PID=$!  # works for both nohup paths above
echo "[start.sh] server pid=${SERVER_PID}"

# Detach so the server outlives this script.
disown "$SERVER_PID" 2>/dev/null || true

# Poll /health until it returns 200 or we time out.
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
while true; do
    if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[start.sh] /health is up — server ready on port ${PORT}"
        exit 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[start.sh] ERROR: server process died — see ${LOG_FILE}" >&2
        tail -n 50 "$LOG_FILE" >&2 || true
        exit 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "[start.sh] ERROR: /health did not come up within ${HEALTH_TIMEOUT}s" >&2
        tail -n 50 "$LOG_FILE" >&2 || true
        exit 1
    fi
    sleep 1
done
