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
#   ./scripts/start.sh --tp 8                   # Phase 1+
#
# Env vars (all optional):
#   PORT            port to listen on                          (default: 8000)
#   MODEL           HF model id or local path                  (default: Qwen/Qwen3.5-35B-A3B)
#   LOG_FILE        path to server log file                    (default: <project>/server.log)
#   HEALTH_TIMEOUT  seconds to wait for /health                (default: 600)
#   PYTHON          python interpreter to use                  (default: python)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

PORT="${PORT:-8000}"
MODEL="${MODEL:-Qwen/Qwen3.5-35B-A3B}"
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/server.log}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
PYTHON="${PYTHON:-python}"
PID_FILE="/tmp/server-${PORT}.pid"

# ---------------------------------------------------------------------------
# TP > 1: re-exec under torchrun so every TP rank gets its own python process
# with WORLD_SIZE / RANK / LOCAL_RANK env vars. Rank 0 binds the HTTP port;
# rank >0 enters tp_worker_loop. We parse a leading --tp N out of the user's
# args and use it as nproc_per_node.
# ---------------------------------------------------------------------------
TP_VAL=1
_prev=""
for _arg in "$@"; do
    if [ "$_prev" = "--tp" ]; then
        TP_VAL="$_arg"
    fi
    _prev="$_arg"
done
TP_MASTER_PORT="${TP_MASTER_PORT:-29500}"

# ---------------------------------------------------------------------------
# Bug 3: Fail loud if the port is already in use.
# ---------------------------------------------------------------------------
if lsof -ti "tcp:${PORT}" >/dev/null 2>&1; then
    echo "[start.sh] ERROR: port ${PORT} is already in use — a server may already be running." >&2
    echo "[start.sh] Run:  PORT=${PORT} ./scripts/stop.sh   to stop it first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Bug 4: Print the full command line before launching.
# ---------------------------------------------------------------------------
if [ "$TP_VAL" -gt 1 ]; then
    CMD=("$PYTHON" -m torch.distributed.run \
         --nproc_per_node="$TP_VAL" \
         --master_port="$TP_MASTER_PORT" \
         -m server.main --model "$MODEL" --port "$PORT" "$@")
else
    CMD=("$PYTHON" -m server.main --model "$MODEL" --port "$PORT" "$@")
fi
echo "[start.sh] launching: ${CMD[*]}"
echo "[start.sh] model=${MODEL} port=${PORT} log=${LOG_FILE}"

nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &

NOHUP_PID=$!

# Detach so the server outlives this script.
disown "$NOHUP_PID" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Bug 1: Resolve the actual python PID via pgrep and write a PID file.
# ---------------------------------------------------------------------------
# Give the process a moment to exec so pgrep can see the real argv.
sleep 1
SERVER_PID=$(pgrep -f "server\.main.*--port ${PORT}" 2>/dev/null | head -1 || true)
if [ -z "$SERVER_PID" ]; then
    # Fall back to the shell's $! if pgrep comes up empty (e.g. very fast startup).
    SERVER_PID="$NOHUP_PID"
fi
echo "$SERVER_PID" > "$PID_FILE"
echo "[start.sh] server pid=${SERVER_PID} (pid file: ${PID_FILE})"

# Poll /health until it returns 200 or we time out.
deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
while true; do
    if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[start.sh] /health is up — server ready on port ${PORT}"
        exit 0
    fi
    # Check both the stored pid and the nohup pid so we catch early death.
    if ! kill -0 "$SERVER_PID" 2>/dev/null && ! kill -0 "$NOHUP_PID" 2>/dev/null; then
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
