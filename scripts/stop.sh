#!/usr/bin/env bash
#
# Stop a single-engine server started by scripts/start.sh.
#
# Strategy:
#   1. Read /tmp/server-$PORT.pid if present.
#   2. Fall back to pgrep -f "server.main.*--port $PORT".
#   3. SIGTERM; wait KILL_TIMEOUT seconds; SIGKILL survivors.
#   4. Verify the port is no longer listening.
#
# Env vars (all optional):
#   PORT            port the server is listening on  (default: 8000)
#   KILL_TIMEOUT    grace period before SIGKILL       (default: 5)
#
# Usage:
#   ./scripts/stop.sh
#   PORT=9000 ./scripts/stop.sh
#
set -euo pipefail

PORT="${PORT:-8000}"
KILL_TIMEOUT="${KILL_TIMEOUT:-5}"
PID_FILE="/tmp/server-${PORT}.pid"

echo "[stop.sh] stopping server on port ${PORT}"

# ---------------------------------------------------------------------------
# Collect PIDs.
# ---------------------------------------------------------------------------
declare -A SEEN_PIDS

add_pid() {
    local pid="$1"
    if [[ -n "$pid" ]] && [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        SEEN_PIDS["$pid"]=1
    fi
}

# 1. PID file written by start.sh.
if [ -f "$PID_FILE" ]; then
    file_pid=$(cat "$PID_FILE")
    echo "[stop.sh] found pid file ${PID_FILE} — pid=${file_pid}"
    add_pid "$file_pid"
fi

# 2. pgrep fallback — catches processes started without the pid file.
pgrep_pids=$(pgrep -f "server\.main.*--port ${PORT}" 2>/dev/null || true)
for pid in $pgrep_pids; do add_pid "$pid"; done

# 3. lsof: anything actually listening on the port (catches proxy wrappers etc.).
lsof_pids=$(lsof -ti "tcp:${PORT}" 2>/dev/null || true)
for pid in $lsof_pids; do add_pid "$pid"; done

PIDS=("${!SEEN_PIDS[@]}")

if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "[stop.sh] nothing to stop — no matching processes found for port ${PORT}"
    # Clean up stale pid file if present.
    rm -f "$PID_FILE"
    exit 0
fi

echo "[stop.sh] sending SIGTERM to ${#PIDS[@]} process(es): ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
    echo "[stop.sh]   kill -TERM ${pid}"
    kill -TERM "$pid" 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# Wait for graceful exit, then SIGKILL survivors.
# ---------------------------------------------------------------------------
deadline=$(( $(date +%s) + KILL_TIMEOUT ))
while true; do
    still_alive=()
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            still_alive+=("$pid")
        fi
    done
    if [ "${#still_alive[@]}" -eq 0 ]; then
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "[stop.sh] timeout — sending SIGKILL to: ${still_alive[*]}"
        for pid in "${still_alive[@]}"; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        break
    fi
    sleep 1
done

# Clean up pid file.
rm -f "$PID_FILE"

# ---------------------------------------------------------------------------
# Verify the port is released.
# ---------------------------------------------------------------------------
sleep 1
if lsof -ti "tcp:${PORT}" >/dev/null 2>&1; then
    echo "[stop.sh] WARNING: port ${PORT} is still in use after kill — check manually." >&2
    exit 1
fi

echo "[stop.sh] done — port ${PORT} is free"
