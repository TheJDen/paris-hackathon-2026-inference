#!/usr/bin/env bash
#
# Cleanly stop the DP=8 proxy and all 8 backend engines.
#
# Sends SIGTERM to any process listening on the proxy port and the 8 backend
# ports, then waits up to KILL_TIMEOUT seconds for them to exit before sending
# SIGKILL as a last resort.
#
# Env vars (all optional):
#   PORT            front-end proxy port  (default: 8765)
#
# Usage:
#   ./scripts/stop_dp.sh
#   PORT=8765 ./scripts/stop_dp.sh
#

set -euo pipefail

PORT="${PORT:-8765}"
BASE_BACKEND_PORT=7001
NUM_RANKS=8
KILL_TIMEOUT="${KILL_TIMEOUT:-15}"

# Build the full list of ports to kill: proxy + 8 backends.
ALL_PORTS=("$PORT")
for i in $(seq 0 $((NUM_RANKS - 1))); do
    ALL_PORTS+=($((BASE_BACKEND_PORT + i)))
done

echo "[stop_dp.sh] stopping proxy (port ${PORT}) and ${NUM_RANKS} backends (ports ${BASE_BACKEND_PORT}...$((BASE_BACKEND_PORT + NUM_RANKS - 1)))"

PIDS=()
for p in "${ALL_PORTS[@]}"; do
    # lsof -ti returns the PID(s) listening on the given TCP port.
    # Ignore errors (port might already be free).
    port_pids=$(lsof -ti "tcp:${p}" 2>/dev/null || true)
    if [ -n "$port_pids" ]; then
        for pid in $port_pids; do
            PIDS+=("$pid")
            echo "[stop_dp.sh] found pid=${pid} on port ${p} — sending SIGTERM"
            kill -TERM "$pid" 2>/dev/null || true
        done
    else
        echo "[stop_dp.sh] port ${p}: no process found (already stopped?)"
    fi
done

if [ ${#PIDS[@]} -eq 0 ]; then
    echo "[stop_dp.sh] nothing to stop"
    exit 0
fi

# Deduplicate pids (a single uvicorn process might own multiple ports).
UNIQUE_PIDS=($(printf '%s\n' "${PIDS[@]}" | sort -u))

# Wait for clean exit, then force-kill any survivors.
deadline=$(( $(date +%s) + KILL_TIMEOUT ))
while true; do
    still_alive=()
    for pid in "${UNIQUE_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            still_alive+=("$pid")
        fi
    done
    if [ ${#still_alive[@]} -eq 0 ]; then
        break
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "[stop_dp.sh] timeout — sending SIGKILL to: ${still_alive[*]}"
        for pid in "${still_alive[@]}"; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        break
    fi
    sleep 1
done

echo "[stop_dp.sh] done"
