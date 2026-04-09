#!/usr/bin/env bash
#
# Cleanly stop the DP proxy and all backend engines.
#
# Strategy:
#   1. Try lsof to find PIDs listening on the proxy port and backend ports.
#   2. Also use pgrep to find any "server.main --port 90XX" processes that
#      may have been disowned and are not the direct child of this shell
#      (nohup + disown means $! is unreliable after the fact).
#   3. SIGTERM first; wait KILL_TIMEOUT seconds; SIGKILL survivors.
#
# Env vars (all optional):
#   PORT            front-end proxy port  (default: 8765)
#   NUM_RANKS       number of backend ranks  (default: 5)
#   BASE_BACKEND_PORT  first backend port  (default: 9000)
#   KILL_TIMEOUT    grace period in seconds  (default: 15)
#
# Usage:
#   ./scripts/stop_dp.sh
#   PORT=8765 NUM_RANKS=5 ./scripts/stop_dp.sh
#

set -euo pipefail

PORT="${PORT:-8765}"
NUM_RANKS="${NUM_RANKS:-5}"
BASE_BACKEND_PORT="${BASE_BACKEND_PORT:-9000}"
KILL_TIMEOUT="${KILL_TIMEOUT:-15}"

echo "[stop_dp.sh] stopping proxy (port ${PORT}) and ${NUM_RANKS} backends (ports ${BASE_BACKEND_PORT}..$((BASE_BACKEND_PORT + NUM_RANKS - 1)))"

# ---------------------------------------------------------------------------
# Collect PIDs via lsof (port listeners) and pgrep (process name pattern).
# ---------------------------------------------------------------------------
declare -A SEEN_PIDS  # dedup

add_pid() {
    local pid="$1"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        SEEN_PIDS["$pid"]=1
    fi
}

# 1. Find processes listening on the proxy port.
port_pids=$(lsof -ti "tcp:${PORT}" 2>/dev/null || true)
for pid in $port_pids; do add_pid "$pid"; done

# 2. Find processes listening on backend ports.
for i in $(seq 0 $((NUM_RANKS - 1))); do
    bp=$((BASE_BACKEND_PORT + i))
    port_pids=$(lsof -ti "tcp:${bp}" 2>/dev/null || true)
    for pid in $port_pids; do add_pid "$pid"; done
done

# 3. Use pgrep as a safety net: find python processes running server.main on
#    any of our backend ports. This catches processes that were disowned and
#    whose socket may not yet show up in lsof (race after startup), as well as
#    processes whose port is already closed but the process is still winding down.
pgrep_pids=$(pgrep -af "server\.main.*--port 90" 2>/dev/null | awk '{print $1}' || true)
for pid in $pgrep_pids; do add_pid "$pid"; done

# 4. Also catch the proxy process by name.
proxy_pids=$(pgrep -af "server\.dp_proxy" 2>/dev/null | awk '{print $1}' || true)
for pid in $proxy_pids; do add_pid "$pid"; done

PIDS=("${!SEEN_PIDS[@]}")

if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "[stop_dp.sh] nothing to stop — no matching processes found"
    exit 0
fi

echo "[stop_dp.sh] sending SIGTERM to ${#PIDS[@]} process(es): ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
    echo "[stop_dp.sh]   kill -TERM ${pid}"
    kill -TERM "$pid" 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# Wait for graceful exit, then force-kill any survivors.
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
        echo "[stop_dp.sh] timeout — sending SIGKILL to: ${still_alive[*]}"
        for pid in "${still_alive[@]}"; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        break
    fi
    sleep 1
done

echo "[stop_dp.sh] done"
