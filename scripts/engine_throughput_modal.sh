#!/usr/bin/env bash
# H200 engine graph verification + profile traces at batch 64 AND batch 8.
# One model load, two short profiled runs (16 decode steps each) so the traces
# stay small and export fast. Verifies the CUDA-graph engine runs on H200 and
# lets us compare the decode packing at high vs low batch.
#   -> profile/trace_b64.json.gz, profile/trace_b8.json.gz
# Artifacts come back via the cloudbench envelope (staged even on a graceful timeout).
set -uo pipefail
PORT=8080
ART="${CB_ARTIFACTS_DIR:-artifacts}"
export ENGINE_PROFILER_DIR="$ART/profile"
# NOTE: do NOT set KINETO_USE_DAEMON — it's presence-checked, so ANY value (even "0")
# turns the dynolog daemon ON, which spams IpcFabric errors and yields empty traces.
# Unset = daemon off = real traces, matching the local env that works.
export TRITON_CACHE_DIR="/models/triton"   # persist Triton compile/tune cache on the volume
mkdir -p "$ART" "$ENGINE_PROFILER_DIR" "$TRITON_CACHE_DIR"

echo "[thru] starting server"
python -m uvicorn engine.serving:app --host 0.0.0.0 --port "$PORT" > /tmp/server.log 2>&1 &
PID=$!
cleanup() { kill "$PID" 2>/dev/null || true; cp /tmp/server.log "$ART/server.log" 2>/dev/null || true; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM INT

echo "[thru] waiting for /health (model load + warmup + graph capture)"
for i in $(seq 1 240); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health")" = "200" ] && { echo "[thru] ready after ~$((i*5))s"; break; }
  [ $((i % 6)) -eq 0 ] && echo "[thru] ...still loading (~$((i*5))s elapsed)"
  sleep 5
done

# profile_at CONC NAME: run a short profiled sweep at concurrency=CONC (one wave, so
# the active batch sits at CONC), then wait for the export and rename the trace.
profile_at() {
  local conc="$1" name="$2"
  echo "[thru] === PROFILE batch=$conc -> trace_${name} ==="
  curl -s -X POST "http://localhost:$PORT/start_profile"; echo
  python -m eval.throughput.run_throughput --base-url "http://localhost:$PORT" \
    --concurrency "$conc" --num-requests "$conc" --input-tokens 64 --max-tokens 16 \
    --output "$ART/throughput_${name}.json"
  curl -s -X POST "http://localhost:$PORT/stop_profile"; echo
  echo "[thru] waiting for trace export..."
  for _ in $(seq 1 60); do
    T=$(ls -t "$ENGINE_PROFILER_DIR"/trace-*.json.gz 2>/dev/null | head -1)
    [ -n "$T" ] && { mv "$T" "$ENGINE_PROFILER_DIR/trace_${name}.json.gz"; echo "[thru] wrote trace_${name}.json.gz ($(du -h "$ENGINE_PROFILER_DIR/trace_${name}.json.gz"|cut -f1))"; break; }
    sleep 2
  done
}

profile_at 64 b64
profile_at 8  b8

echo "[thru] artifacts:"; ls -la "$ENGINE_PROFILER_DIR" 2>/dev/null
echo "DONE"
