#!/usr/bin/env bash
#
# Generate baseline numbers using vLLM on 8xH200.
#
# Prerequisites:
#   pip install -e ".[baseline]"
#
# Usage:
#   ./baseline/run_baseline.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "$RESULTS_DIR"

MODEL="Qwen/Qwen3.5-35B-A3B"
PORT=8000
BASE_URL="http://localhost:${PORT}"

echo "=== Baseline Generation ==="
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo ""

# --- Launch vLLM ---
echo "Starting vLLM server..."
vllm serve "$MODEL" \
    --tensor-parallel-size 8 \
    --port "$PORT" \
    --max-model-len 4096 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --gpu-memory-utilization 0.90 \
    &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

cleanup() {
    echo "Stopping vLLM server (PID: ${VLLM_PID})..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# --- Wait for server ---
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
        echo "Server ready after ~$((i * 5))s"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died"
        exit 1
    fi
    sleep 5
done

# Verify server is actually up
if ! curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
    echo "ERROR: Server not ready after 10 minutes"
    exit 1
fi

# --- Health check ---
echo ""
echo "=== Running health check ==="
python -m eval.check_server --base-url "${BASE_URL}"

# --- Correctness eval ---
echo ""
echo "=== Running correctness evaluation ==="
python -m eval.correctness.run_correctness \
    --base-url "${BASE_URL}" \
    --output "${RESULTS_DIR}/correctness_baseline.json" \
    --output-dir "${RESULTS_DIR}/correctness_raw"

# --- Throughput benchmark ---
echo ""
echo "=== Running throughput benchmark ==="
python -m eval.throughput.run_throughput \
    --base-url "${BASE_URL}" \
    --output "${RESULTS_DIR}/throughput_baseline.json"

echo ""
echo "=== Baseline generation complete ==="
echo "Results saved to ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
