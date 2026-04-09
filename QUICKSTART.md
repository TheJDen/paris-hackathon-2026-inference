# QUICKSTART

5-minute walkthrough for evaluators / judges who want to see what this
inference engine does and interact with it.

> **Latest numbers and live profile** are in [`STATUS.md`](STATUS.md) —
> regenerated on every push to main, no need to re-run anything to see
> where we stand.

## What this is

A from-scratch inference engine for **Qwen/Qwen3.5-35B-A3B** (Hybrid
Gated DeltaNet + Sparse MoE, 35B total / 3B active, 256 experts) on
**8×H200**. OpenAI-compatible HTTP server. No vLLM, no SGLang, no
TensorRT — built on PyTorch + low-level libraries.

Current state (Phase 1, single H200, no TP yet) is documented in
[`STATUS.md`](STATUS.md). The full plan is in
[`docs/parallelism.md`](docs/parallelism.md).

## 1. Boot the server (~15 seconds)

On the H200 node:

```bash
cd ~/paris-hackathon-2026-inference

CUDA_VISIBLE_DEVICES=7 \
  PYTHON=.venv/bin/python \
  PORT=8765 \
  MODEL=/mnt/data/$USER/models/Qwen3.5-35B-A3B \
  HEALTH_TIMEOUT=300 \
  ./scripts/start.sh --device cuda:0 --max-batch 64 --batch-window-ms 500
```

`scripts/start.sh` is the **submission entrypoint**: it backgrounds the
server, polls `GET /health`, and exits 0 once the server is ready —
exactly the contract the rules require.

## 2. Verify it's running

```bash
# Health check (returns 200)
curl http://localhost:8765/health

# Single chat completion (the OpenAI-compatible endpoint we're scored on)
curl http://localhost:8765/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "messages": [
      {"role": "user", "content": "What is 17 * 23? Answer with just the number."}
    ],
    "max_tokens": 32,
    "temperature": 0.0
  }' | jq
```

You should get back a standard OpenAI chat completion response with
`choices[0].message.content == "391"` and a populated `usage` object
with verified prompt/completion token counts.

## 3. See the live engine signal

Two endpoints expose the throughput-focused metrics in real time:

```bash
# JSON snapshot of throughput, batch size, fill ratio, etc.
curl -s http://localhost:8765/metrics | jq

# CLI region table — sorted by total time, the primary profiling surface
curl -s http://localhost:8765/metrics/regions
```

The region table looks like this:

```
region                                  n    total_s    mean_ms     p50_ms     p99_ms
-------------------------------------------------------------------------------------
engine.model.generate                   7    340.751  48678.730  39348.461  83337.498
engine.batch.tokenize                   7      0.189     26.956     20.092     60.707
engine.batch.render                     7      0.093     13.291      5.213     68.561
engine.batch.decode                     7      0.065      9.332      7.542     28.387
```

That's "where time is being spent right now," at any granularity we've
instrumented. Wrapping more code with `time_region("name")` adds rows.

## 4. Run the official scoring evals

The harness in `eval/` is upstream and unmodified — use it to verify
correctness and throughput exactly the way the official scoring does.

### API conformance (~5 seconds)

```bash
.venv/bin/python -m eval.check_server --base-url http://localhost:8765
```

Two checks: `GET /health` returns 200, and `POST /v1/chat/completions`
returns a structurally valid OpenAI response. Both must pass.

### Throughput sweep (the number we're optimizing)

```bash
# Quick iteration sweep — c=16, 32, 64 (the three high-weight levels) ~5 min
.venv/bin/python -m bench.quick_throughput --base-url http://localhost:8765

# Full sweep mirroring the official harness (all 7 levels) ~10 min
.venv/bin/python -m bench.quick_throughput \
  --base-url http://localhost:8765 \
  --concurrency 1 2 4 8 16 32 64 \
  --num-requests 64 \
  --with-spot-checks
```

The output prints per-level tok/s, the partial weighted score, the
engine region table, and live metrics. Both versions use the **same
prompt-generation code path** the official `eval/throughput/run_throughput.py`
uses (runtime random vocab tokens via vLLM-style iterative decode-encode
adjustment) so the numbers transfer.

### GSM8K-CoT correctness gate (~3 minutes)

```bash
.venv/bin/python -m eval.correctness.run_correctness \
  --base-url http://localhost:8765 \
  --num-concurrent 32
```

200 problems, gate ≥ 87.5%. Phase 1 currently passes at 92.0% flexible /
91.0% strict (vLLM baseline: 91.5% / 91.0%).

### Combined scoring

```bash
.venv/bin/python -m eval.score \
  --correctness results/correctness_<sha>.json \
  --throughput  results/throughput_<sha>.json
```

Computes the official weighted-tok/s score gated on correctness ≥ 87.5%.

## 5. Refresh the published status

A single command rewrites [`STATUS.md`](STATUS.md) with the latest
numbers and a fresh torch.profiler trace:

```bash
.venv/bin/python -m bench.refresh_status --base-url http://localhost:8765
```

This drives the same quick throughput sweep as above, fetches the
engine's `/metrics` and `/metrics/regions`, picks up the freshest
torch.profiler artifact in `profiles/`, and embeds the top kernel
hotspots in STATUS.md. Run this before every push to main.

## 6. Capture a fresh PyTorch profile (optional, for diagnosis)

The engine can one-shot capture a single steady-state batch under
`torch.profiler.profile` and dump:

- `profiles/torch_<tag>_<ts>.json.gz` — Chrome trace (open in Perfetto / `chrome://tracing`)
- `profiles/torch_<tag>_<ts>.txt` — sorted op table by self CUDA time
- `profiles/torch_<tag>_<ts>.summary.json` — top-30 kernels structured (consumed by `refresh_status`)

To arm: launch the server with `--profile-torch-after-batches N`, then
send N+M requests. The engine captures batch N+1 and disables itself.
The chrome trace lets you see kernel launches, H2D syncs, attention
kernel bookends, MoE expert dispatch, etc.

## What to read next

| If you want to know... | Read |
|---|---|
| **Where we are right now** (numbers, profile, top kernels) | [`STATUS.md`](STATUS.md) |
| Setup, dependency landmines, full iteration loop | [`HACKING.md`](HACKING.md) |
| Why the parallelism choice for Phase 2 looks the way it does | [`docs/parallelism.md`](docs/parallelism.md) |
| The architecture facts and load path for the model | [`engine/model/qwen3_next.py`](engine/model/qwen3_next.py) |
| The core engine loop (batcher, generate, profile capture) | [`engine/runtime/engine.py`](engine/runtime/engine.py) |
