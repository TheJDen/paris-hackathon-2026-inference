# HACKING

Setup, commands, and conventions. **For latest stats and the current
implementation, see [`STATUS.md`](STATUS.md)** — auto-regenerated on
every push by `bench/refresh_status.py`. This document is the slow-changing
how-to.

## Setup on the H200 box (one-time)

The system Python is missing `Python.h`, which silently breaks Triton
and pushes `flash-linear-attention` into a CPU fallback that crashes
inside the model forward. **Use a uv-managed Python.**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv python install 3.12

git clone git@github.com:TheJDen/paris-hackathon-2026-inference.git
cd paris-hackathon-2026-inference
uv venv --python 3.12
uv pip install -e ".[engine]" hf_transfer accelerate flash-linear-attention

# Optional but worth it (~9 min one-shot build):
CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=16 \
  uv pip install --no-build-isolation causal-conv1d

./scripts/download_weights.sh   # ~67 GB → ~/models/Qwen3.5-35B-A3B
```

### Dependency landmines (battle scars)

- **`Python.h` missing** with system Python → fla detects "cpu" → crash. Fix: uv-managed Python.
- **`accelerate`** is required for HF `device_map=...`.
- **`causal-conv1d`** wants to build a CUDA ext; uv's build env pulls in cu128 torch which mismatches the cu130 we have. Fix: `--no-build-isolation`.
- **`fla` 0.4.2** has a `torch.cpu.device(...)` call that only fires when triton fell back to CPU. Resolved by the `Python.h` fix above.

## Launch the server

```bash
CUDA_VISIBLE_DEVICES=7 \
  PYTHON=.venv/bin/python \
  PORT=8765 \
  MODEL=/mnt/data/$USER/models/Qwen3.5-35B-A3B \
  HEALTH_TIMEOUT=300 \
  ./scripts/start.sh \
    --device cuda:0 \
    --max-batch 64 \
    --batch-window-ms 500 \
    --profile-torch-after-batches 3   # optional, see below
```

`scripts/start.sh` backgrounds uvicorn, polls `/health`, exits 0 leaving
the server running — the contract submission requires.

GPU 7 is the least-contested on this shared box; pick a free index.
`--profile-torch-after-batches 3` arms a one-shot torch profile capture
of batch 4 (i.e. one steady-state batch after warmup).

## The iteration loop

**Default per-change loop (≈1-5 min):**

```bash
.venv/bin/python -m bench.quick_throughput --base-url http://localhost:8765
```

Runs `c=16, 32, 64` (the levels worth 16/22 = **73% of the score**),
32 reqs/level, runtime random prompts via the same code path the
official scoring uses (`eval.throughput.run_throughput.run_concurrency_level`).
Prints per-level tok/s, partial weighted score, the engine region table,
and live `/metrics`. Spot checks are off by default — they fragment the
batches into greedy/sampled sub-batches and add noise. Use
`--with-spot-checks` to mirror the official harness exactly.

**Pre-push refresh (the only command teammates need to memorize):**

```bash
.venv/bin/python -m bench.refresh_status --base-url http://localhost:8765
git add STATUS.md profiles/
git commit -m "..."
git push
```

`refresh_status` runs the quick throughput, fetches `/metrics` +
`/metrics/regions`, picks up the freshest `profiles/torch_*.summary.json`
left by the engine, extracts the top-12 kernels by self CUDA time, and
**rewrites `STATUS.md`** so teammates always see the latest commit's
implementation + numbers + profile in one place. The implementation
summary itself lives in `bench/.status_intro.md` — update it manually
when the engine architecture meaningfully changes.

**Pre-merge gate (≈3 min, only before merging to main):**

```bash
.venv/bin/python -m eval.correctness.run_correctness \
  --base-url http://localhost:8765 \
  --num-concurrent 32 \
  --output results/correctness_<sha>.json \
  --output-dir results/correctness_raw_<sha>
```

Full 200-problem GSM8K-CoT, gate ≥ 87.5%. **Do not run per-iteration** —
it's not the loss function we're optimizing, and it costs ~3 minutes.

**Final scoring helper:**

```bash
.venv/bin/python -m eval.score \
  --correctness results/correctness_<sha>.json \
  --throughput  results/throughput_<sha>.json
```

## Profiling — start with the CLI region table

`engine/runtime/profiling.py` defines `RegionTimer`. Wrap any chunk:

```python
from engine.runtime.profiling import time_region
with time_region("moe.grouped_gemm"):
    ...
```

Dump the table any time:

```bash
curl -s http://localhost:8765/metrics/regions
```

That's the **default** profiling surface. `torch.profiler` is wired but
reserved for drilling **inside** a region the CLI table has already
flagged. To capture: launch with `--profile-torch-after-batches N` and
the engine will one-shot-capture batch N+1 inside
`torch.profiler.profile`, exporting:

- `profiles/torch_<tag>_<ts>.json.gz` — Chrome trace (open in Perfetto)
- `profiles/torch_<tag>_<ts>.txt` — sorted-by-self-CUDA-time op table
- `profiles/torch_<tag>_<ts>.summary.json` — top kernels structured (consumed by `refresh_status`)

## Coordinating

- **Server port**: I use `8765`. Teammate `mghanmi` uses `8000`. Pick a free port.
- **GPU**: pick a low-contention index with `nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv`.
- **Weights cache**: `/mnt/data/$USER/models/Qwen3.5-35B-A3B`.
- **Profiles**: `profiles/` is gitignored except the README and the structured `*.summary.json` files. Drop your captures there.
