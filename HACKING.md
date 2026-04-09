# HACKING

How to get the engine running and what's been built so far. Read this first.

The official rules + scoring live in [`README.md`](README.md). This document is
about *our* implementation — directory layout, dependency gotchas, how to launch
the server, how to iterate, and what's done vs what's next.

## TL;DR — what works today

- **Server**: FastAPI app at `server/app.py`, OpenAI-compatible
  `POST /v1/chat/completions` + `GET /health`. Passes `eval/check_server.py`
  against both the laptop stub and the real GPU engine.
- **Engine**: `engine/runtime/engine.py` — loads `Qwen3_5MoeForCausalLM`
  (text-only branch of the multimodal `Qwen3_5MoeForConditionalGeneration`
  checkpoint) via HuggingFace `transformers` 5.5.1, runs **static
  microbatching** on a single H200. No paged KV, no continuous batching, no TP
  yet (Phase 2).
- **Tokenizer / chat template**: `engine/tokenizer/chat_template.py` wraps
  `apply_chat_template(..., enable_thinking=False)`. Important: Qwen3.5
  implements thinking-disabled by inserting an *empty* `<think>\n\n</think>\n\n`
  block in the rendered prompt — that is the correct behavior, do not assert
  it away. The output side is guarded with `strip_think`.
- **Profiling, primary**: `engine/runtime/profiling.py` defines a
  `RegionTimer` exposed via `time_region("name")` / `@timed("name")`. Per-region
  call counts, mean, p50, p99 are dumped via `GET /metrics/regions` (or on
  shutdown to stdout). This is the **default** profiling surface — every
  optimization decision starts here.
- **Profiling, secondary**: `torch.profiler` is wired but only enabled with
  `--profile-torch --profile-window=10:20`. Use it to drill *inside* a region
  the CLI table has already flagged. Don't reach for it by default.
- **Live metrics**: `engine/runtime/metrics.py` always-on counters at
  `GET /metrics`: tok/s in/out, request p50/p99, batch hist, KV occupancy
  (populated in Phase 2+).

Phase 0 conformance + Phase 1 correctness are both green:

- `eval/check_server.py` → both checks PASS against the real engine.
- **Full GSM8K gate (200 problems, num_concurrent=32, microbatch=32):
  92.0% flexible-extract / 91.0% strict-match.** Gate is 87.5%; vLLM's
  reported baseline is 91.5% / 91.0%, so we're tied. Wall time 3:01.
- Mini GSM8K (limit=20, num_concurrent=32) → 85% on a randomly seeded
  20-problem subset (small-sample noise — full 200 is the source of truth).

## Status: where we are in the plan

| Phase | Goal | Status |
|---|---|---|
| 0 | Server skeleton + chat template + stub engine, laptop-side | ✅ done |
| 1 | HF reference on a single H200, pass GSM8K ≥ 87.5% | ✅ done — 92.0% flex / 91.0% strict, full 200-problem run, wall 3:01 |
| 2 | Continuous batching, paged KV, DeltaNet state cache, **TP=8**, real perf | next |
| 3 | Custom Helion kernels for MoE grouped GEMM + DeltaNet recurrence | after |
| 4 | TP vs hybrid TP+EP for MoE, decided on profile data | after |
| 5 | Helion autotuning, scheduler tuning, CUDA graphs, prefix cache | last |

The throughput target is vLLM's **12,810 tok/s @ c=64** (see README baseline).
Phase 1 is correctness only — do not optimize Phase 1, it's a stepping stone.

## Repository layout (added by us — `eval/` and `baseline/` are upstream)

```
engine/
  config.py                       # (placeholder)
  model/
    qwen3_next.py                 # HF Qwen3_5MoeForCausalLM loader + arch facts
  runtime/
    engine.py                     # Engine: stub + Phase 1 microbatcher
    metrics.py                    # always-on counters
    profiling.py                  # RegionTimer (CLI-first), torch.profiler hookable
  tokenizer/
    chat_template.py              # apply_chat_template wrapper, enable_thinking=False
server/
  app.py                          # FastAPI app, /health + /v1/chat/completions + /metrics{,/regions}
  main.py                         # entrypoint: parses CLI, runs uvicorn
  protocol.py                     # pydantic models pinned to eval/check_server expectations
scripts/
  start.sh                        # SUBMISSION ENTRYPOINT — backgrounds server, waits /health, exit 0
  download_weights.sh             # snapshot_download with hf_transfer
  smoke_gen.py                    # single-prompt smoke test bypassing the server
profiles/
  README.md                       # naming convention for profile artifacts
HACKING.md                        # this file
```

## Architecture facts (Qwen3.5-35B-A3B text branch)

Captured from `cfg.text_config`:

| | |
|---|---|
| Total / active params | 35B / 3B |
| Layers | 40 |
| Layer pattern | groups of `[linear, linear, linear, full]` → 30 linear-attention (Gated DeltaNet) + 10 full-attention layers |
| `full_attention_interval` | 4 |
| hidden_size | 2048 |
| Attention heads | 16 (GQA, 2 KV heads, head_dim=256) |
| `attn_output_gate` | True |
| Linear-attn (DeltaNet) | 32 value heads, 16 key heads, head_dim=128, conv_kernel=4, mamba SSM in fp32 |
| MoE | 256 experts, 8 active, moe_intermediate_size=512, +1 shared expert (intermediate=512) |
| Vocab | 248320 |
| Max context | 262144 |
| RoPE | mrope_interleaved (multimodal RoPE), theta=1e7, partial_rotary_factor=0.25 |
| Multi-token prediction | `mtp_num_hidden_layers=1` (interesting for spec-decode in Phase 5) |

The **multimodal checkpoint** ships as `Qwen3_5MoeForConditionalGeneration`
(text + vision + video). For our text-only inference, we instantiate
`Qwen3_5MoeForCausalLM` directly against the same checkpoint — vision tower
weights are silently ignored, saving GPU memory.

## Setup from scratch on the H200 box

The node has 8x H200 (140 GB each), driver 580.126, CUDA 13.0, Python 3.12.3
in `/usr/bin`. **Do not use the system Python** — it ships without `Python.h`
and Triton fails to compile its CUDA utility at import time, which silently
breaks `flash-linear-attention`. Use a uv-managed Python instead.

```bash
# 1. uv (per-user, no sudo)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

# 2. uv-managed CPython 3.12 (this one HAS Python.h)
uv python install 3.12

# 3. clone + venv + deps
git clone git@github.com:TheJDen/paris-hackathon-2026-inference.git
cd paris-hackathon-2026-inference
uv venv --python 3.12
uv pip install -e ".[engine]" hf_transfer accelerate flash-linear-attention

# 4. weights (~67 GB, ~1 min on this network)
./scripts/download_weights.sh
# → /mnt/data/$USER/models/Qwen3.5-35B-A3B
```

### Dependency landmines we already hit

1. **`Python.h` missing** with system Python → Triton fallback to CPU →
   `flash-linear-attention` thinks the device is CPU → crashes with
   `module 'torch.cpu' has no attribute 'device'`. **Fix**: use
   `uv python install 3.12` so headers come along.
2. **`accelerate` is required** for HF `from_pretrained(device_map=...)`.
   Plain `pip install accelerate` works.
3. **`causal-conv1d`** wants to build a CUDA extension and the default
   uv build env pulls in a torch wheel built against cu128 which then
   refuses to compile against the cu130 toolchain. The fix is
   `--no-build-isolation` so the build sees our installed cu130 torch:
   ```bash
   CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=16 \
     uv pip install --no-build-isolation causal-conv1d
   ```
   The build takes ~9 minutes (it compiles for many sm targets even with
   `TORCH_CUDA_ARCH_LIST` set — setup.py overrides). Already installed
   on the box.
4. **`fla` 0.4.2** needs the Python.h fix above. Once that's in, it
   detects CUDA + Hopper correctly and the DeltaNet layers stop running
   on the slow torch fallback.

## Launching the server

### Real engine (one H200)

```bash
CUDA_VISIBLE_DEVICES=0 \
  PYTHON=.venv/bin/python \
  PORT=8765 \
  MODEL=/mnt/data/$USER/models/Qwen3.5-35B-A3B \
  HEALTH_TIMEOUT=300 \
  ./scripts/start.sh \
    --device cuda:0 \
    --max-batch 32 \
    --batch-window-ms 20
```

`scripts/start.sh` backgrounds the server, polls `/health` until ready, and
exits 0 leaving the server running — that's the contract the submission
rules require.

### Stub (laptop, no GPU)

```bash
./scripts/start.sh --stub
```

Returns canned `"ok"` text. Useful for verifying the OpenAI shape locally.

## Iteration loop — throughput first, correctness as a pre-merge gate

We are scored on **weighted output tok/s** at concurrency 1..64. The
high-concurrency levels carry the most weight (`c=16`+`c=32`+`c=64` =
**16/22 ≈ 73% of the score**, and `c=64` alone is 8/22). The correctness
eval is a hard gate but it does not move the score — we should run it
**sparingly** (before pushing to main, after large changes, on CI),
**not** on every per-change iteration.

### Default per-change loop: quick throughput

```bash
.venv/bin/python -m bench.quick_throughput \
  --base-url http://localhost:8765 \
  --output profiles/throughput_<scenario>_<sha>.json
```

Defaults to **c=16, 32, 64** (the high-weight levels), 16 requests/level
(vs the harness's 64), runtime random prompts via the same code path the
official scoring uses (`eval.throughput.run_throughput.run_benchmark`).
Finishes in ~1 minute on a working engine. Prints:

- per-level tok/s + wall + spot-check results
- the **partial weighted score** for the levels we ran
- a rough extrapolation to all 7 levels (NOT the official number — just a
  hint of where we'd land)
- the engine's CLI region table fetched from `/metrics/regions` (so each
  iteration drops a snapshot you can diff against the previous one)
- the live `/metrics` snapshot (rolling-window tok/s, batch fill, etc.)

For the full sweep that mirrors the official eval (~5x slower):

```bash
.venv/bin/python -m bench.quick_throughput \
  --base-url http://localhost:8765 \
  --concurrency 1 2 4 8 16 32 64 \
  --num-requests 64
```

Or directly:

```bash
.venv/bin/python -m eval.throughput.run_throughput \
  --base-url http://localhost:8765
```

### Pre-merge gate: full GSM8K-CoT (≥ 87.5%, ~3 min on Phase 1)

Run **before pushing to main** and after any change that touches the model
forward path or sampler. Do **not** run per-iteration.

```bash
.venv/bin/python -m eval.correctness.run_correctness \
  --base-url http://localhost:8765 \
  --num-concurrent 32 \
  --output results/correctness_<sha>.json \
  --output-dir results/correctness_raw_<sha>
```

A 20-problem mini check is too noisy at this sample size — just run the
full 200. It's ~3 min.

### API conformance

```bash
.venv/bin/python -m eval.check_server --base-url http://localhost:8765
```

### Final scoring (the official combined number)

```bash
.venv/bin/python -m eval.score \
  --correctness results/correctness_<sha>.json \
  --throughput results/throughput_<sha>.json
```

## Throughput-aware metrics at `/metrics`

The engine's `/metrics` endpoint gives you the throughput-relevant
counters at any point during a run. The per-change loop is:
**run quick_throughput → diff `/metrics` and `/metrics/regions` against the
previous snapshot.**

```bash
curl -s http://localhost:8765/metrics
```

```json
{
  "tok_per_s_recent": 612.4,            # rolling 60s window — "right now"
  "tok_per_s_lifetime": 549.0,
  "prompt_tok_per_s_lifetime": 489.0,
  "completion_tok_per_s_lifetime": 60.1,
  "batch_tok_per_s_p50": 580.2,         # one batched generate's tok/s
  "batch_tok_per_s_p99": 720.3,
  "max_batch": 32,
  "avg_batch_size": 24.4,
  "avg_batch_fill": 0.762,              # how full the batches are
  "running": 0,
  "waiting": 0,
  ...
}
```

`avg_batch_fill < 0.5` is a clear signal that the batcher is leaving
throughput on the floor — the window is too short or `num_concurrent` on
the client side is too low.

`tok_per_s_recent` is the number to watch when iterating. The `/metrics`
endpoint costs nothing to hit so you can `watch -n 0.5 'curl -s …'`.

## Profiling — start with the CLI region table

`engine/runtime/profiling.py` defines `RegionTimer`. Wrap any chunk of code
with `time_region("name")`:

```python
from engine.runtime.profiling import time_region

with time_region("moe.grouped_gemm"):
    out = grouped_gemm(...)
```

Then dump the table at any time:

```bash
curl -s http://localhost:8765/metrics/regions
```

```
region                                  n    total_s    mean_ms     p50_ms     p99_ms
-------------------------------------------------------------------------------------
engine.model.generate                  20    160.234   8011.700   7895.123   9876.456
engine.batch.tokenize                  20      1.234     61.700     58.123    120.456
engine.batch.render                    20      0.234     11.700     10.123     20.456
...
```

This is the primary feedback loop. Per-change diff = before vs after table.

`torch.profiler` is wired in the same module (`enable_torch_profiler(...)`,
`--profile-torch --profile-window=10:20`) but reserved for **drilling inside**
a region the CLI table has already flagged. Don't run with `--profile-torch`
by default — the overhead and the friction of opening the chrome trace make it
the wrong tool for the inner loop.

## Engine internals (Phase 1)

```
                   ┌─────────────────────┐
   request ──────► │ FastAPI handler     │
                   │ (server/app.py)     │
                   └──────────┬──────────┘
                              │ Engine.generate()
                              ▼
                   ┌─────────────────────┐
                   │ asyncio.Queue       │  ◄── Engine._req_queue
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ _batcher_loop       │  background coroutine
                   │  ── gather batch    │
                   │     up to N or      │
                   │     batch_window_ms │
                   │  ── left-pad        │
                   │  ── one model.gen() │
                   │  ── trim @ EOS      │
                   │  ── set futures     │
                   └──────────┬──────────┘
                              │ asyncio.to_thread(...)
                              ▼
                   ┌─────────────────────┐
                   │ HF Qwen3_5MoeFor… │
                   │ .generate(inputs:[B,L]) │
                   └─────────────────────┘
```

- The batcher keys batches on `do_sample` (temperature == 0 vs > 0); same
  batch shares one `(temperature, top_p)`.
- `max_new_tokens = max(req.max_tokens for req in batch)`. Each output is
  trimmed at first EOS / pad before returning, so short answers don't
  pay for the longest one in `usage.completion_tokens`.
- Padding is **left** (HF generate requirement). `tok.pad_token` is set
  to EOS at engine load time if unset.

This is the smallest change that buys real throughput at TP=1 *before*
we tear out HF's KV cache. It is **not** the final design — Phase 2
replaces this with a continuous (in-flight) batcher + paged KV +
DeltaNet per-sequence state cache + TP=8.

## Performance so far (Phase 1, single H200)

| Configuration | Mini GSM8K (20 problems) wall time |
|---|---|
| asyncio.Lock + concurrent=2 (no batching) | 168 s |
| Microbatch=16 + concurrent=8 | 74 s |
| Microbatch=32 + concurrent=32 | **44 s** |

Full 200-problem gate at microbatch=32 + concurrent=32: **3:01** wall.

From the post-gate region table:

| region | n | mean_ms | p50_ms | p99_ms |
|---|---|---|---|---|
| `engine.generate` (whole request) | 220 | 24407 | 18654 | 41147 |
| `engine.model.generate` (one batch) | 9 | 23575 | 18551 | 36992 |
| `engine.batch.tokenize` | 9 | 35 | 39 | 61 |
| `engine.batch.render` | 9 | 22 | 27 | 32 |
| `engine.batch.decode` | 9 | 8 | 10 | 14 |

The batching scaffolding is **0.3% of batch time** — it's pure model
forward time at this stage. Avg batch size 24.4 (concurrent=32 doesn't
fully fill because requests trickle in). Phase 2 attacks the model.generate
slab via TP=8 + paged KV + continuous batching + CUDA graphs.

These are correctness-iteration numbers, not the final throughput numbers
we'll be scored on. The throughput sweep at high concurrency (where
scoring weight is highest) lives in Phase 2 onward.

## What's NOT done yet (and where to start)

In rough priority order:

1. **TP=8 + continuous batching**. The single biggest perf gap. This
   is the bulk of Phase 2. See the plan section in HACKING for the
   sub-tasks (paged KV, DeltaNet state cache, FA2 paged-decode, scheduler).
2. **Real throughput sweep**. We have not yet run
   `eval/throughput/run_throughput.py` end-to-end against our engine.
   Doing so will give us our first "where do we stand vs vLLM baseline"
   number — useful as a clean Phase 1 floor before Phase 2 work begins.
3. **Multi-GPU eval parallelism for fast iteration**. We have 8 GPUs
   sitting idle. Running 4 engine instances on 4 GPUs and sharding the
   harness across them would drop the gate eval from ~7 min to ~2 min.
   Easy follow-up if iteration speed pinches.
4. **Helion kernels** (Phase 3). We'll need `helion-docs` and
   `helion-perf` knowledge for this — deferred until the Phase 2 scaffolding
   is in place and the CLI region table tells us which kernel to write first.

## Coordinating with each other

- **Server port**: I run on `8765` to leave `8000` for whoever wants the
  default. You'll see `mghanmi` running on `8000` too — heads up.
- **Weights cache**: `/mnt/data/$USER/models/Qwen3.5-35B-A3B`. Don't
  re-download into a shared dir without coordinating.
- **Branches**: feel free to branch off `main`. The codebase is small
  enough that big-bang merges are fine.
- **Profiling artifacts**: `profiles/` is gitignored except for its
  README. Drop your CLI region tables there using the naming convention
  in `profiles/README.md` — `regions_phaseX_<scenario>_<sha>_<ts>.txt`.
