# STATUS

**branch:** `main`
**commit:** `8bb84f7` (`scheduler: revert length bucketing`)
**date:** 2026-04-09
**hardware:** single H200 (GPU 6) — DP across the full 8x H200 not yet wired

## Current best result

| concurrency | tok/s | reqs ok | wall_s | vs vLLM baseline |
|---:|---:|---:|---:|---:|
| 64 | **951.4** | 64/64 | 88.5 | 7.4% (vLLM: 12810) |

_config: c=64, num_requests=64, isl=1024, osl=1024_
_artifact: `profiles/throughput_iter5_n64_152940.json`_

**Correctness sanity (greedy, temperature=0):**
- API conformance: `eval/check_server.py` — PASS
- 247 + 153 → **400** ✓
- 60mph × 2.5h → **150 miles** ✓
- 12 × 8 → **96** ✓
- (Full GSM8K-CoT gate not yet rerun on this commit.)

## How 951 tok/s was achieved

Starting point (Phase 1, commit `44959f7`): 525 tok/s @ c=64. Static
microbatcher around `model.generate()`, no continuous batching.

Stack of changes that landed (in order):

1. **Replaced `model.generate()` with our own engine.** Custom `SlotPoolCache`
   that duck-types HF's `Cache` interface — pre-allocated `[num_slots,
   max_seq_len, num_kv_heads, head_dim]` per full-attention layer + per-slot
   conv/recurrent state for the 30 DeltaNet layers. The engine calls
   `model.model.forward(...)` directly and applies `lm_head` + sampler in
   the runner. (`engine/runtime/{kv_cache,model_runner,scheduler,engine}.py`)

2. **Continuous batching scheduler.** `scheduler.step()` admits prefill OR
   runs decode each tick; running sequences live in a slot dict, finished
   slots are recycled. (`engine/runtime/scheduler.py`)

3. **Vectorised `kv_cache.update()`.** Replaced the per-row Python for-loop
   with `torch.cat(write_positions)` + `repeat_interleave` for slot indices
   + a single advanced-index scatter for the writes; reads use `index_select`
   + a `[B, max_s]` mask multiply to zero out per-row tails. (commit `dfa7b3a`)

4. **Per-row temperature/top-p sampler.** Greedy fast-path when all temps
   are 0; otherwise per-row sort + cumsum + nucleus mask + multinomial with
   a dedicated sampler `torch.Generator` (separate from the default CUDA
   generator so it's unaffected by graph-capture state). (commit `dfa7b3a`)

5. **Real batched prefill compute.** Instead of N sequential single-prompt
   prefills, one padded forward pass over B prompts. Right-pad to `max(L)`
   with `eos_token_id`, build `attention_mask` and `position_ids`, write
   only the **real** positions per row to the cache (the cache update was
   extended to slice `key_states[b, :, :real_lens[b], :]` so pad-position
   K/V never lands in the slot pool). First tokens are sampled from the
   last real position per row via `torch.gather`. (commit `47c25de`)

6. **`max_batch=64`, `max_model_len=4096`.** Workload is 1024+1024 tokens;
   4096 max_model_len leaves headroom for the chat template prefix +
   safety margin (`SlotPoolCache` ~9 GB at these dims, easily fits 140 GB).

The win at c=64 came almost entirely from changes 1, 2, 3, 5. The real
batched prefill (5) compresses 8 sequential prefills (~800 ms eager) into
one padded forward (~233 ms) — a 3.4× compression on the prefill compute
itself, which removes most of the c=64 ramp-up stall.

## What's wired but disabled (broken)

- **CUDA graphs** (`engine/runtime/cuda_graphs.py`) — captures crash with
  `cudaErrorStreamCaptureUnsupported` because `fla.chunk_gated_delta_rule`
  uses a non-capturable CUDA API. Worth ~30% of decode latency once fixed
  (Phase 1 profile: `Command Buffer Full = 126 ms`). Currently `--no-cuda-graphs`.

- **`torch.compile`** (`engine/runtime/compile_helpers.py`) — wraps
  `inner_model` with `mode="reduce-overhead"`. First request hits an
  `AssertionError` deep in `torch._dynamo` TLS state. Needs more work to
  identify the offending node. Currently `--no-torch-compile`.

- **Helion MoE dispatch + grouped MLP** (`engine/kernels/{moe_dispatch,
  moe_grouped_mlp,patch}.py`) — auto-applied at model load via runtime
  monkeypatch. Crashes the engine on first request. Disabled with
  `PARIS_DISABLE_HELION_MOE=1`.

- **Helion Gated DeltaNet** (`engine/kernels/delta_rule.py`) — same
  pattern. Disabled with `PARIS_DISABLE_HELION_DELTA=1`.

- **Length bucketing in scheduler** — even small (±5 token) prompt-length
  variance split sequences into different buckets and starved the decode
  batch. Reverted in commit `8bb84f7` to plain FIFO admit.

- **Prefix caching for chat-template prefix** — works mechanically but the
  shared prefix on the throughput workload is only 3 tokens (≈0.3% of
  prompt), so the gain is in the noise. Disabled in `engine.py` until the
  3-token shift / batched-prefill interaction is debugged.

- **Chunked prefill / `mixed_step`** — the runner method exists at
  `engine/runtime/model_runner.py:441` (one ragged-batch forward with
  decode + prefill rows interleaved) but `scheduler.step` doesn't call it
  yet. This is the highest-leverage scheduler change still on the table.

## Iteration history (tok/s @ c=64)

| commit | label | tok/s | notes |
|---:|---|---:|---|
| `44959f7` | Phase 1 floor | 525 | static `model.generate()` microbatcher |
| `9dd2621` | Phase 2a v0 | 758 | custom engine, continuous batching |
| `dfa7b3a` | iter1 | 758 | vectorised KV update + sampler (no win at c=64) |
| `dfadf45` | iter1 v0 batched prefill | 679 | Python-loop "batched" prefill — net loss (decode starvation) |
| `47c25de` | iter2 | 816 | real padded batched prefill compute — first real win |
| `8bb84f7` | **iter5 (current best)** | **951** | + max_batch 64, max_model_len 4096, length-bucketing reverted |

## Repository pointers

- Engine entrypoint: `engine/runtime/engine.py` (`Engine.build`)
- Runner: `engine/runtime/model_runner.py`
- Scheduler: `engine/runtime/scheduler.py`
- KV cache: `engine/runtime/kv_cache.py`
- Throughput bench: `bench/quick_throughput.py` (fast iteration)
- Official eval: `eval/throughput/run_throughput.py`
- Iteration loop: `HACKING.md`
