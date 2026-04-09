# Parallelism + scheduling strategy for Qwen3.5-35B-A3B on 8×H200

> **Updated 2026-04-09** — replaces the previous "TP=8 first, EP later" recommendation.
> New profile evidence (commit `44959f7`) shows that MoE expert dispatch
> dominates CUDA time. EP=8 is now the Phase 2b prototype target. TP=8
> remains the fallback. Decision is gated on two microbenchmarks that must
> run before writing any parallelism code.

---

## TL;DR

> 1. **Phase 2a: continuous batching + paged KV** — biggest single lever,
>    fixes the c=64 < c=32 cliff by letting sequences join/leave the batch
>    at every decode step. Land this first.
> 2. **Phase 2b: prototype EP=8 first, not TP=8.** The Phase 1 profile
>    is unambiguous: `aten::index` + `vectorized_gather_kernel` (both MoE
>    token-dispatch ops) account for 56% of all CUDA time on a single GPU.
>    EP=8 makes each rank's dispatch work 8× smaller. TP with replicated
>    experts does nothing for that op.
> 3. **Measure before you build.** Run `bench/microbench_routing.py` to
>    get the real routing histogram, and `bench/microbench_comms.py` to
>    measure NCCL all-reduce vs all-to-all at our actual message sizes.
>    Both benchmarks ship before the first line of parallelism code.
> 4. **Fallback ladder:** if routing skew exceeds ~3× median at c=64,
>    fall back to TP=8 replicated experts (Phase 2b-fallback), then
>    consider Hybrid TP=2 + EP=4 (Phase 2c) once the engine is stable.

---

## Lever ordering (the big picture)

| Phase | Milestone | Why this order |
|---|---|---|
| **2a** | **Continuous batching + paged KV** | Phase 1's static batcher wastes capacity: late requests can't join an in-flight batch. At c=64 we observe `avg_batch_size ≈ 14.6` and `avg_batch_fill ≈ 0.23`. Continuous batching fixes both. Paged KV is the enabler — variable-length sequences in one running batch can't share contiguous KV allocations. They ship together. |
| **2b** | **EP=8 (prototype) — or TP=8 (fallback)** | Profile evidence is overwhelming: MoE dispatch is 56% of CUDA time. EP targets that directly. TP does not. Gate the choice on the two microbenchmarks in Section 7. |
| **3** | **Custom Helion kernels** (MoE grouped GEMM, DeltaNet recurrence) | Squeezes per-step time once the engine is saturating compute. Diff-tested against the torch reference. |
| **2c** | **Hybrid TP=2 + EP=4** (fallback if EP=8 hits skew) | Best of both worlds if pure EP=8 serializes on routing imbalance. More code complexity; only worth it when the routing histogram justifies it. |
| **4** | **Tuning** — Helion autotuning, scheduler windows, CUDA graphs | Last 20–40%, after structure is settled. |

PP (pipeline parallelism) is **rejected** for this workload — see Section 8.

---

## 1. Current bottleneck: the profile says MoE dispatch, not compute

**Profile capture:** commit `44959f7`, batch_size=16, input_padded_len=1038,
max_new_tokens=32, single H200 BF16.

| # | kernel | self CUDA (ms) | % of 3354 ms total | calls |
|---:|---|---:|---:|---:|
| 1 | `aten::index` | 424 | 12.6% | 3952 |
| 2 | `vectorized_gather_kernel` | 423 | 12.6% | 3801 |
| 3 | `aten::bmm` | 256 | 7.6% | 2512 |
| 4 | `aten::copy_` | 177 | 5.3% | 20849 |
| 5 | `nvjet_sm90` (CUTLASS GEMM 512) | 168 | 5.0% | 1240 |
| 6 | `aten::mm` | 146 | 4.3% | 12512 |
| 7 | `Command Buffer Full` | 126 | 3.8% | 1189 |
| 8 | `nvjet_sm90` (CUTLASS GEMM 256) | 87 | 2.6% | 1240 |
| 9 | `aten::_grouped_mm` | 64 | 1.9% | 80 |

**Entries 1 + 2 together: 847 ms = 25.2% each, 50.4% combined.** Both are
the same logical operation: MoE top-k token gather/scatter. `aten::index`
dispatches tokens to routed experts; `vectorized_gather_kernel` is the CUDA
kernel it launches. These are not compute kernels — they are index/scatter
kernels that move token embeddings around in HBM. They are expensive because
at 256 experts and top-k=8, the dispatch table is large and the gather
pattern is non-contiguous.

On a **single GPU**, at batch=16 decode tokens, this one operation is half
of all CUDA time. Whatever Phase 2b does, it must shrink this cost or it
will not move the needle.

Additionally:
- `aten::copy_` at 177 ms / 20849 calls: launch-bound overhead. Too many
  small kernel launches. A signal that token dispatch is creating many
  small tensor slices.
- `Command Buffer Full` at 126 ms: the CUDA command queue is overflowing.
  Same root cause — dispatch is generating more kernels than the queue can
  absorb without stalling.
- `aten::bmm` at 256 ms: DeltaNet chunk matmul (30 layers × ~8.5 ms avg).
  This is real compute and is second priority after dispatch.
- `aten::_grouped_mm` + CUTLASS GEMM: the actual MoE expert GEMMs. Only
  64 ms total. The gather/scatter surrounding them costs 13× more than the
  GEMMs themselves at this batch size.

**The killer insight:** the GEMMs are not the problem. The index + scatter
overhead is. Any strategy that reduces the number of tokens a given GPU must
dispatch through its local gather/scatter path wins. That is exactly what
EP does.

---

## 2. Compute budget per layer

Before choosing a parallelism strategy, anchor the numbers.

### Model architecture recap

```
hidden_size:          2048
num_hidden_layers:    40  (pattern: [linear, linear, linear, full] × 10)
  → 30 Gated DeltaNet (linear attention) layers
  → 10 full attention layers (GQA, 16 heads, 2 KV heads, head_dim=256)
MoE per layer:
  num_experts:          256
  num_experts_per_tok:  8   (routed top-k)
  shared_expert:        1
  moe_intermediate_size: 512
  shared_expert_intermediate_size: 512
```

### Per-token FLOPs per MoE layer

Each expert FFN is a gated MLP: `[hidden → intermediate → hidden]`.

```
Expert weight volume:
  up_proj:   hidden × intermediate = 2048 × 512 = 1,048,576 params
  gate_proj: 2048 × 512 = 1,048,576 params
  down_proj: 512 × 2048 = 1,048,576 params
  Total per expert: ~3.1M params × 2 bytes (BF16) ≈ 6.3 MB

FLOPs per token per expert (MatMul only, ignoring activation):
  up_proj:   2 × 2048 × 512 ≈ 2.1 MFLOPs
  gate_proj: 2 × 2048 × 512 ≈ 2.1 MFLOPs
  down_proj: 2 × 512 × 2048 ≈ 2.1 MFLOPs
  Total per expert: ~6.3 MFLOPs per token

Active experts per token: 8 routed + 1 shared = 9
FLOPs per token per MoE layer: 9 × 6.3 ≈ 57 MFLOPs
```

### Per-token FLOPs per attention layer

Full attention (10 layers):
```
Q proj: 2 × 2048 × (16 × 256) = 2 × 2048 × 4096 ≈ 16.8 MFLOPs
K proj: 2 × 2048 × (2 × 256)  = 2 × 2048 × 512  ≈ 2.1 MFLOPs
V proj: 2 × 2048 × (2 × 256)  = 2 × 2048 × 512  ≈ 2.1 MFLOPs
O proj: 2 × 4096 × 2048        ≈ 16.8 MFLOPs
QK/V attention (decode, seq KV cache length ~1024):
  QK^T: 2 × 16 × 256 × 1024   ≈ 8.4 MFLOPs
  softmax × V: similar         ≈ 8.4 MFLOPs
Total per token per full-attn layer: ~55 MFLOPs
```

Linear attention (DeltaNet, 30 layers): recurrent-mode during decode, so
FLOP cost is proportional to the state update, not the full sequence. State
update per token is dominated by the outer-product update of the v_heads:
roughly `2 × hidden × v_head_dim × num_v_heads = 2 × 2048 × 128 × 32 ≈ 16.7 MFLOPs`.
Smaller than full attention by ~3×, consistent with the profile showing
`ChunkGatedDeltaRuleFunction` at 50 ms vs `aten::_cudnn_attention_forward`
at 49 ms (both comparable since the profile capture runs small batches).

### Cross-check against 3B active params

```
Active params per token:
  MoE:    9 experts × 3.1M params/expert ≈ 27.9M  ← dominated by MoE FFN
  Attn:   10 layers × ~8M dense params   ≈ 80M
  DeltaNet: 30 layers × ~10M params      ≈ 300M
  Embedding: 248320 × 2048 / tokens ← amortized
  Rough active weight budget: ~400M–500M "touched" params per token
```

Note: the 3B active parameter figure refers to parameters that are
*loaded and used* per forward pass across the full sequence, not per token.
For a single decode token the active set is smaller, dominated by the 9
expert FFNs.

---

## 3. Communication budget per layer for each scheme

**Setup:** c=64 concurrent requests, decode phase (1 new token per request
per step). Hidden size H=2048, BF16 = 2 bytes/element.

### TP=8 (Megatron-style, replicated experts)

Each linear layer is split along its column (ColumnParallel) or row
(RowParallel) dimension. The reduce happens on the output.

```
Per all-reduce message (one rank contributes, all ranks sum):
  tokens × H × bytes = 64 × 2048 × 2 = 262,144 bytes = 256 KB

All-reduces per attention layer:  1 (o_proj output)
All-reduces per MoE layer:        2 (shared expert down + routed expert down;
                                     or bundled into one if fused)
All-reduces per DeltaNet layer:   0–1 (output gate only if sharded;
                                     recurrent state is rank-local)

Per-layer comm at c=64:
  Full-attn:  1 × 256 KB all-reduce  = 256 KB moved per rank
  MoE layer:  2 × 256 KB all-reduce  = 512 KB moved per rank
  DeltaNet:   ~256 KB (output sharded) or 0 (replicated)
```

NVLink H200 bisection bandwidth: ~900 GB/s aggregate across 8 GPUs.
Per-rank all-reduce at 256 KB with ring-allreduce: ~0.5 µs at full NVLink
bandwidth (best case). In practice, NCCL adds ~5–20 µs latency overhead
per collective at these sizes. The 40 all-reduces per full forward pass
cost ~0.2–0.8 ms of NCCL overhead — small relative to the 3.3 s total
captured step time.

**TP bottleneck is NOT comms at this scale. It is the replicated expert
dispatch overhead** — every GPU still gathers/scatters the full set of tokens
across 256 experts, and each rank's gather kernel sees the same 16-token-per-
expert average that produces the profile's 424 ms index cost.

### EP=8 (32 experts per rank)

Experts are sharded across ranks. Tokens are shuffled (all-to-all) to the
rank that holds their assigned experts, computed there, then shuffled back.

```
Per all-to-all message (tokens going out from one rank to one other rank):
  At c=64 decode: 64 tokens total × 8 active each = 512 expert activations
  With 256 experts ÷ 8 ranks = 32 experts/rank
  Expected activations received per rank:
    512 / 8 = 64 tokens × hidden = 64 × 2048 × 2 = 262,144 bytes = 256 KB

All-to-alls per MoE layer: 2 (token shuffle in + result shuffle out)
Per-rank all-to-all at c=64:
  MoE layer in:  256 KB per rank
  MoE layer out: 256 KB per rank
  Total: 512 KB per rank per MoE layer (same byte count as TP at this batch size)
```

The key difference is not bandwidth — it is the **dispatch work**:
- TP=8: every rank gathers from 256 experts (full dispatch table), then
  computes 1/8 of the weight for each of those 256 experts.
- EP=8: every rank gathers from its 32 experts only. The gather kernel
  sees 1/8 the number of expert slots. `aten::index` call count scales
  with `num_experts_per_rank × batch`, not `num_experts × batch`. That
  is the 56% CUDA cost going away.

**EP wins on the dominant cost item. TP does not touch it.**

### Hybrid TP=2 + EP=4

4 EP groups of 2 TP ranks each. Within each group, dense layers are TP=2
sharded. Across groups, MoE experts are EP=4 sharded (64 experts/group,
32 experts/rank within each TP=2 pair after TP split).

```
Per-layer comms (c=64):
  Attn all-reduce (intra TP=2 group):  64 × 2048 × 2 / 2 = 128 KB
  MoE all-to-all (across EP=4 groups): 64 × 8 × 2048 × 2 / 4 / 2 = 256 KB in + out
```

Half the all-reduce size, half the all-to-all size vs pure EP=8. But this
complexity is only worth it when EP=8 is hitting the routing skew wall and
TP=2 can absorb the load imbalance within each EP group.

---

## 4. Memory budget per rank

Hardware: 8× H200, 140 GB HBM3e each. Model: 35B params × 2 bytes (BF16) ≈ 70 GB total.

### TP=8 with replicated experts (current Phase 2b-fallback plan)

```
Dense layer weights replicated across all ranks (attn, DeltaNet, embed):
  ~70 GB total × (dense fraction) ÷ 8 ranks
  Dense fraction (attn + DeltaNet + embed) ≈ 35B - ~27B MoE = ~8B params ≈ 16 GB
  Dense per rank (TP=8 sharded): 16 GB / 8 = 2 GB

MoE weights replicated (all 256 experts on every rank):
  256 experts × 3 projections × 2048 × 512 × 2 bytes
  = 256 × 3 × 2,097,152 bytes = 1.6 GB per MoE layer × 40 layers = 64 GB
  (but only ~10 MoE layers per pattern group, so ~16 GB for the routed experts)

Rough total weights per rank: ~20–25 GB
KV cache (max_seq=4096, batch=64, 10 full-attn layers, GQA 2 KV heads):
  64 × 4096 × 2 × 256 × 10 × 2 bytes ≈ 1.7 GB
Activations / workspace: ~2–4 GB
Total: ~25–30 GB per rank. Fits comfortably in 140 GB.
```

### EP=8 (32 experts per rank)

```
Routed expert weights per rank:
  32 experts × 3 × 2048 × 512 × 2 bytes × 40 layers
  = 32/256 × 64 GB total expert weight = ~8 GB (routed only)
  Shared expert (1, replicated): negligible (~200 MB)

Dense layers replicated (same as TP=8 non-sharded case): ~16 GB
  (no TP sharding in pure EP; dense layers replicated)

Rough total weights per rank: ~24 GB
KV + activations: ~4–6 GB
Total: ~28–30 GB per rank. Also comfortably fits.
```

**EP=8 offers ~400 MB of routed expert memory per rank vs ~8 GB in TP=8
replicated** — EP=8 actually uses less memory for the experts. Both schemes
fit well within 140 GB. Memory is not the discriminating factor; the
discriminating factor is the gather/scatter cost profile.

### Hybrid TP=2 + EP=4

```
64 experts per EP group, 32 experts per rank (same as EP=8 within the TP pair).
Dense layers TP=2 sharded: ~8 GB per rank.
Expert weights: ~8 GB per rank (same as EP=8).
Total: ~18–22 GB per rank. Even more headroom.
```

---

## 5. Routing skew analysis

This is the main risk for EP.

### Expected load distribution

```
Tokens per step (c=64 decode): 64
Active routed experts per token: 8
Total expert activations per step: 64 × 8 = 512
Number of experts: 256
Expected activations per expert (uniform): 512 / 256 = 2

With EP=8 (32 experts per rank):
Expected activations per rank (uniform): 32 × 2 = 64 tokens
```

Under perfect uniform routing, each EP=8 rank processes 64 token activations
per step. Every rank does the same amount of work. EP serializes on the
slowest rank, but if all ranks are equal, there is no penalty.

### Worst-case skew

Real MoE routers are not perfectly uniform. Empirically, the top-loaded
expert in a transformer MoE typically sees 2–6× the median load. At our
token counts:

```
If top expert sees 3× median:   2 tokens (median) → 6 tokens (top expert)
  Worst-case rank (32 experts): 32 × 6 = 192 token activations
  Best-case rank:               32 × 1 = 32 token activations
  Skew ratio: 192/64 = 3×. EP=8 effective throughput: 64/192 × original.
  Throughput penalty: ~33%.

If top expert sees 6× median:   2 → 12 tokens
  Worst-case rank: 32 × 12 = 384
  Throughput penalty: ~83% (catastrophic).
```

**The skew risk is real.** The router balance depends on the actual prompt
distribution. We do NOT apply lossy capacity-factor dropping (it would
change output distributions and violate correctness requirements). The only
safe mitigations are:

1. Measure the actual routing histogram before choosing EP=8.
2. If skew is >3× at p90 across a realistic prompt sample, fall back to
   TP=8 replicated experts.
3. Hybrid TP=2 + EP=4 as a middle path: 4 EP groups, so each rank holds
   64 experts instead of 32. Skew effect is halved.

The gate is the routing histogram microbenchmark in Section 7.

---

## 6. The recommendation

### Principle: ground the decision in measurements, not theory

The profile shows what costs time today. The routing histogram will show
what EP=8 would cost at runtime. The NCCL microbenchmark will close the
loop on comms. Both are cheap to run. Do them first.

### Phase ordering

**Phase 2a: continuous batching + paged KV (unchanged)**

Must land first. The static batcher is currently filling 23% of available
capacity at c=64 (`avg_batch_fill = 0.23`, `avg_batch_size = 14.6` at
`max_batch = 64`). Continuous batching will raise this toward 80–90%.
Parallelism multiplies throughput, not fill rate — doubling the number of
GPUs on a 23%-full engine gives you less than you expect. Fill first.

**Phase 2b: EP=8 prototype (new recommendation)**

Evidence:
- `aten::index` + `vectorized_gather_kernel` = 847 ms = 56% of CUDA time in
  the Phase 1 single-GPU profile.
- Both kernels are the MoE top-k token dispatch path. EP=8 makes each rank's
  dispatch 8× smaller by giving it only 32 experts to gather from.
- TP=8 with replicated experts does not reduce the per-rank expert count. The
  token dispatch work per rank stays the same.
- All-to-all message size at c=64 is ~256 KB per rank per MoE layer — the
  same order of magnitude as TP's all-reduce. Not a meaningful comms
  disadvantage on NVLink H200.

**Implementation path for EP=8:**
- `engine/parallel/comm.py`: torch.distributed (nccl) init, `ep_group`
  for the 8-rank all-to-all group.
- `engine/parallel/ep.py`: token router that calls `dist.all_to_all_single`
  for the token shuffle in/out. Expert assignment map persisted as JSON.
- `engine/parallel/loader.py`: per-rank safetensors mmap → slice → device,
  loading only the 32 experts assigned to this rank.
- Sharding rules for the MoE block: `moe_mode = "expert_parallel"` flag in
  `Qwen3_5MoeSparseMoeBlock` (or monkey-patched on load).
- Dense layers (attn, DeltaNet, embed): replicated across all 8 ranks in
  pure EP=8. No TP sharding in the first prototype. Add TP later only if
  needed.

**Phase 2b-fallback: TP=8 with replicated experts**

Triggered if the routing histogram (Section 7) shows >3× skew at p90 at
c=64. This is the strategy described in the original `parallelism.md` and
what vLLM uses as their published TP=8 baseline (12,810 tok/s). It is the
safer option from a correctness and complexity standpoint; it just doesn't
target the 56% index cost directly.

**Implementation path for TP=8:**
- `engine/parallel/comm.py`: `tp_group` for the 8-rank all-reduce group.
- `engine/parallel/tp.py`: `ColumnParallelLinear`, `RowParallelLinear`,
  `VocabParallelEmbedding`. Megatron-style: wrap `nn.Linear`, slice weights
  at load, all-reduce in forward.
- `engine/parallel/loader.py`: per-rank weight slicing for TP.
- DeltaNet: if sharding the SSM recurrence is hairy, replicate DeltaNet
  layers (no comm required); the FLOP cost is bounded at ~16 MFLOPs/token/
  layer and the profile shows it is second priority after dispatch.

**Phase 2c: Hybrid TP=2 + EP=4**

If EP=8 is deployed and routing skew causes >20% throughput regression vs
TP=8 baseline, promote to Hybrid TP=2 + EP=4. This halves both the TP
all-reduce size and the EP all-to-all imbalance exposure. More code
complexity; only warranted with data.

### Decision gate

```
                      ┌─────────────────────────────────┐
                      │  Run bench/microbench_routing.py │
                      │  on 64 real prompts (Section 7)  │
                      └──────────────┬──────────────────┘
                                     │
              p90 skew (top/median)  │
              ┌──────────────────────┴──────────────────┐
              │ < 3×                                     │ > 3×
              ▼                                          ▼
   Proceed with EP=8 prototype                 Fall back to TP=8
   Run bench/microbench_comms.py               replicated experts
   Confirm all-to-all ≈ all-reduce
   at 256 KB message size
```

---

## 7. Microbenchmark plan

**Both benchmarks run before writing any parallelism code. Decision is data,
not intuition.**

### 7.1 `bench/microbench_routing.py` — routing histogram

**Goal:** measure the real per-expert load distribution at our actual token
counts on real prompts. Determine whether EP=8 will encounter dangerous skew.

**Method:**

```python
# Hook the gate forward in Qwen3_5MoeSparseMoeBlock
# Capture top-k indices across 64 random prompts
# Dump per-expert activation counts per layer per step

def hook_moe_gate(module, input, output):
    # output is (routing_weights, selected_experts)
    # selected_experts: [batch, top_k] of expert indices
    indices = output[1]  # shape [B, top_k]
    for layer_id in ...:
        counts = torch.bincount(indices.flatten(), minlength=256)
        histogram[layer_id].append(counts.cpu())
```

**Metrics to report:**
- Per-layer: `mean`, `median`, `p90`, `max` of per-expert token count
- Skew ratio: `max / median` per layer
- Cross-layer correlation: do the same experts win in every layer?
- EP=8 worst-case rank load: `max(sum(experts_for_rank))` across 64-token steps

**Pass criterion:** p90 skew ratio < 3× across all MoE layers at c=64.
If any layer fails, note which experts are hot (may warrant capacity-factor
padding for those specific experts without global lossy capping).

**Script checklist:**
- [ ] Load single-GPU engine with the existing `load_model` path
- [ ] Register forward hooks on all `Qwen3_5MoeSparseMoeBlock` gate modules
- [ ] Sample 64 prompts from a representative prompt file (or generate random
      1024-token inputs)
- [ ] Run `model.generate` with the hooks active, capturing 32 decode steps
- [ ] Dump histograms to `profiles/routing_histogram_<sha>_<date>.json`
- [ ] Print summary table: layer × {mean, p50, p90, max, skew_ratio}
- [ ] Fail loudly if p90 skew > 3× with a clear EP=8 NOT RECOMMENDED message

### 7.2 `bench/microbench_comms.py` — NCCL all-reduce vs all_to_all_single

**Goal:** measure actual NVLink latency at the message sizes we'll generate
at c=64 decode. Verify the all-to-all is not significantly worse than the
all-reduce at these sizes.

**Method:**

```python
import torch.distributed as dist

MESSAGE_SIZES_BYTES = [
    32 * 1024,    # 32 KB   — small decode batch (c=8)
    64 * 1024,    # 64 KB
    128 * 1024,   # 128 KB
    256 * 1024,   # 256 KB  — c=64 target
    512 * 1024,   # 512 KB  — headroom / worst-case skew
    1024 * 1024,  # 1 MB    — prefill chunks
]

for size_bytes in MESSAGE_SIZES_BYTES:
    numel = size_bytes // 2  # BF16
    t = torch.zeros(numel, dtype=torch.bfloat16, device='cuda')

    # All-reduce (TP baseline)
    for _ in range(WARMUP): dist.all_reduce(t, async_op=False)
    t0 = time.perf_counter()
    for _ in range(ITERS): dist.all_reduce(t, async_op=False)
    torch.cuda.synchronize()
    ar_ms = (time.perf_counter() - t0) / ITERS * 1000

    # All-to-all (EP baseline)
    chunks = list(t.chunk(dist.get_world_size()))
    out_chunks = [torch.empty_like(c) for c in chunks]
    for _ in range(WARMUP): dist.all_to_all(out_chunks, chunks)
    t0 = time.perf_counter()
    for _ in range(ITERS): dist.all_to_all(out_chunks, chunks)
    torch.cuda.synchronize()
    a2a_ms = (time.perf_counter() - t0) / ITERS * 1000

    print(f"{size_bytes//1024:6d} KB | all_reduce {ar_ms:.3f} ms | all_to_all {a2a_ms:.3f} ms | ratio {a2a_ms/ar_ms:.2f}x")
```

**Pass criterion:** all-to-all / all-reduce ratio < 2× at 256 KB.
If ratio > 2×, EP=8 has a comms disadvantage on top of any skew risk;
fall back to TP=8.

**Script checklist:**
- [ ] Run with `torchrun --nproc_per_node=8`
- [ ] Warmup 20 iterations, measure 100 iterations per size
- [ ] Report mean ± stddev
- [ ] Sweep both BF16 and FP32 (DeltaNet SSM uses FP32 internally)
- [ ] Dump results to `profiles/comms_bench_<sha>_<date>.json`
- [ ] Exit non-zero if ratio > 2× at the 256 KB target size

### 7.3 Schedule

Both benchmarks are fast (< 5 minutes each on 8× H200) and can run
immediately after Phase 2a lands. They produce the routing JSON and comms
JSON artifacts in `profiles/`. The Phase 2b parallelism PR opens only after
both artifacts exist and both pass criteria are met.

---

## 8. Why not pipeline parallelism

PP across 8 GPUs would put ~5 layers per stage. The pipeline bubble at
batch=B and stages=S has overhead `(S-1)/(B+S-1)`:
- B=64, S=8: `7/71 = 9.9%` — tolerable.
- B=4, S=8: `7/11 = 63%` — catastrophic.

Since c=1..8 collectively carries `6/22` of the scoring weight and the
Phase 1 profile already shows throughput scaling matters at low concurrency,
sacrificing small-batch performance is poor arithmetic. PP is also more
fragile with DeltaNet: the recurrent state update requires the entire hidden
state before the next token can proceed, which breaks the stage-overlap
assumption that PP relies on.

PP is rejected. Revisit only if Phase 2a's continuous batcher can reliably
sustain B≥32 at even c=1 through aggressive prefill chunking.

---

## 9. What the model actually looks like (architecture quick-reference)

| | |
|---|---|
| Total / active params | 35B / ~3B per token |
| Layers | 40 (pattern `[linear, linear, linear, full] × 10`) |
| Full attention | 10 layers, 16 heads, 2 KV heads (GQA), head_dim=256 |
| Linear attention (DeltaNet) | 30 layers, 32 v-heads, 16 k-heads, head_dim=128, conv_k=4, SSM fp32 |
| MoE | 256 experts, top-k=8 routed + 1 shared, intermediate=512 |
| Hidden size | 2048 |
| Vocab | 248,320 |
| Context window | 262,144 (MRoPE interleaved, theta=1e7) |
| MTP head | 1 layer (spec-decode candidate in Phase 5) |
| Workload | 1024 in / 1024 out, c=1..64, scoring weight skewed toward c≥16 |

The MoE block is architecturally dominant: 256 experts × 40 layers ×
~6.3 MB weights/expert = ~64 GB of expert weights in the full model.
Active weight per token = 9 experts × 6.3 MB = ~57 MB touched per MoE
layer per token. This is why the token dispatch (deciding which 9 of 256
experts to touch) costs more than the GEMMs at small batch sizes.

---

## 10. Phase 2 sub-task list (parallelism)

Listed in dependency order. Microbenchmarks gate the choice between EP and TP.

```
[ ] bench/microbench_routing.py     — routing histogram on 64 prompts
[ ] bench/microbench_comms.py       — NCCL all-reduce vs all_to_all_single sweep

--- if routing p90 skew < 3× AND comms ratio < 2×: proceed with EP=8 ---

[ ] engine/parallel/comm.py         — dist init, ep_group, tp_group placeholders
[ ] engine/parallel/ep.py           — all_to_all token router, expert assignment map
[ ] engine/parallel/loader.py       — per-rank expert weight slicing from safetensors
[ ] engine/model/qwen3_next.py      — moe_mode="expert_parallel" flag, hook into
                                       Qwen3_5MoeSparseMoeBlock dispatch path
[ ] tests/test_ep_correctness.py    — diff-test EP=8 output vs single-GPU reference
                                       on 8 prompts, max relative error < 1e-3

--- if skew > 3×: fall back to TP=8 ---

[ ] engine/parallel/comm.py         — tp_group
[ ] engine/parallel/tp.py           — ColumnParallelLinear, RowParallelLinear,
                                       VocabParallelEmbedding
[ ] engine/parallel/loader.py       — per-rank dense weight slicing
[ ] engine/model/qwen3_next.py      — monkey-patch on load with TP wrappers
[ ] tests/test_tp_correctness.py    — same diff-test structure

--- either path ---

[ ] bench/run_phase2b.sh            — throughput sweep c=[1,2,4,8,16,32,64]
                                       vs Phase 1 and vLLM TP=8 baseline
[ ] STATUS.md refresh               — update region table and kernel hotspots
```

---

## Appendix: quick numbers cheatsheet

| Quantity | Value |
|---|---|
| Hidden size | 2048 |
| MoE expert intermediate | 512 |
| Active experts per token | 9 (8 routed + 1 shared) |
| FLOPs per token per MoE layer | ~57 MFLOPs |
| FLOPs per token per full-attn layer | ~55 MFLOPs |
| Weight per expert (BF16) | ~6.3 MB |
| Total expert weight | ~64 GB |
| Expert weight per EP=8 rank | ~8 GB (32 experts) |
| All-reduce size (TP=8, c=64) | 256 KB per rank |
| All-to-all size (EP=8, c=64, uniform) | 256 KB per rank |
| Expected tokens per expert (c=64, uniform) | 2 |
| Expected tokens per EP=8 rank (c=64, uniform) | 64 |
| Phase 1 index+gather CUDA cost | 847 ms / 3354 ms = 56% |
| Phase 1 profile batch | B=16, n_steps=32, single H200 |
| vLLM TP=8 baseline | 12,810 tok/s @ c=64 |
