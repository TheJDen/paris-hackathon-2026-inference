# Parallelism + scheduling strategy for Qwen3.5-35B-A3B on 8×H200

Two separable axes:

1. **Scheduling** — how we form batches over time (static microbatching
   vs continuous in-flight batching). Currently static; this is the
   biggest lever we have not pulled.
2. **Parallelism** — how we split the model across GPUs (TP, EP, PP, or
   hybrids). Currently TP=1; the second biggest lever.

We do them in **that order**, and bundle each with whatever it requires
to be useful.

## Lever ordering (the big picture)

| Phase | Milestone | Why this order |
|---|---|---|
| **2a** | **Continuous batching + paged KV** (one milestone) | Phase 1's static batcher wastes capacity: late requests can't join an in-flight batch, so even at c=64 we observe `avg_batch_size ≈ 15` and **c=64 throughput < c=32**. Continuous batching fixes both (sequences can join/leave the batch at any decode step). Paged KV is the enabler — variable-length sequences in a single running batch can't share contiguous KV allocations. They ship together. |
| **2b** | **TP=8** (Megatron-style, replicated experts) | Multiplies compute and memory bandwidth by ~7×. Has to come **after** 2a, because TP doesn't help if you can't fill the existing GPU first. Also gives the routed experts more aggregate capacity. |
| **3** | **Custom Helion kernels** (MoE grouped GEMM, DeltaNet recurrence) | Squeezes the per-step time once the engine is saturating compute. Diff-tested against the torch reference. |
| **4** | **EP / hybrid TP+EP A/B** | Only meaningful once Phase 2a+2b is stable and the routing histogram is populated. Gated on three measurable signals (see below). |
| **5** | **Tuning** — Helion autotuning, scheduler windows, CUDA graphs | Last 20-40%, after the structure is settled. |

PP (pipeline parallelism) is **rejected** for this workload — see below.

The decision below for parallelism is **TP=8 first, then evaluate hybrid TP+EP**.
This document is the receipt — what we considered, why, and which
microbenchmarks settle the open questions.

## What the model actually looks like (the constraints, not theory)

| | |
|---|---|
| Total / active params | 35B / 3B |
| Layers | 40 (10 full-attn + 30 linear-attn, pattern `[L,L,L,F]×10`) |
| Attention | 16 heads, 2 KV heads (GQA, head_dim=256) |
| Linear-attn (Gated DeltaNet) | 32 v-heads, 16 k-heads, head_dim=128, conv_k=4, mamba SSM in fp32 |
| MoE | 256 experts, 8 routed + 1 shared per token, intermediate=512 |
| MoE active params | ~3B per token (dominates compute) |
| Vocab | 248,320 |
| Workload | 1024 in / 1024 out, c=1..64, scoring weight skewed toward c≥16 |

The MoE block is the bulk of the compute. With only 8 experts active per
token and 256 total, **routing is sparse** — at our largest evaluation
batches (64 tokens of new decode work) only `min(64×8, 256) ≤ 256`
experts touch any given step, meaning most experts sit idle most of the
time. This is the parameter that dominates the parallelism choice.

## The four candidates

| | per-MoE-layer comm | per-attn-layer comm | memory of routed experts/rank | code complexity | comments |
|---|---|---|---|---|---|
| **TP=8 (replicated experts)** | 2× all-reduce (down-proj output) | 1× all-reduce (o-proj output) | 256 experts × ~12 MB = ~3 GB (full set) | low | the simplest thing that works; what vLLM does for this model family |
| **EP=8 (32 experts/rank)** | 2× all-to-all (token shuffle in/out) | unchanged from non-TP | ~400 MB | medium | wins iff all-to-all messages > all-reduce at the actual token counts; sensitive to routing skew |
| **Hybrid TP=2 + EP=4** | all-reduce (intra-group) + all-to-all (cross-group) | intra-group all-reduce | ~800 MB | high | most flexible, also most code |
| **PP** | activations between stages (cheap) | unchanged | full model on subset of ranks | medium | bubble at small batches kills throughput; only worth it once continuous batching is in |

We're explicitly **not** considering pure data parallelism (8 separate
copies). 8 × 70 GB > 140 GB per GPU; doesn't fit at BF16, and even if it
did the inter-rank weight memory waste is enormous.

## Why TP=8 first

1. **Single biggest lever, smallest delta from Phase 1.** Continuous
   batching + paged KV is *also* a huge change — bundling TP=8 in the
   same Phase 2 milestone means one big step instead of two cliff-edge
   integrations. The Megatron-style ColumnParallel/RowParallel pattern
   maps cleanly onto HF's modeling code via wrap-on-load.
2. **MoE all-reduce sizes are not much larger than dense TP.** Each MoE
   down-proj output is `[tokens, hidden=2048]`, same as a dense attention
   layer. We're already paying that with full attention; another 30
   layers of similar all-reduces is not catastrophic.
3. **Replicated experts kill the routing-skew failure mode.** If at c=64
   most expert activations land on 16 of 256 experts, an EP=8 layout
   serializes 16 expert workers behind 1 GPU's worth of compute. TP=8
   sees no skew — every rank does 1/8 of every expert.
4. **vLLM picks TP=8 here.** Their published baseline numbers
   (12,810 tok/s @ c=64 in the README) are TP=8 with replicated
   experts. That's the bar we're judged against; matching the baseline's
   strategy first is the lowest-risk way to know we're not regressing on
   something orthogonal.

## When EP becomes interesting

Three measurable signals need to flip before we move from "TP=8
replicated" to "TP+EP hybrid":

1. **Routing histogram is not skewed at the actual token counts.**
   `engine/runtime/metrics.py` already has a slot for the per-layer
   per-expert routing histogram. Once Phase 2 populates it, we want to
   see that the *average ratio of (top-loaded expert)/(median expert)* is
   under ~3× at our eval batches. If it's 10×, EP loses to skew.
2. **All-to-all latency at our message sizes < 2× all-reduce latency.**
   `bench/microbench_comms.py` (to be written) sweeps NCCL all-reduce
   and all_to_all_single across the message sizes we actually see in
   the trace. NVLink's bandwidth makes this a measurement, not a
   prediction.
3. **TP=8's MoE region in the CLI table is the dominant cost.** The
   region table exists. If "moe.grouped_gemm" is < 30% of `engine.model.generate`
   total time, EP doesn't have enough to take from to be worth it.

If all three flip, we prototype TP=8 with **EP-inside-TP** (still 8
ranks, MoE experts shuffled across the 8 with 2× all-to-all per MoE
layer instead of 2× all-reduce). The plumbing for this lives in
`engine/parallel/ep.py` (to be written). Linear-attention layers stay
TP-sharded the same way regardless.

## Why not pipeline parallelism

PP across 8 GPUs would put 5 layers per stage. The pipeline bubble at
batch=B and stages=S has overhead `(S-1)/(B+S-1)`, so for B=64 that's
`7/71 = 9.9%` lost — modest. The killer is small batches: B=4 gives
`7/11 = 63%` lost. Since c=1..8 collectively carries 6/22 of the score
weight, sacrificing them is bad math. Only worth revisiting if Phase 2's
continuous batcher reliably sustains B≥32 even at low concurrency
(e.g. via prefill chunking that fills the pipe).

## Specific Phase 2 sub-tasks for parallelism

1. `engine/parallel/comm.py` — `torch.distributed` (nccl) init, rank discovery, `tp_group` / `ep_group` placeholders.
2. `engine/parallel/tp.py` — hand-rolled `ColumnParallelLinear`, `RowParallelLinear`, `VocabParallelEmbedding`. Megatron-style: wrap an `nn.Linear`, slice weights at load, all-reduce in forward.
3. `engine/parallel/loader.py` — per-rank safetensors mmap → slice → device. Sharding plan persisted as JSON for fast relaunches.
4. Sharding rules applied to the HF modeling code monkey-patch on load:
   - **Self-attention** (`q_proj`, `k_proj`, `v_proj` → Column; `o_proj` → Row).
   - **DeltaNet** (head-dim split for in/out + gate projections; recurrent state stays rank-local). If sharding the SSM is hairy, fall back to **replicating DeltaNet layers** (no comm) — bounded ceiling hit, 30 layers vs the bigger MoE+attn cost.
   - **MoE** (shared up/gate Column, down Row; routed experts replicated for Phase 2). EP behind a `parallel.moe_mode = {"replicated", "expert_parallel"}` flag.
   - **Embedding / lm_head** → VocabParallel.
5. `bench/microbench_comms.py` — NCCL all-reduce vs all_to_all_single sweeps for the EP-vs-TP A/B.

## TL;DR

> 1. **Phase 2a: continuous batching + paged KV** — biggest single lever,
>    fixes the c=64 < c=32 cliff, fills the batches we already have.
> 2. **Phase 2b: TP=8 with replicated experts** — multiplies the now-saturated
>    GPU compute by ~7×. Megatron-style hand-rolled wrappers, no fancy
>    parallelism libraries fighting our MoE/DeltaNet ops.
> 3. **Phase 4: EP / hybrid TP+EP A/B** — only after the routing histogram
>    is populated and we have NCCL all-to-all microbench numbers. Gate
>    promotion on ≥10% weighted score win without regressing c=1..8.
