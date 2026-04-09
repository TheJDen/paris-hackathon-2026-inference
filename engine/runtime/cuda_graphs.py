"""CUDA graph capture for the decode hot path.

Motivation
----------
Phase 1 torch profile showed `Command Buffer Full` at 126 ms (~4% of CUDA
time) and 20 K `aten::copy_` calls per decode step — classic CPU-side
dispatch overhead from eager-mode HF modeling code.  CUDA graphs collapse
the entire per-step kernel-launch chain into a single replay, eliminating
all of that overhead for decode at small/medium batch sizes.

Scope
-----
DECODE ONLY.  Prefill is variable-length and not graph-friendly.

Bucket strategy (option a)
--------------------------
We capture ONE graph per batch-size bucket.  Buckets are the smallest
power-of-2 value in BUCKETS that is >= the real batch size B.  Smaller
real batches are padded to the bucket size by repeating slot 0 (which
always has a valid pool row); dummy output rows are discarded after replay.

The *kv* dimension varies every step (each slot grows by 1 token per step).
The existing attention_mask + cache `update()` path already handles this:
the attention_mask is written into a static persistent buffer before each
replay, and the cache pool tensors are fixed-size globals that the graph
accesses by reference.

What is INSIDE the graph
-------------------------
  * `inner_model(input_ids, attention_mask, position_ids, past_key_values,
                 use_cache=True)` — the full Qwen3_5MoeTextModel forward,
    including every attention layer, MoE MLP, norm, embed.
  * `lm_head(last_hidden)` — a single matmul; cheap to include, avoids an
    extra sync point.

What is OUTSIDE the graph
--------------------------
  * `cache.set_batch(...)` — calls `gather_for_batch` (index_select +
    clone) on every linear-attention layer.  index_select across batches
    with a data-dependent index is not graph-safe across calls.
  * `cache.commit_batch()` — scatters mutated linear-attention views back
    into the pool via index_copy_.  Same reason.
  * The sampler (`_sample`) — per-step temperature/top_p tensors vary per
    row, not worth the complexity of static buffers for the sampler.
  * Per-step Python tensor construction (input_ids list → tensor, etc.) —
    we do this before copy_-ing into the static buffers.

Linear-attention state handling
--------------------------------
`gather_for_batch` produces `layer.conv_states` and `layer.recurrent_states`
as CLONES of the pool rows, allocated *outside* the graph.  The graph replay
reads these (the model mutates them in-place via fla / causal_conv1d_update).
After replay, `commit_batch` scatter-writes the mutated views back to the
pool.  This is correct because the static input buffers *do not include*
the recurrent state — the graph sees the same Python-level attribute names
(`cache.layers[i].conv_states`) on every replay, and those attributes point
to freshly gathered tensors that the pre-replay setup populated.

IMPORTANT: because `gather_for_batch` allocates new clone tensors each step,
the *pointer* that the graph captured for `conv_states` / `recurrent_states`
is stale after the first replay.  The model forward inside the graph reads
these via Python attribute lookup (`cache.layers[i].conv_states`), NOT via
a persistent pointer captured at trace time.  CUDA graphs capture *kernel
launches* and their argument buffers, not Python attribute reads — so this
is safe as long as the tensors that `gather_for_batch` produced are the same
shape/dtype/device as the ones present during capture.  They will be,
because the pool has a fixed shape per-slot.

TODO (if this assumption breaks at runtime): pre-allocate static persistent
conv/recurrent buffers in the cache and copy into them instead of clone,
so the graph captured addresses are the exact persistent buffers.

Fla / Triton interaction
------------------------
`chunk_gated_delta_rule` and `causal_conv1d_update` are Triton kernels.
They *can* be captured in CUDA graphs, but:
  1. First call triggers JIT compilation — this happens at warmup, so
     capture (which calls the forward once before tracing) pays the Triton
     compile cost.  Subsequent replays are pure CUDA kernel replays.
  2. If fla uses `triton.autotune`, the best config is selected on the
     first real call and fixed for the graph.  Make sure warmup uses a
     representative batch size.
  3. If any Triton kernel allocates scratch memory dynamically based on
     tensor shapes, that allocation must be stable across replays.  fla's
     decode kernels are written to be graph-safe (fixed shapes at decode),
     but verify with `TRITON_PRINT_AUTOTUNING=1` if you see errors.

Capture timing
--------------
Lazy on first use of each bucket.  First invocation of a new bucket pays
~1-3 s capture cost (including Triton JIT on first ever capture).
Subsequent invocations hit the cached graph.

Usage (from model_runner.py)
-----------------------------
    graphs = DecodeGraphCache(
        inner_model=self.inner_model,
        lm_head=self.lm_head,
        cache=self.cache,
        max_seq_len=self.max_seq_len,
        device=self.device,
        vocab_size=vocab_size,
    )
    # per decode step:
    logits = graphs.replay(
        batch_size, input_ids_t, attention_mask_t,
        position_ids_t, slot_ids_t, kv_seq_lens_t,
    )  # shape [batch_size, vocab_size] (real rows only)
"""

from __future__ import annotations

import contextlib
import logging
from typing import Callable, NamedTuple

import torch


log = logging.getLogger(__name__)

# Bucket list — batch size is rounded UP to the next value here.
# Add 64, 128 if you scale beyond 64 slots.
BUCKETS: list[int] = [1, 2, 4, 8, 16, 32, 64]


def _bucket_for(batch_size: int) -> int:
    """Return the smallest bucket >= batch_size."""
    for b in BUCKETS:
        if b >= batch_size:
            return b
    raise ValueError(
        f"batch_size={batch_size} exceeds max bucket {BUCKETS[-1]}. "
        f"Add a larger bucket to BUCKETS."
    )


@contextlib.contextmanager
def _graph_safe_delta_rule():
    """Context manager: force the pure-torch fallback for
    ``chunk_gated_delta_rule`` across all known call sites.

    Why
    ---
    fla's ``chunk_gated_delta_rule`` (and our Helion rewrite) dispatch into
    Triton/Helion kernels that perform operations illegal under CUDA graph
    capture — specifically ``triton.autotune`` benchmark launches and
    host-visible event sync used for kernel specialization.  The first such
    call inside ``torch.cuda.graph(...)`` raises
    ``cudaErrorStreamCaptureUnsupported``.

    The pure-torch reference path in ``engine.kernels.delta_rule._torch_fallback``
    uses only stock PyTorch ops and is graph-capture-safe: its allocations go
    through PyTorch's graph-aware CUDA caching allocator, and it issues no
    host-side sync.  At decode (T=1 → padded to one chunk of 64) the extra
    kernel launches are irrelevant because they are *all* captured into the
    single decode graph, collapsed on replay.

    What this patches
    -----------------
    Every known import site of ``chunk_gated_delta_rule``:
      * ``transformers.models.qwen3_next.modeling_qwen3_next.chunk_gated_delta_rule``
        — the binding Qwen3_5MoeGatedDeltaNet.forward actually calls.
      * ``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` — belt-and-braces
        in case transformers re-imports from fla.
      * ``engine.kernels.delta_rule.chunk_gated_delta_rule`` — our own
        exported symbol, for any user code that imports it directly.

    Originals are restored on ``__exit__``.
    """
    from engine.kernels import delta_rule as _dr_mod

    # The graph-safe callable: bind the fallback with decode-default args.
    fallback = _dr_mod._torch_fallback

    def _safe(
        query,
        key,
        value,
        g,
        beta,
        chunk_size: int = 64,
        initial_state=None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        **_ignored,
    ):
        return fallback(
            query,
            key,
            value,
            g,
            beta,
            chunk_size,
            initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
        )

    patched: list[tuple[object, str, object]] = []

    def _try_patch(module_path: str, attr: str = "chunk_gated_delta_rule") -> None:
        try:
            import importlib

            mod = importlib.import_module(module_path)
        except Exception:
            return
        if not hasattr(mod, attr):
            return
        original = getattr(mod, attr)
        setattr(mod, attr, _safe)
        patched.append((mod, attr, original))

    _try_patch("transformers.models.qwen3_next.modeling_qwen3_next")
    _try_patch("fla.ops.gated_delta_rule")
    _try_patch("engine.kernels.delta_rule")

    if patched:
        log.info(
            "cuda-graph capture: patched %d chunk_gated_delta_rule site(s) "
            "to pure-torch fallback for the duration of capture",
            len(patched),
        )
    else:
        log.warning(
            "cuda-graph capture: no chunk_gated_delta_rule sites were found "
            "to patch — capture will likely fail if linear-attention layers "
            "invoke fla/Helion kernels"
        )

    try:
        yield
    finally:
        for mod, attr, original in patched:
            setattr(mod, attr, original)
        if patched:
            log.info("cuda-graph capture: restored original chunk_gated_delta_rule bindings")


class _GraphEntry(NamedTuple):
    """Everything we hold for one captured bucket."""
    graph: torch.cuda.CUDAGraph
    # Static input buffers (persistent, reused every step).
    buf_input_ids: torch.Tensor       # [B_bucket, 1]  int64
    buf_attention_mask: torch.Tensor  # [B_bucket, max_seq_len]  int64
    buf_position_ids: torch.Tensor    # [B_bucket, 1]  int64
    # Output buffer written by the captured forward.
    buf_logits: torch.Tensor          # [B_bucket, vocab_size]  (model dtype → float)
    bucket: int                       # B_bucket


class DecodeGraphCache:
    """Manages per-bucket CUDA graphs for the decode forward pass.

    Parameters
    ----------
    inner_model:
        The `Qwen3_5MoeTextModel` (i.e. `hf_model.model`) — the object
        whose `forward(input_ids, attention_mask, position_ids,
        past_key_values, use_cache)` we capture.
    lm_head:
        `hf_model.lm_head` — the final projection.  Captured inside the
        graph for efficiency.
    cache:
        The `SlotPoolCache` instance.  Its pool tensors are permanent
        globals; the graph captures operations that index into them.
        `set_batch` / `commit_batch` are called OUTSIDE the graph.
    max_seq_len:
        The maximum sequence length the cache was built for.  Used to
        size the static `attention_mask` buffer.
    device:
        CUDA device.
    vocab_size:
        Size of the vocabulary (output dimension of lm_head).
    """

    def __init__(
        self,
        inner_model: torch.nn.Module,
        lm_head: torch.nn.Module,
        cache,  # SlotPoolCache — avoid circular import, use duck type
        max_seq_len: int,
        device: torch.device,
        vocab_size: int,
    ) -> None:
        self.inner_model = inner_model
        self.lm_head = lm_head
        self.cache = cache
        self.max_seq_len = max_seq_len
        self.device = device
        self.vocab_size = vocab_size

        # bucket -> _GraphEntry
        self._entries: dict[int, _GraphEntry] = {}

        # Dedicated capture stream (separate from default).
        self._capture_stream = torch.cuda.Stream(device=device)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def replay(
        self,
        batch_size: int,
        input_ids: torch.Tensor,       # [B, 1] int64
        attention_mask: torch.Tensor,  # [B, max_s] int64  (max_s <= max_seq_len)
        position_ids: torch.Tensor,    # [B, 1] int64
    ) -> torch.Tensor:
        """Run the decode forward via CUDA graph replay.

        Captures the graph for this batch-size bucket on first call.

        Returns logits of shape [batch_size, vocab_size] — real rows only,
        dummy padding rows are sliced off.
        """
        bucket = _bucket_for(batch_size)
        if bucket not in self._entries:
            entry = self._capture(bucket)
        else:
            entry = self._entries[bucket]

        B_bucket = entry.bucket

        # --- copy real rows into static input buffers ---
        entry.buf_input_ids[:batch_size].copy_(input_ids)
        # Pad extra rows: repeat slot 0's input (any valid token works).
        if batch_size < B_bucket:
            entry.buf_input_ids[batch_size:].copy_(input_ids[:1].expand(B_bucket - batch_size, -1))

        # attention_mask: real rows from caller, dummy rows get mask=[1,0,...,0]
        # (position 0 is always valid for slot 0; this keeps causal attn healthy).
        max_s = attention_mask.shape[1]
        entry.buf_attention_mask.zero_()
        entry.buf_attention_mask[:batch_size, :max_s].copy_(attention_mask)
        if batch_size < B_bucket:
            # Dummy rows: attend to position 0 only (slot 0, kv_len=1).
            entry.buf_attention_mask[batch_size:, 0] = 1

        entry.buf_position_ids[:batch_size].copy_(position_ids)
        if batch_size < B_bucket:
            entry.buf_position_ids[batch_size:].zero_()  # pos 0 for dummies

        # --- replay ---
        # Synchronize so the copy_() ops above are visible to the capture
        # stream before replay begins.
        torch.cuda.current_stream(self.device).synchronize()
        entry.graph.replay()

        # scatter linear-attention views back to the pool — must happen AFTER
        # replay so the mutated conv/recurrent states are committed.
        # NOTE: set_batch was called by the runner BEFORE replay(); commit_batch
        # is our responsibility here since we replaced the eager forward.
        self.cache.commit_batch()

        # Return only the real rows (slice off padding).
        return entry.buf_logits[:batch_size].clone()

    # ------------------------------------------------------------------ #
    # capture
    # ------------------------------------------------------------------ #

    def _capture(self, bucket: int) -> _GraphEntry:
        """Capture the decode forward for a fixed batch size `bucket`.

        Allocates static input/output buffers, runs two warmup forwards
        (to trigger Triton JIT, fla kernel compilation, etc.), then
        captures the third forward into a CUDAGraph.
        """
        log.info(
            "CUDA graph capture: bucket=%d  (first call for this batch size; "
            "expect ~1-3 s for Triton JIT + graph trace)",
            bucket,
        )
        device = self.device

        # ---- Allocate static input buffers ----
        buf_input_ids = torch.zeros(bucket, 1, dtype=torch.int64, device=device)
        # attention_mask padded to full max_seq_len so shape is fixed.
        buf_attention_mask = torch.zeros(bucket, self.max_seq_len, dtype=torch.int64, device=device)
        buf_position_ids = torch.zeros(bucket, 1, dtype=torch.int64, device=device)

        # Seed with valid dummy values so warmup / capture don't hit
        # out-of-range index errors.  Slot 0 is always allocated.
        buf_input_ids.fill_(0)      # token 0 (pad/bos — any valid token)
        buf_attention_mask[:, 0] = 1  # each row attends to at least position 0
        buf_position_ids.fill_(0)

        # ---- Build a warm BatchSlots for capture ---
        # We need set_batch called with bucket rows before capture so the
        # cache has valid batch metadata (kv_seq_lens, etc.).  Use slot 0
        # for all dummy rows with kv_seq_len=1 (minimum valid length).
        from engine.runtime.kv_cache import BatchSlots  # local import to avoid circular

        dummy_slot_ids = torch.zeros(bucket, dtype=torch.int64, device=device)
        dummy_write_positions = [
            torch.zeros(1, dtype=torch.int64, device=device) for _ in range(bucket)
        ]
        dummy_query_lens = torch.ones(bucket, dtype=torch.int64, device=device)
        dummy_kv_seq_lens = torch.ones(bucket, dtype=torch.int64, device=device)
        dummy_batch = BatchSlots(
            slot_ids=dummy_slot_ids,
            write_positions=dummy_write_positions,
            query_lens=dummy_query_lens,
            kv_seq_lens=dummy_kv_seq_lens,
            is_prefill=False,
        )

        def _run_forward() -> torch.Tensor:
            """One warmup/capture forward — returns logits [bucket, vocab]."""
            self.cache.set_batch(dummy_batch)
            outputs = self.inner_model(
                input_ids=buf_input_ids,
                attention_mask=buf_attention_mask,
                position_ids=buf_position_ids,
                past_key_values=self.cache,
                use_cache=True,
            )
            self.cache.commit_batch()
            hidden_states = outputs.last_hidden_state  # [bucket, 1, hidden]
            logits = self.lm_head(hidden_states[:, -1, :])  # [bucket, vocab]
            return logits

        # ---- Warmup (outside graph, no_grad) ----
        # Two passes: first triggers Triton/fla JIT, second confirms stable state.
        #
        # IMPORTANT: warmup AND capture must run with the delta-rule call sites
        # patched to the pure-torch fallback.  fla's (and our Helion rewrite's)
        # Triton kernels call host-visible sync / autotune benchmarking that
        # raises ``cudaErrorStreamCaptureUnsupported`` inside ``torch.cuda.graph``.
        # The fallback is stock PyTorch and captures cleanly.  We patch across
        # warmup so the bf16 → fp32 cast shapes the fallback produces are what
        # the downstream layers see during capture as well (stable pool writes).
        _delta_patch = _graph_safe_delta_rule()
        _delta_patch.__enter__()
        try:
            with torch.no_grad():
                for warmup_i in range(2):
                    log.debug("CUDA graph warmup pass %d for bucket=%d", warmup_i + 1, bucket)
                    _ = _run_forward()
                torch.cuda.synchronize(device)

            # ---- Capture ----
            # NOTE: Do NOT use torch.inference_mode() inside the capture context.
            # inference_mode produces views that can't be graph-captured correctly
            # across calls (the inference mode bit on the output tensor makes it
            # incompatible with later non-inference-mode use).  Use no_grad only.
            #
            # The outer decode() method is decorated @torch.inference_mode() but
            # torch.cuda.graph() internally disables inference mode during capture.
            #
            # RNG STATE GUARD: torch.cuda.graph() saves the default CUDA generator's
            # RNG state at entry and registers it for "graph mode".  After the
            # capture context exits, the generator is restored — but PyTorch leaves
            # a flag that causes any subsequent offset increment OUTSIDE a graph
            # (e.g. torch.multinomial during prefill) to raise:
            #   "Offset increment outside graph capture encountered unexpectedly."
            # We explicitly snapshot and restore the default generator's state
            # around the entire capture block so it is never permanently put into
            # graph mode from the perspective of eager code running after this call.
            _saved_rng_state = torch.cuda.get_rng_state(device)

            g = torch.cuda.CUDAGraph()
            with torch.no_grad():
                # Run one more set_batch/commit_batch outside the capture so that
                # linear-attention state is fresh (the gather/scatter lives outside).
                self.cache.set_batch(dummy_batch)
                with torch.cuda.graph(g, stream=self._capture_stream):
                    # Only the inner_model forward + lm_head are in the graph.
                    # set_batch and commit_batch are OUTSIDE (they do index_select
                    # + index_copy_ with data-dependent indices — not graph-safe).
                    outputs = self.inner_model(
                        input_ids=buf_input_ids,
                        attention_mask=buf_attention_mask,
                        position_ids=buf_position_ids,
                        past_key_values=self.cache,
                        use_cache=True,
                    )
                    hidden_states = outputs.last_hidden_state   # [bucket, 1, hidden]
                    buf_logits = self.lm_head(hidden_states[:, -1, :])  # [bucket, vocab]
                self.cache.commit_batch()

            # Restore the default CUDA generator state so it is not permanently
            # left in "graph mode" after capture.  Eager callers (e.g. _sample's
            # torch.multinomial) must be able to increment the RNG offset freely.
            torch.cuda.set_rng_state(_saved_rng_state, device)

            torch.cuda.synchronize(device)
            log.info("CUDA graph capture complete: bucket=%d", bucket)
        finally:
            # Restore fla/Helion chunk_gated_delta_rule bindings — eager code
            # after this point (e.g. prefill) can use the fast path again.
            _delta_patch.__exit__(None, None, None)

        entry = _GraphEntry(
            graph=g,
            buf_input_ids=buf_input_ids,
            buf_attention_mask=buf_attention_mask,
            buf_position_ids=buf_position_ids,
            buf_logits=buf_logits,
            bucket=bucket,
        )
        self._entries[bucket] = entry
        return entry

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #

    def captured_buckets(self) -> list[int]:
        return sorted(self._entries.keys())

    def __repr__(self) -> str:
        return (
            f"DecodeGraphCache(buckets_captured={self.captured_buckets()}, "
            f"max_seq_len={self.max_seq_len}, vocab_size={self.vocab_size})"
        )
