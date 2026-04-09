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

# Graph-safe torch reference implementations for the linear-attention fast
# paths.  We patch each Qwen3NextGatedDeltaNet INSTANCE's bound method
# references to these during capture + replay so that:
#   * causal_conv1d_update — no Triton kernel, pure torch.nn.functional.
#   * recurrent_gated_delta_rule — no fla Triton kernel, pure torch loop.
# Both are importable from the HF modeling module; they exist precisely as
# the CPU/fallback path.
try:
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        torch_causal_conv1d_update as _torch_causal_conv1d_update,
        torch_recurrent_gated_delta_rule as _torch_recurrent_gated_delta_rule,
        torch_chunk_gated_delta_rule as _torch_chunk_gated_delta_rule,
    )
except Exception:  # pragma: no cover - import-time defensive
    _torch_causal_conv1d_update = None
    _torch_recurrent_gated_delta_rule = None
    _torch_chunk_gated_delta_rule = None


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


def _torch_rmsnorm_gated_forward(weight, eps, hidden_states, gate=None):
    """Pure-torch reference matching Qwen3NextRMSNormGated.forward.

    Used to replace fla's FusedRMSNormGated Triton kernel during graph
    capture, which would otherwise hit cudaErrorStreamCaptureUnsupported.
    """
    import torch.nn.functional as F

    input_dtype = hidden_states.dtype
    hs = hidden_states.to(torch.float32)
    variance = hs.pow(2).mean(-1, keepdim=True)
    hs = hs * torch.rsqrt(variance + eps)
    hs = weight * hs.to(input_dtype)
    if gate is not None:
        hs = hs * F.silu(gate.to(torch.float32))
    return hs.to(input_dtype)


def _patch_delta_net_instances(inner_model) -> list[tuple[object, str, object]]:
    """Swap each Qwen3NextGatedDeltaNet instance's bound fast-path functions
    to the pure-torch references so that capture AND replay never touch
    Triton / fla kernels on the decode hot path.

    The HF module captures its fast-path function references at
    ``__init__`` time as ``self.chunk_gated_delta_rule``,
    ``self.recurrent_gated_delta_rule``, ``self.causal_conv1d_update``.
    Module-level monkeypatching therefore has NO effect on already-built
    layers.  We walk the live module tree and rebind the attributes on
    each instance directly.

    Returns a list of ``(module, attr_name, original_value)`` tuples that
    the caller can pass to :func:`_restore_delta_net_instances` to undo.
    """
    saved: list[tuple[object, str, object]] = []
    if (
        _torch_causal_conv1d_update is None
        or _torch_recurrent_gated_delta_rule is None
    ):
        log.warning(
            "graph-capture: HF torch fallbacks not importable; CUDA graph "
            "capture will almost certainly fail on the linear-attention "
            "layers.  Rebuild transformers or disable CUDA graphs."
        )
        return saved

    patches = {
        "causal_conv1d_update": _torch_causal_conv1d_update,
        "recurrent_gated_delta_rule": _torch_recurrent_gated_delta_rule,
        # chunk path is only used for prefill (seq_len > 1); decode takes the
        # recurrent branch.  We patch it too in case warmup hits seq_len==1
        # with no prior state (first-ever forward), which would fall through
        # the chunk branch.
        "chunk_gated_delta_rule": _torch_chunk_gated_delta_rule
        if _torch_chunk_gated_delta_rule is not None
        else None,
    }

    count = 0
    for _name, m in inner_model.named_modules():
        if type(m).__name__ != "Qwen3NextGatedDeltaNet":
            continue
        count += 1
        for attr, replacement in patches.items():
            if replacement is None or not hasattr(m, attr):
                continue
            saved.append((m, attr, getattr(m, attr)))
            setattr(m, attr, replacement)
        # Also replace ``m.norm`` if it is an fla FusedRMSNormGated module.
        # The fla variant calls a Triton kernel that autotunes / host-syncs
        # and is not graph-capture-safe.  We override its ``forward`` with
        # the pure-torch reference, closing over the live weight/eps.
        norm = getattr(m, "norm", None)
        if norm is not None and type(norm).__name__ == "FusedRMSNormGated":
            w = norm.weight
            eps = getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6))

            def _patched_norm_forward(
                hidden_states, gate=None, _w=w, _eps=eps
            ):
                return _torch_rmsnorm_gated_forward(_w, _eps, hidden_states, gate)

            saved.append((norm, "forward", norm.forward))
            norm.forward = _patched_norm_forward  # type: ignore[method-assign]

    log.info(
        "graph-capture: patched %d Qwen3NextGatedDeltaNet instance(s) "
        "to torch fallbacks (%d attribute rebinds)",
        count,
        len(saved),
    )
    return saved


def _restore_delta_net_instances(saved: list[tuple[object, str, object]]) -> None:
    for m, attr, original in saved:
        setattr(m, attr, original)
    if saved:
        log.info(
            "graph-capture: restored %d Qwen3NextGatedDeltaNet attribute(s)",
            len(saved),
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

    def _activate_linear_layers_for_bucket(self, bucket: int) -> None:
        """Make every linear-attention layer in the cache point its active
        conv/recurrent attributes at the bucket's persistent buffers."""
        from engine.runtime.kv_cache import _LinearAttentionSlotLayer

        for layer in self.cache.layers:
            if isinstance(layer, _LinearAttentionSlotLayer):
                layer.activate_graph_bucket(bucket)

    def replay(
        self,
        batch_size: int,
        input_ids: torch.Tensor,       # [B, 1] int64
        attention_mask: torch.Tensor,  # [B, max_s] int64  (max_s <= max_seq_len)
        position_ids: torch.Tensor,    # [B, 1] int64
    ) -> torch.Tensor:
        """Run the decode forward via CUDA graph replay.

        Expects the runner to have already called ``cache.set_batch(...)``
        with the REAL-sized ``BatchSlots`` (B rows, not padded).  We then
        rebuild a bucket-padded ``BatchSlots`` and re-run ``set_batch`` so
        the linear-attention layers gather into the bucket's persistent
        CUDA-graph buffers.  The full-attention ``update()`` path also
        reads ``self.cache.batch`` at forward time; the padded view has
        the same real rows in slots [0:B] and dummy rows that replicate
        row 0 (same slot, kv_len, position), which is idempotent.

        Returns logits of shape [batch_size, vocab_size] — real rows only,
        dummy padding rows are sliced off.
        """
        bucket = _bucket_for(batch_size)
        if bucket not in self._entries:
            entry = self._capture(bucket)
        else:
            entry = self._entries[bucket]

        B_bucket = entry.bucket

        # --- build the bucket-padded BatchSlots and re-set the cache -----
        # The runner already called set_batch with real B rows; that ran
        # gather_for_batch in EAGER mode (clones).  We now switch every
        # linear layer into graph mode for this bucket and re-run set_batch
        # with a padded BatchSlots so the gather writes into the persistent
        # buffers the captured graph references.
        assert self.cache.batch is not None, "runner must call cache.set_batch before replay"
        real_batch = self.cache.batch

        from engine.runtime.kv_cache import BatchSlots

        if B_bucket == batch_size:
            padded_batch = real_batch
        else:
            pad_n = B_bucket - batch_size
            # Repeat row 0 for dummy rows (same slot, same kv_len, same
            # write_positions, same query_len).  Dummy rows compute the
            # same thing as row 0 → idempotent scatter.
            pad_slot_ids = torch.cat(
                [real_batch.slot_ids, real_batch.slot_ids[:1].expand(pad_n)],
                dim=0,
            )
            pad_query_lens = torch.cat(
                [real_batch.query_lens, real_batch.query_lens[:1].expand(pad_n)],
                dim=0,
            )
            pad_kv_seq_lens = torch.cat(
                [real_batch.kv_seq_lens, real_batch.kv_seq_lens[:1].expand(pad_n)],
                dim=0,
            )
            pad_write_positions = list(real_batch.write_positions) + [
                real_batch.write_positions[0] for _ in range(pad_n)
            ]
            padded_batch = BatchSlots(
                slot_ids=pad_slot_ids,
                write_positions=pad_write_positions,
                query_lens=pad_query_lens,
                kv_seq_lens=pad_kv_seq_lens,
                is_prefill=False,
            )

        # Activate graph-mode persistent buffers for every linear-attn layer
        # for THIS bucket, then re-run set_batch so gather_for_batch copies
        # into those persistent buffers.
        self._activate_linear_layers_for_bucket(B_bucket)
        self.cache.set_batch(padded_batch)

        # --- copy real rows into static input buffers ---
        entry.buf_input_ids[:batch_size].copy_(input_ids)
        if batch_size < B_bucket:
            entry.buf_input_ids[batch_size:].copy_(
                input_ids[:1].expand(B_bucket - batch_size, -1)
            )

        # attention_mask: real rows from caller, dummy rows mirror row 0.
        max_s = attention_mask.shape[1]
        entry.buf_attention_mask.zero_()
        entry.buf_attention_mask[:batch_size, :max_s].copy_(attention_mask)
        if batch_size < B_bucket:
            entry.buf_attention_mask[batch_size:, :max_s].copy_(
                attention_mask[:1].expand(B_bucket - batch_size, -1)
            )

        entry.buf_position_ids[:batch_size].copy_(position_ids)
        if batch_size < B_bucket:
            entry.buf_position_ids[batch_size:].copy_(
                position_ids[:1].expand(B_bucket - batch_size, -1)
            )

        # --- replay ---
        entry.graph.replay()

        # Scatter the (now mutated) persistent buffers back into the pool.
        # commit_batch uses self.cache.batch which is the padded batch; the
        # dummy rows write to their replicated slot (same value) — idempotent.
        self.cache.commit_batch()

        # Leave graph mode so subsequent eager code paths (prefill, other
        # non-graph decode) see the clone-based fast path.  The persistent
        # buffers themselves remain alive on the layers (they're held by
        # ``_graph_conv_bufs`` / ``_graph_rec_bufs`` dicts) so the captured
        # graph still references valid memory when the next replay runs.
        from engine.runtime.kv_cache import _LinearAttentionSlotLayer

        for layer in self.cache.layers:
            if isinstance(layer, _LinearAttentionSlotLayer):
                layer.deactivate_graph_mode()

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
        # IMPORTANT: warmup AND capture must run with every
        # Qwen3NextGatedDeltaNet INSTANCE's fast-path functions patched to the
        # pure-torch fallbacks.  fla's (and our Helion rewrite's) Triton
        # kernels call host-visible sync / autotune benchmarking that raises
        # ``cudaErrorStreamCaptureUnsupported`` inside ``torch.cuda.graph``.
        # Module-level monkeypatching does NOT work because
        # ``Qwen3NextGatedDeltaNet.__init__`` captures the function REFERENCES
        # at construction time (``self.chunk_gated_delta_rule = ...``,
        # ``self.recurrent_gated_delta_rule = ...``,
        # ``self.causal_conv1d_update = ...``), so we walk the module tree
        # and rebind each instance attribute directly.
        #
        # ALSO: every linear-attention cache layer must be in "graph mode"
        # for this bucket so that ``gather_for_batch`` copies state into the
        # PERSISTENT conv/recurrent buffers the captured graph will reference.
        # Without this, each decode step allocates fresh clones at new
        # addresses and the captured graph's kernel-arg pointers become stale
        # on the next replay (cudaErrorStreamCaptureInvalidated).
        saved_instance_patches = _patch_delta_net_instances(self.inner_model)
        self._activate_linear_layers_for_bucket(bucket)
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
            # Restore the original fla / Helion / causal_conv1d bindings on
            # each layer instance so that EAGER decode and PREFILL after
            # capture use the fast kernels.  Replay does not call Python, so
            # the captured graph is unaffected by this restoration.
            _restore_delta_net_instances(saved_instance_patches)
            # Return the linear-attn layers to eager (clone-based) mode so
            # prefill and other non-graph code paths work normally.
            from engine.runtime.kv_cache import _LinearAttentionSlotLayer

            for layer in self.cache.layers:
                if isinstance(layer, _LinearAttentionSlotLayer):
                    layer.deactivate_graph_mode()

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
