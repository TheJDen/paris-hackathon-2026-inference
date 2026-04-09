"""Slot-pool KV cache + DeltaNet state cache, sliced into HF's Cache contract.

We bypass `model.generate()` entirely, but we still call HF's modeling layers
(`Qwen3_5MoeAttention`, `Qwen3_5MoeGatedDeltaNet`, ...) directly and they
expect a `transformers.cache_utils.Cache`-shaped object as `past_key_values`.
This file gives them one.

The big idea is **slot pooling** — pre-allocate one big tensor per layer
that holds state for *all* concurrent slots, and gather/scatter the active
batch's rows around each forward pass.

  * Full-attention layers see a `[num_slots, max_seq_len, num_kv_heads, head_dim]`
    tensor per layer. Each in-flight sequence owns one row (`slot_idx`); per-token
    K/V append happens at `(slot_idx, position)`. The `update()` method writes
    the new K/V into the right slot positions and returns a padded
    `[B, H, max_kv_seq, D]` view for the attention call.
  * Linear-attention (Gated DeltaNet) layers read `cache.layers[i].conv_states`
    and `recurrent_states` as direct attributes and mutate them in place via
    fla / causal_conv1d kernels. We give them an **active view** that's
    gathered from the slot pool before forward and scattered back after.
  * The runner hands a `BatchSlots` view to the cache before each forward
    pass via `set_batch()`, and reclaims any in-place mutations via
    `commit_batch()` afterward.

Per-layer state lives in `self.layers[layer_idx]` like HF's DynamicCache,
so any model code that reaches in via `cache.layers[layer_idx].conv_states`
also works.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch


log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# BatchSlots — what the runner sets before each forward pass
# --------------------------------------------------------------------------- #


@dataclass
class BatchSlots:
    """Per-forward routing info the cache reads via `cache.batch`.

    `slot_ids[i]` is the slot index of batch row i (so the cache knows
    where to write/read inside its big slot-pool tensors).

    `write_positions[i]` is a tensor of int64 — the absolute positions
    inside slot `slot_ids[i]` the new K/V tokens land at.

    For prefill of one slot with L prompt tokens:
        slot_ids = [s]
        write_positions = [[0, 1, 2, ..., L-1]]

    For decode of N slots, each emitting one token at its current length:
        slot_ids = [s0, s1, ..., sN-1]
        write_positions = [[len0], [len1], ..., [lenN-1]]

    `kv_seq_lens[i]` = TOTAL cached length for slot i AFTER this step's
    write. The attention layer needs this to slice K_full / V_full.
    """

    slot_ids: torch.Tensor          # [B], int64
    write_positions: list[torch.Tensor]  # B tensors of int64 (variable length per row)
    query_lens: torch.Tensor        # [B], int64
    kv_seq_lens: torch.Tensor       # [B], int64
    is_prefill: bool

    @property
    def batch_size(self) -> int:
        return int(self.slot_ids.shape[0])


# --------------------------------------------------------------------------- #
# Per-layer slot-pool sub-objects
# --------------------------------------------------------------------------- #


class _SlotLayerBase:
    """Common per-layer surface our HF-facing cache exposes via .layers[i].

    HF's modeling code occasionally pokes at `cache.layers[layer_idx]`
    methods directly, so the per-layer objects need to look enough like
    HF's `CacheLayerMixin` to satisfy duck typing.
    """

    parent: "SlotPoolCache | None" = None

    def get_seq_length(self) -> int:
        if self.parent is None or self.parent.batch is None:
            return 0
        return int(self.parent.batch.kv_seq_lens.max().item())

    def get_max_cache_shape(self) -> int:
        if self.parent is None:
            return 0
        return self.parent.max_seq_len

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        if self.parent is None or self.parent.batch is None:
            return query_length, 0
        return int(self.parent.batch.kv_seq_lens.max().item()), 0


class _FullAttentionSlotLayer(_SlotLayerBase):
    """Slot-pool K/V tensors for one full-attention layer.

    Shape `[num_slots, max_seq_len, num_kv_heads, head_dim]`. We allocate
    once at startup, then write per-slot per-position.
    """

    def __init__(
        self,
        num_slots: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.k = torch.zeros(
            num_slots, max_seq_len, num_kv_heads, head_dim,
            dtype=dtype, device=device,
        )
        self.v = torch.zeros_like(self.k)


class _LinearAttentionSlotLayer(_SlotLayerBase):
    """Per-slot recurrent state + conv state for one Gated DeltaNet layer.

    Pool tensors are **pre-allocated** at construction using shapes derived
    from text_config (we used to lazy-init from the model's first emission,
    but that ran *inside* `torch.inference_mode()` and produced inference
    tensors that the scheduler couldn't reset later).

    Conv state shape per HF modeling code:
        in_proj_qkv = nn.Linear(hidden_size, key_dim*2 + value_dim, bias=False)
        mixed_qkv:    [B, L, key_dim*2 + value_dim]  → transpose → [B, conv_dim, L]
        conv_state:   [B, conv_dim, conv_kernel_size]   (F.pad to fill kernel width)
    where `conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim`.

    Recurrent state shape from fla `chunk_gated_delta_rule`:
        [B, num_v_heads, head_k_dim, head_v_dim]
    after the model's repeat_interleave to lift K up to num_v_heads.

    The `conv_states` and `recurrent_states` attributes are the **active
    batch view** — gathered from the pool before each forward, scattered
    back after. The model reads them as `cache.layers[i].conv_states`,
    mutates them in place via `causal_conv1d_update`, and the mutations
    flow back to the pool on `cache.commit_batch()`.
    """

    def __init__(
        self,
        num_slots: int,
        conv_dim: int,
        conv_kernel: int,
        rec_num_heads: int,
        rec_head_k_dim: int,
        rec_head_v_dim: int,
        device: torch.device,
        kv_dtype: torch.dtype,
        state_dtype: torch.dtype,
    ) -> None:
        self.num_slots = num_slots
        self.device = device
        # Conv state lives in the model's working dtype (BF16). The
        # recurrent SSM state is fp32 (text_config.mamba_ssm_dtype).
        self._pool_conv_states = torch.zeros(
            num_slots, conv_dim, conv_kernel,
            dtype=kv_dtype, device=device,
        )
        self._pool_recurrent_states = torch.zeros(
            num_slots, rec_num_heads, rec_head_k_dim, rec_head_v_dim,
            dtype=state_dtype, device=device,
        )
        # Active batch view — set by parent.set_batch(), scattered by commit.
        self.conv_states: torch.Tensor | None = None
        self.recurrent_states: torch.Tensor | None = None

        # Graph-mode persistent buffers.  When the cache is in graph mode for
        # a specific bucket, these hold PERSISTENT tensors that the captured
        # CUDA graph references.  gather_for_batch writes INTO them (copy_)
        # instead of allocating a fresh clone, so the graph's captured
        # pointers remain valid across replays.  Keyed by bucket size.
        self._graph_conv_bufs: dict[int, torch.Tensor] = {}
        self._graph_rec_bufs: dict[int, torch.Tensor] = {}
        # The currently active bucket (None → eager, clone-based path).
        self._graph_active_bucket: int | None = None

    # ------------------------- graph-mode buffers -----------------------

    def ensure_graph_buffers(self, bucket: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate persistent conv/recurrent buffers for this bucket if missing.

        Returns the (conv_buf, rec_buf) pair as stable persistent tensors
        of shape ``[bucket, conv_dim, conv_kernel]`` and
        ``[bucket, num_v_heads, head_k_dim, head_v_dim]`` respectively.
        The captured CUDA graph will reference these exact tensors.
        """
        if bucket not in self._graph_conv_bufs:
            self._graph_conv_bufs[bucket] = torch.zeros(
                bucket,
                *self._pool_conv_states.shape[1:],
                dtype=self._pool_conv_states.dtype,
                device=self.device,
            )
            self._graph_rec_bufs[bucket] = torch.zeros(
                bucket,
                *self._pool_recurrent_states.shape[1:],
                dtype=self._pool_recurrent_states.dtype,
                device=self.device,
            )
        return self._graph_conv_bufs[bucket], self._graph_rec_bufs[bucket]

    def activate_graph_bucket(self, bucket: int) -> None:
        """Switch this layer into graph mode for the given bucket.

        After this call, ``self.conv_states`` / ``recurrent_states`` point at
        the persistent buffers for ``bucket`` and will be reused by subsequent
        ``gather_for_batch`` calls.
        """
        conv_buf, rec_buf = self.ensure_graph_buffers(bucket)
        self.conv_states = conv_buf
        self.recurrent_states = rec_buf
        self._graph_active_bucket = bucket

    def deactivate_graph_mode(self) -> None:
        """Return to the eager clone-based path (does not free buffers)."""
        self._graph_active_bucket = None
        # Drop the active references so later eager gather_for_batch calls
        # allocate fresh clones (which is what eager code expects).
        self.conv_states = None
        self.recurrent_states = None

    # ------------------------- gather / scatter -------------------------

    def gather_for_batch(self, slot_ids: torch.Tensor) -> None:
        """Materialize the active view from the pool. Called before forward.

        In graph mode, copies INTO the persistent buffers so pointers stay
        stable across CUDA graph replays.  The number of real rows must be
        <= ``bucket``; callers are expected to pad ``slot_ids`` up to the
        bucket size themselves (the runner does this in the graph path).
        """
        if self._graph_active_bucket is not None:
            bucket = self._graph_active_bucket
            assert slot_ids.shape[0] == bucket, (
                f"graph mode bucket={bucket} but got {slot_ids.shape[0]} slot_ids"
            )
            conv_buf = self._graph_conv_bufs[bucket]
            rec_buf = self._graph_rec_bufs[bucket]
            # Index-select into the persistent buffers.  Using out= would be
            # ideal but index_select doesn't support out= with cuda indices in
            # all torch versions; a copy_() of index_select's result is
            # equivalent and keeps the destination pointer stable.
            conv_buf.copy_(self._pool_conv_states.index_select(0, slot_ids))
            rec_buf.copy_(self._pool_recurrent_states.index_select(0, slot_ids))
            # Attributes already point at the persistent buffers (set by
            # activate_graph_bucket); nothing else to do.
            return

        # Eager path: clone so the model's in-place mutations don't fight
        # other slots.
        self.conv_states = self._pool_conv_states.index_select(0, slot_ids).clone()
        self.recurrent_states = self._pool_recurrent_states.index_select(0, slot_ids).clone()

    def scatter_after_batch(self, slot_ids: torch.Tensor) -> None:
        """Persist the (possibly mutated) active view back into the pool."""
        if self.conv_states is not None:
            # Reshape if the model emitted a different trailing layout
            # (e.g. fla can pack the recurrent state in [num_heads, k, v]
            # vs [num_heads, v, k]). The pool was allocated with the
            # config-derived layout and that's our source of truth.
            if self.conv_states.shape[1:] != self._pool_conv_states.shape[1:]:
                pass  # let the assignment raise so we notice
            self._pool_conv_states.index_copy_(0, slot_ids, self.conv_states.to(self._pool_conv_states.dtype))
        if self.recurrent_states is not None:
            if self.recurrent_states.shape[1:] != self._pool_recurrent_states.shape[1:]:
                pass
            self._pool_recurrent_states.index_copy_(0, slot_ids, self.recurrent_states.to(self._pool_recurrent_states.dtype))

    def reset_slot(self, slot_id: int) -> None:
        self._pool_conv_states[slot_id].zero_()
        self._pool_recurrent_states[slot_id].zero_()


# --------------------------------------------------------------------------- #
# The Cache itself
# --------------------------------------------------------------------------- #


class SlotPoolCache:
    """Hybrid slot-pool cache for Qwen3.5-MoE.

    NOT subclassing `transformers.cache_utils.Cache` because subclassing
    HF's Cache pulls in its own constructor expectations. We expose the
    same DUCK-typed surface (`update`, `get_seq_length`, `layers`,
    `update_conv_state`, `update_recurrent_state`, `get_mask_sizes`,
    `has_previous_state`) — that's all the modeling code touches.
    """

    def __init__(
        self,
        *,
        num_slots: int,
        max_seq_len: int,
        layer_types: list[str],
        num_kv_heads: int,
        head_dim: int,
        # Linear-attention dims (from text_config)
        linear_num_k_heads: int,
        linear_num_v_heads: int,
        linear_head_k_dim: int,
        linear_head_v_dim: int,
        linear_conv_kernel: int,
        kv_dtype: torch.dtype,
        state_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len
        self.layer_types = layer_types
        self.device = device
        self.kv_dtype = kv_dtype
        self.state_dtype = state_dtype

        # Per-slot active sequence length (number of tokens whose K/V is in
        # the cache). Indexed by slot_idx.
        self.slot_lengths = torch.zeros(num_slots, dtype=torch.int64, device=device)

        # Per-layer slot-pool sub-objects.
        layers: list[_FullAttentionSlotLayer | _LinearAttentionSlotLayer] = []
        for i, t in enumerate(layer_types):
            if t == "full_attention":
                layers.append(
                    _FullAttentionSlotLayer(
                        num_slots=num_slots,
                        max_seq_len=max_seq_len,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        dtype=kv_dtype,
                        device=device,
                    )
                )
            elif t == "linear_attention":
                # Conv state width: 2 × key_dim + value_dim where
                # key_dim = num_k_heads × head_k_dim, value_dim = num_v_heads × head_v_dim
                key_dim = linear_num_k_heads * linear_head_k_dim
                value_dim = linear_num_v_heads * linear_head_v_dim
                conv_dim = 2 * key_dim + value_dim
                layers.append(
                    _LinearAttentionSlotLayer(
                        num_slots=num_slots,
                        conv_dim=conv_dim,
                        conv_kernel=linear_conv_kernel,
                        # Recurrent state shape after K repeat-interleave to v_heads:
                        rec_num_heads=linear_num_v_heads,
                        rec_head_k_dim=linear_head_k_dim,
                        rec_head_v_dim=linear_head_v_dim,
                        device=device,
                        kv_dtype=kv_dtype,
                        state_dtype=state_dtype,
                    )
                )
            else:
                raise ValueError(f"unknown layer_type at layer {i}: {t!r}")
        self.layers = layers
        for layer in self.layers:
            layer.parent = self

        # Set by the runner before each model forward pass.
        self.batch: BatchSlots | None = None

    # --------------------------------------------------------------- helpers

    def reset_slot(self, slot_id: int) -> None:
        """Free a slot — called when a sequence finishes."""
        self.slot_lengths[slot_id] = 0
        for layer in self.layers:
            if isinstance(layer, _LinearAttentionSlotLayer):
                layer.reset_slot(slot_id)
        # Full-attention pool tensors don't need explicit zeroing — new
        # prefill overwrites them and slot_lengths gates the read range.

    def set_batch(self, batch: BatchSlots) -> None:
        """Called by the runner before each model forward.

        Materializes per-layer active views for the linear-attention
        layers (gather from pool). Full-attention layers handle their
        own write/read inside `update()`.
        """
        self.batch = batch
        slot_ids = batch.slot_ids.long()
        for layer in self.layers:
            if isinstance(layer, _LinearAttentionSlotLayer):
                layer.gather_for_batch(slot_ids)

    def commit_batch(self) -> None:
        """Called by the runner after each model forward.

        Scatters any in-place mutations to per-layer active views (linear
        attention layers) back into the slot pool.
        """
        if self.batch is None:
            return
        slot_ids = self.batch.slot_ids.long()
        for layer in self.layers:
            if isinstance(layer, _LinearAttentionSlotLayer):
                layer.scatter_after_batch(slot_ids)
        # Note: we leave self.batch set so any post-forward code that
        # peeks at it (e.g. metrics) still sees the right thing. The next
        # set_batch() call resets it.

    # --------------------------------------------------------------- DynamicCache-shaped API

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Max active slot length, used by the model to derive position_ids."""
        if self.batch is None:
            return int(self.slot_lengths.max().item()) if self.num_slots else 0
        return int(self.batch.kv_seq_lens.max().item())

    def get_max_cache_shape(self) -> int | None:
        return self.max_seq_len

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    @property
    def seen_tokens(self) -> int:
        return self.get_seq_length()

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        """HF mask construction calls this. Return (kv_length, kv_offset).

        For our slot-pool, every full-attention layer's update() returns a
        [B, H, max_kv_seq, D] view, so the mask is built against max_kv_seq
        with offset 0. We do NOT add `query_length` because our `kv_seq_lens`
        already accounts for the new tokens.
        """
        if self.batch is None:
            return query_length, 0
        return int(self.batch.kv_seq_lens.max().item()), 0

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        """Linear-attention layers ask this to decide between
        'first time, fresh state' (prefill) vs 'continuing recurrence'
        (decode). True iff we're in a decode step — every running slot
        already has a recurrent state from its prefill.
        """
        if self.batch is None:
            return False
        return not self.batch.is_prefill

    def is_compileable(self) -> bool:
        return False

    # --------------------------------------------------------------- full-attention update

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append (K, V) for one full-attention layer's batched forward.

        Inputs from HF modeling code:
          key_states / value_states: [B, num_kv_heads, L, head_dim]
          B = batch size of this forward pass (== self.batch.batch_size)
          L = number of new tokens this row contributes (1 for decode,
              prompt_len for prefill of one slot at a time)

        We write into the slot-pool tensor at the per-row write_positions,
        then return (K_full, V_full) of shape
          [B, num_kv_heads, max(seq_lens), head_dim]
        with the unused tail zero (the attention mask masks it out).
        """
        layer = self.layers[layer_idx]
        assert isinstance(layer, _FullAttentionSlotLayer)
        assert self.batch is not None, "cache.set_batch must be called before forward"
        batch = self.batch
        B, H, L, D = key_states.shape
        assert B == batch.batch_size, f"batch mismatch: K has {B}, batch says {batch.batch_size}"

        # ---- Vectorized write ------------------------------------------------
        # Build flat (slot_idx, position) index pairs covering all (b, l) pairs.
        # write_positions is list[Tensor], variable length per row (though in
        # practice all rows share the same L: =1 for decode, =prompt_len for
        # B=1 prefill).  We cat to get a single flat positions vector and
        # repeat each slot_id once per position.
        slot_ids_long = batch.slot_ids.long()  # [B]
        write_pos_list = batch.write_positions  # list of B int64 tensors
        # Lengths per row: tensor [B] of ints (each == L in uniform case).
        pos_lens = torch.tensor(
            [p.shape[0] for p in write_pos_list],
            dtype=torch.int64,
            device=layer.k.device,
        )  # [B]
        # Flat position indices: [total_tokens]
        flat_positions = torch.cat(write_pos_list)  # [total_tokens]
        # Repeat each slot_id pos_lens[b] times: [total_tokens]
        flat_slot_ids = slot_ids_long.repeat_interleave(pos_lens)  # [total_tokens]

        # key_states / value_states: [B, H, L, D] → [B, real_len_b, H, D] → [total, H, D]
        # Slice each row to its real length (pos_lens[b]) before flattening.
        # This is critical for batched prefill with right-padding: the model
        # computes K/V for all B*max_L positions including pad tokens, but we
        # only write the real (non-pad) positions into the cache.  Using the
        # full L here would write garbage K/V from pad positions into the slot
        # pool, corrupting subsequent decode steps.
        real_lens = pos_lens.tolist()  # list[int], one per row
        k_flat = torch.cat(
            [key_states[b, :, :real_lens[b], :].permute(1, 0, 2) for b in range(B)]
        ).to(layer.k.dtype)   # [total_tokens, H, D]
        v_flat = torch.cat(
            [value_states[b, :, :real_lens[b], :].permute(1, 0, 2) for b in range(B)]
        ).to(layer.v.dtype)   # [total_tokens, H, D]

        layer.k[flat_slot_ids, flat_positions] = k_flat
        layer.v[flat_slot_ids, flat_positions] = v_flat

        # Update slot lengths in one shot (decode: all B rows; prefill: B==1).
        self.slot_lengths.index_copy_(0, slot_ids_long, batch.kv_seq_lens.long())

        # ---- Vectorized read -------------------------------------------------
        # Gather all slot rows in one call → [B, max_seq_len, H, D]
        max_s = int(batch.kv_seq_lens.max().item())
        gathered_k = torch.index_select(layer.k, 0, slot_ids_long)  # [B, max_seq_len, H, D]
        gathered_v = torch.index_select(layer.v, 0, slot_ids_long)  # [B, max_seq_len, H, D]

        # Slice to the active window and permute to [B, H, max_s, D].
        out_k = gathered_k[:, :max_s, :, :].permute(0, 2, 1, 3).contiguous()
        out_v = gathered_v[:, :max_s, :, :].permute(0, 2, 1, 3).contiguous()

        # Zero-out tail tokens for rows whose kv_seq_len < max_s so that
        # attention scores on padding positions are forced to zero (matching
        # the prior loop behaviour of writing only [:seq_len] into zeros).
        # Build mask [B, 1, max_s, 1]: True where position < kv_seq_len[b].
        if max_s > 1:
            # arange [1, 1, max_s, 1] vs kv_seq_lens [B, 1, 1, 1]
            pos_range = torch.arange(max_s, device=layer.k.device).view(1, 1, max_s, 1)
            kv_lens = batch.kv_seq_lens.view(B, 1, 1, 1)  # [B,1,1,1]
            valid_mask = pos_range < kv_lens  # [B, 1, max_s, 1] bool
            out_k = out_k * valid_mask
            out_v = out_v * valid_mask

        return out_k, out_v

    # --------------------------------------------------------------- linear-attention updates

    def update_conv_state(
        self,
        new_conv_state: torch.Tensor,
        layer_idx: int,
        cache_init: bool = False,
        **_kwargs,
    ) -> torch.Tensor:
        """Set the active view for this layer's conv state.

        The shape is `[B, 2*key_dim + value_dim, conv_kernel_size]`. The
        scatter to the pool happens in `commit_batch()`.
        """
        layer = self.layers[layer_idx]
        assert isinstance(layer, _LinearAttentionSlotLayer)
        assert self.batch is not None
        if layer._graph_active_bucket is not None and layer.conv_states is not None:
            layer.conv_states.copy_(new_conv_state.to(layer.conv_states.dtype))
            return layer.conv_states
        layer.conv_states = new_conv_state
        return new_conv_state

    def update_recurrent_state(
        self,
        new_recurrent_state: torch.Tensor,
        layer_idx: int,
        **_kwargs,
    ) -> torch.Tensor:
        """Set the active view for this layer's recurrent state.
        Scatter to the pool happens in commit_batch.

        In graph mode the active recurrent buffer is a PERSISTENT tensor
        that the captured CUDA graph holds a direct pointer to.  We must
        not rebind the attribute (that would leave the graph referencing
        an orphan tensor); instead we copy the new state IN PLACE into
        the persistent buffer so that the copy is itself captured and
        subsequent replays land the updated state in the same memory.
        """
        layer = self.layers[layer_idx]
        assert isinstance(layer, _LinearAttentionSlotLayer)
        assert self.batch is not None
        if layer._graph_active_bucket is not None and layer.recurrent_states is not None:
            # In-place update into the persistent buffer.  Safe under CUDA
            # graph capture: the copy_ becomes part of the captured kernel
            # stream.
            layer.recurrent_states.copy_(new_recurrent_state.to(layer.recurrent_states.dtype))
            return layer.recurrent_states
        layer.recurrent_states = new_recurrent_state
        return new_recurrent_state

    # --------------------------------------------------------------- diagnostics

    def memory_bytes(self) -> int:
        total = 0
        for layer in self.layers:
            if isinstance(layer, _FullAttentionSlotLayer):
                total += layer.k.element_size() * layer.k.numel() * 2  # K + V
            elif isinstance(layer, _LinearAttentionSlotLayer):
                total += layer._pool_conv_states.element_size() * layer._pool_conv_states.numel()
                total += layer._pool_recurrent_states.element_size() * layer._pool_recurrent_states.numel()
        return total

    def __repr__(self) -> str:
        n_full = sum(1 for l in self.layers if isinstance(l, _FullAttentionSlotLayer))
        n_lin = sum(1 for l in self.layers if isinstance(l, _LinearAttentionSlotLayer))
        return (
            f"SlotPoolCache(slots={self.num_slots}, max_seq_len={self.max_seq_len}, "
            f"full_attn_layers={n_full}, linear_layers={n_lin}, "
            f"mem_static={self.memory_bytes() / 1024**3:.2f} GB)"
        )
