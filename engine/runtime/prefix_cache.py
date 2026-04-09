"""Shared-prefix K/V cache for chat-template prefixes.

The Qwen chat template adds a deterministic prefix to every request:
  <|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n
(or just <|im_start|>user\\n if no system message is used).

Since every request in the throughput eval shares this prefix exactly,
we can:
  1. Reserve one slot (RESERVED_SLOT = num_slots - 1) at startup.
  2. Run a one-time prefill of the prefix tokens into that slot, caching
     their K/V (full-attention layers) and final recurrent state
     (linear-attention layers) for the prefix.
  3. At admit time, detect if a new request starts with those same tokens.
     If so, bulk-copy the prefix K/V into the new slot and tell the
     scheduler to start prefill at token prefix_len instead of 0.

Slot 0 is reserved by cuda_graphs.py as the dummy padding row, so we
use the LAST slot index instead to avoid collisions.

This file is intentionally free of torch.compile / CUDA graph concerns.
It only does tensor copies via .copy_() on the slot pool tensors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from engine.runtime.kv_cache import SlotPoolCache, _FullAttentionSlotLayer, _LinearAttentionSlotLayer

log = logging.getLogger(__name__)


class PrefixCache:
    """Manages a reserved slot containing the prefilled chat-template prefix.

    After ``warm_up()`` is called the object knows:
    - ``prefix_token_ids``: the token IDs of the shared prefix
    - ``prefix_len``: how many tokens the prefix spans
    - ``slot_id``: the reserved slot index holding the cached K/V

    ``apply_to_slot(dst_slot_id)`` copies the prefix K/V into another slot
    so that its prefill can start from position ``prefix_len``.
    """

    #: Slot index reserved for the shared-prefix K/V.  Must NOT appear in
    #: the scheduler's free_slots pool.  We use the last slot to avoid
    #: clashing with cuda_graphs' slot-0 dummy rows.
    RESERVED_SLOT: int  # set in __init__ from num_slots

    def __init__(self, cache: "SlotPoolCache", num_slots: int) -> None:
        self.cache = cache
        self.num_slots = num_slots
        self.RESERVED_SLOT = num_slots - 1

        # Set after warm_up():
        self.prefix_token_ids: list[int] = []
        self.prefix_len: int = 0
        self._warmed_up: bool = False

    # ------------------------------------------------------------------ #
    # startup: prefill the reserved slot once
    # ------------------------------------------------------------------ #

    def warm_up(self, runner: "object", tokenizer_encode_fn) -> None:
        """One-time prefill of the chat-template prefix into RESERVED_SLOT.

        Args:
            runner: a ModelRunner instance exposing ``.prefill()``.
            tokenizer_encode_fn: callable(messages) -> list[int].
                                 Should be ``loaded.tokenizer`` encode,
                                 called with add_special_tokens=False.
        """
        # Render the minimal shared prefix: a user turn with empty content.
        # We derive the prefix by rendering a full message and then finding
        # where the user-content boundary is.
        from engine.tokenizer.chat_template import ChatTokenizer

        # Build a minimal "system + empty user turn" prompt to discover the
        # prefix. Render twice: once with a distinctive content sentinel and
        # once without.  The tokens before the sentinel are the shared prefix.
        sentinel_text = "\x00SENTINEL\x00"
        # Render with sentinel user content to isolate the prefix.
        full_tokens = tokenizer_encode_fn(
            [{"role": "user", "content": sentinel_text}],
        )
        sentinel_tokens = tokenizer_encode_fn(
            # Encode just the sentinel text directly (no template).
            sentinel_text,
        )

        # Find where the sentinel starts in the full token sequence.
        # The prefix is everything before it; the suffix after is the user
        # content + the generation prompt.
        prefix_len = _find_prefix_end(full_tokens, sentinel_tokens)

        if prefix_len == 0:
            log.warning(
                "prefix_cache: could not isolate chat-template prefix "
                "(sentinel not found); prefix caching disabled."
            )
            self._warmed_up = False
            return

        self.prefix_token_ids = full_tokens[:prefix_len]
        self.prefix_len = prefix_len

        log.info(
            "prefix_cache: detected prefix_len=%d tokens in slot %d; "
            "running one-time prefill…",
            self.prefix_len, self.RESERVED_SLOT,
        )

        # Run prefill on the reserved slot. We borrow the runner's prefill()
        # method — it writes K/V into the cache's slot-pool tensors and
        # updates slot_lengths.  We pass sampling=None (greedy) since we
        # don't care about the output token.
        from engine.runtime.sequence import SamplingParams

        dummy_sampling = SamplingParams(max_tokens=1, temperature=0.0)
        try:
            runner.prefill(
                self.RESERVED_SLOT,
                self.prefix_token_ids,
                sampling=dummy_sampling,
            )
        except Exception as exc:
            log.exception(
                "prefix_cache: warm-up prefill failed (%s); disabling.", exc
            )
            self._warmed_up = False
            return

        self._warmed_up = True
        log.info(
            "prefix_cache: warm-up done. prefix_len=%d, reserved_slot=%d",
            self.prefix_len, self.RESERVED_SLOT,
        )

    # ------------------------------------------------------------------ #
    # per-request: copy prefix K/V to a new slot
    # ------------------------------------------------------------------ #

    @property
    def ready(self) -> bool:
        return self._warmed_up and self.prefix_len > 0

    def matches(self, prompt_token_ids: list[int]) -> bool:
        """Return True if prompt starts with the cached prefix tokens."""
        if not self.ready:
            return False
        if len(prompt_token_ids) <= self.prefix_len:
            # Prompt is shorter than or equal to prefix — nothing novel to add.
            return False
        return prompt_token_ids[: self.prefix_len] == self.prefix_token_ids

    def apply_to_slot(self, dst_slot_id: int) -> None:
        """Copy the prefix K/V from the reserved slot into dst_slot_id.

        After this call:
        - Full-attention layers: cache.layers[i].k[dst] and .v[dst] hold the
          prefix K/V for positions 0..prefix_len-1.
        - Linear-attention layers: pool_conv_states[dst] and
          pool_recurrent_states[dst] hold the recurrent state at the END of
          the prefix (so the suffix prefill continues from that state).
        - cache.slot_lengths[dst] is set to prefix_len.

        The caller (scheduler) must then prefill ONLY the suffix tokens
        (prompt[prefix_len:]) at positions prefix_len..len(prompt)-1.
        """
        src = self.RESERVED_SLOT
        dst = dst_slot_id
        plen = self.prefix_len
        cache = self.cache

        # --- Full-attention layers: copy K and V slices ---
        for layer in cache.layers:
            if _is_full_attn(layer):
                # layer.k / layer.v: [num_slots, max_seq_len, num_kv_heads, head_dim]
                layer.k[dst, :plen].copy_(layer.k[src, :plen])  # type: ignore[index]
                layer.v[dst, :plen].copy_(layer.v[src, :plen])  # type: ignore[index]

        # --- Linear-attention layers: copy recurrent + conv state ---
        for layer in cache.layers:
            if _is_linear_attn(layer):
                # _pool tensors: [num_slots, ...]
                layer._pool_conv_states[dst].copy_(layer._pool_conv_states[src])  # type: ignore[index]
                layer._pool_recurrent_states[dst].copy_(layer._pool_recurrent_states[src])  # type: ignore[index]

        # Update the slot length so subsequent decode steps know the slot
        # already has prefix_len tokens cached.
        cache.slot_lengths[dst] = plen


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _is_full_attn(layer: object) -> bool:
    return type(layer).__name__ == "_FullAttentionSlotLayer"


def _is_linear_attn(layer: object) -> bool:
    return type(layer).__name__ == "_LinearAttentionSlotLayer"


def _find_prefix_end(full_tokens: list[int], sentinel_tokens: list[int]) -> int:
    """Return the index in full_tokens where sentinel_tokens first appears.

    Scans full_tokens with a sliding window to find the sentinel sub-sequence.
    Returns 0 if not found.
    """
    slen = len(sentinel_tokens)
    if slen == 0:
        return 0
    for i in range(len(full_tokens) - slen + 1):
        if full_tokens[i : i + slen] == sentinel_tokens:
            return i
    return 0
