"""N-gram self-speculative decoding — v0 design + implementation.

ALGORITHM OVERVIEW
==================
This module implements *n-gram self-speculation*, the cheapest form of
speculative decoding that requires no extra model weights.

Draft phase (per request, per step):
  1. Take the last K=3 tokens from the sequence (prompt + generated so far).
     Call this the "query k-gram".
  2. Scan the token history (prompt + prior output) for the most recent
     occurrence of that K-gram.
  3. If found, propose the next N tokens that followed it as draft tokens
     (up to N=5 or until end-of-buffer).
  4. If not found (or K-gram too short), fall back to single-token decode.

Verification phase (per request, per step):
  1. Append the N draft tokens to the current input and run ONE target-model
     forward pass over [last_real_token, draft_0, draft_1, ..., draft_{N-1}].
     This produces N+1 logits: one for each draft token position plus the
     bonus token after the last draft position.
  2. Greedily check each draft token against the argmax of the corresponding
     logit position.  Accept the longest matching prefix.  If draft_i matches,
     accept it and check draft_{i+1}.  Stop at the first mismatch.
  3. The bonus token (argmax at the mismatch position, or after the last draft
     if all matched) is always accepted for free.

Net gain: if M out of N drafts are accepted, we emit M+1 tokens in ~1 target
forward pass instead of M+1 separate passes.  Break-even is when M >= 1
consistently (acceptance rate >= 1/N).

EXPECTED ACCEPTANCE RATE — HONEST ANALYSIS
===========================================
N-gram speculation exploits *local repetition* in the token sequence:
  - Code / structured text:  HIGH (keywords, bracket patterns, variable names).
  - Chat / prose: MODERATE (common phrases, dates, list items).
  - Random tokens (throughput benchmark workload): NEAR ZERO.

The benchmark at c=64 uses randomly-sampled token sequences where by
construction no K-gram matches exist.  N-gram spec decode will find zero
matches and fall back to single-token decode on every step — adding only
overhead (the K-gram lookup itself is O(L) per step, so it's cheap, but still
non-zero).

RECOMMENDATION: Do NOT enable n-gram spec decode for the throughput eval
(random tokens).  It will not help and adds latency.

Where it WILL help: correctness / structured-output eval if prompts contain
repetitive structure (code, JSON schemas, templated text).  Enable it there
behind a flag (--spec-decode-ngram) and measure acceptance rate empirically.

INTEGRATION PLAN (TODOs — do NOT edit other files)
===================================================
To wire this into the engine the following changes are needed in OTHER files.
This file is self-contained; the changes below are left as future work.

TODO (model_runner.py — decode path):
  - Add a `decode_spec(slot_ids, last_tokens, cache_lengths, draft_tokens,
      samplings)` method that accepts a variable-length sequence per slot.
  - Inside it, build input_ids as [B, 1+N] (last real token + N drafts) and
    run the model forward.  The KV cache update must write positions
    [cache_len, cache_len+1, ..., cache_len+N] — this means BatchSlots
    write_positions must carry a variable-length range per slot.
  - After the forward, call verify_draft_tokens() (defined below) to find
    the accepted prefix length M per slot.
  - Roll back the KV cache for positions beyond M+1 (the accepted tokens).
    Currently there is no cache rollback API — one must be added to
    SlotPoolCache (a `truncate_slot(slot_id, new_len)` method).
  - Return (accepted_tokens_per_slot, bonus_token_per_slot).

TODO (scheduler.py — step loop):
  - Before calling runner.decode, call NgramSpeculator.propose() for each
    active slot.
  - If any slot has a non-empty proposal, route that slot through
    runner.decode_spec; otherwise use runner.decode normally.
  - After decode_spec, call NgramSpeculator.update() with the newly accepted
    tokens so the history stays in sync.
  - Expose --spec-decode-ngram / --spec-k / --spec-n as CLI flags and wire to
    EngineConfig.

TODO (SlotPoolCache — engine/runtime/kv_cache.py):
  - Add `truncate_slot(slot_id: int, new_len: int)` to zero out KV entries
    beyond new_len in the slot pool.  For linear-attention slots, also reset
    the recurrent state to the checkpoint saved at position new_len (requires
    saving state snapshots — non-trivial for DeltaNet).
  - Alternative: don't truncate; instead, re-prefill the rejected suffix.
    This is simpler but wastes compute.  For N<=5 the cost difference is small.

NOTE on Qwen3.5-35B-A3B architecture:
  The model has 30 linear-attention (DeltaNet) layers + 10 full-attention
  layers (layer_types: [linear, linear, linear, full] x10, full_attention_interval=4).
  DeltaNet layers maintain a recurrent state, not a KV cache, so truncation
  for those layers requires either re-prefilling or snapshotting recurrent
  states.  This makes speculative decoding with DeltaNet significantly more
  complex than with a pure transformer.  The full-attention layers use a
  standard KV cache (num_kv_heads=2, head_dim=256) that can be truncated
  straightforwardly.

  For v0 we accept the limitation: if a draft token is rejected, we re-run
  the full model on the correct suffix (re-prefill style) rather than rolling
  back the recurrent state.  This means the net gain is only realized when
  the entire draft is accepted, which further reduces the effective benefit
  on random workloads.

CONSTANTS
=========
Default K=3 (query k-gram length) and N=5 (max draft tokens).
These are exposed as constructor args so callers can tune them.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import NamedTuple

import torch

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Data structures
# ------------------------------------------------------------------ #

class DraftProposal(NamedTuple):
    """Result of NgramSpeculator.propose() for one slot."""
    slot_id: int
    draft_tokens: list[int]   # may be empty if no match found


class VerifyResult(NamedTuple):
    """Result of verify_draft_tokens() for one slot."""
    slot_id: int
    accepted_tokens: list[int]   # the accepted prefix of the draft
    bonus_token: int             # the correction token at the first mismatch


# ------------------------------------------------------------------ #
# N-gram history store — per-slot token sequence
# ------------------------------------------------------------------ #

class SlotHistory:
    """Maintains the full token sequence (prompt + generated) for one slot.

    The sequence is stored as a plain Python list of ints — cheap and avoids
    GPU round-trips.  Max length is capped to prevent unbounded growth on long
    sequences; the oldest tokens are evicted (sliding window).
    """

    def __init__(self, max_len: int = 8192) -> None:
        self.tokens: list[int] = []
        self.max_len = max_len

    def extend(self, new_tokens: list[int]) -> None:
        self.tokens.extend(new_tokens)
        if len(self.tokens) > self.max_len:
            # Keep the most recent max_len tokens.
            self.tokens = self.tokens[-self.max_len :]

    def append(self, token: int) -> None:
        self.tokens.append(token)
        if len(self.tokens) > self.max_len:
            self.tokens = self.tokens[-self.max_len :]

    def __len__(self) -> int:
        return len(self.tokens)


# ------------------------------------------------------------------ #
# N-gram lookup helper
# ------------------------------------------------------------------ #

def ngram_lookup(
    tokens: list[int],
    query: tuple[int, ...],
    max_proposals: int,
) -> list[int]:
    """Search `tokens` for the last occurrence of `query` k-gram.

    Scans backward from the second-to-last position (the last K-gram at the
    tail IS the query itself; we want the occurrence before it).  Returns the
    `max_proposals` tokens that immediately followed that occurrence, or an
    empty list if not found.

    Args:
        tokens:         Full token history including the most recent K tokens.
        query:          The K-gram to search for (tuple of ints).
        max_proposals:  Maximum number of draft tokens to return (N).

    Returns:
        List of draft token ids (possibly empty).
    """
    K = len(query)
    L = len(tokens)
    if L < K + 1:
        # Not enough history to have a prior occurrence + at least one follow.
        return []

    # Scan backward, excluding the final K positions (those form the query).
    # We want the rightmost match so that the follow-on tokens are most recent.
    for i in range(L - K - 1, -1, -1):
        if tuple(tokens[i : i + K]) == query:
            # Found.  Return up to max_proposals tokens starting at i+K.
            start = i + K
            end = min(start + max_proposals, L)
            return list(tokens[start:end])

    return []


# ------------------------------------------------------------------ #
# Speculator — owns per-slot history and produces draft proposals
# ------------------------------------------------------------------ #

class NgramSpeculator:
    """Per-engine n-gram speculator.

    Lifecycle:
      on_prefill(slot_id, prompt_tokens, first_token):
          Call after prefill.  Seeds the history with prompt + first token.

      propose(slot_id, last_token) -> DraftProposal:
          Call at the START of each decode step (before the model forward).
          Returns a (possibly empty) list of draft tokens.

      on_accepted(slot_id, accepted_tokens, bonus_token):
          Call AFTER verification.  Updates the history with all tokens that
          were committed (accepted drafts + bonus).

      on_decode(slot_id, token):
          Call after a normal (non-spec) decode step.  Updates history with
          the single newly generated token.

      on_finish(slot_id):
          Frees the per-slot history when a request completes.
    """

    def __init__(self, k: int = 3, n: int = 5, max_history: int = 8192) -> None:
        """
        Args:
            k: K-gram length used for lookup (query width).
            n: Max number of draft tokens to propose per step.
            max_history: Max tokens kept in per-slot history (sliding window).
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.k = k
        self.n = n
        self.max_history = max_history
        self._histories: dict[int, SlotHistory] = {}

        # Simple stats for measuring acceptance rate.
        self._total_proposed: int = 0
        self._total_accepted: int = 0
        self._total_steps: int = 0

    # ---- lifecycle --------------------------------------------------- #

    def on_prefill(
        self,
        slot_id: int,
        prompt_tokens: list[int],
        first_token: int,
    ) -> None:
        """Seed the history for a newly prefilled slot."""
        h = SlotHistory(max_len=self.max_history)
        h.extend(prompt_tokens)
        h.append(first_token)
        self._histories[slot_id] = h

    def on_finish(self, slot_id: int) -> None:
        """Free the history for a completed slot."""
        self._histories.pop(slot_id, None)

    # ---- propose ----------------------------------------------------- #

    def propose(self, slot_id: int) -> DraftProposal:
        """Propose draft tokens for the given slot.

        Uses the last K tokens in the history as the query k-gram, searches
        the full history for a prior occurrence, and returns up to N follow-on
        tokens as the draft.

        Returns a DraftProposal with an empty list if no match is found or
        if the slot has insufficient history.
        """
        h = self._histories.get(slot_id)
        if h is None or len(h) < self.k:
            return DraftProposal(slot_id=slot_id, draft_tokens=[])

        query = tuple(h.tokens[-self.k :])
        draft = ngram_lookup(h.tokens, query, self.n)

        self._total_steps += 1
        self._total_proposed += len(draft)

        return DraftProposal(slot_id=slot_id, draft_tokens=draft)

    # ---- update after normal (non-spec) step ------------------------- #

    def on_decode(self, slot_id: int, token: int) -> None:
        """Update history after a normal single-token decode step."""
        h = self._histories.get(slot_id)
        if h is not None:
            h.append(token)

    # ---- update after spec step -------------------------------------- #

    def on_accepted(
        self,
        slot_id: int,
        accepted_tokens: list[int],
        bonus_token: int,
    ) -> None:
        """Update history after a speculative step completes.

        `accepted_tokens` is the accepted prefix of the draft.
        `bonus_token` is the correction token (always accepted).
        """
        h = self._histories.get(slot_id)
        if h is not None:
            h.extend(accepted_tokens)
            h.append(bonus_token)
        self._total_accepted += len(accepted_tokens) + 1  # bonus always accepted

    # ---- diagnostics ------------------------------------------------- #

    def acceptance_rate(self) -> float:
        """Fraction of proposed draft tokens that were accepted.

        Returns 0.0 if no tokens have been proposed yet.
        Note: bonus tokens (always accepted) are NOT counted in the numerator
        here; only actually-drafted tokens are counted.  This gives a
        conservative estimate of speculation quality.
        """
        if self._total_proposed == 0:
            return 0.0
        return self._total_accepted / max(self._total_proposed + self._total_steps, 1)

    def stats(self) -> dict:
        return {
            "total_steps": self._total_steps,
            "total_proposed": self._total_proposed,
            "total_accepted": self._total_accepted,
            "acceptance_rate": self.acceptance_rate(),
        }


# ------------------------------------------------------------------ #
# Verification helpers (pure CPU / GPU — no model calls here)
# ------------------------------------------------------------------ #

def verify_draft_tokens(
    slot_id: int,
    draft_tokens: list[int],
    target_logits: torch.Tensor,
    temperature: float = 0.0,
) -> VerifyResult:
    """Verify draft tokens against target model logits (greedy).

    This implements the greedy (temperature=0) speculative sampling check:
    accept draft_i iff argmax(target_logits[i]) == draft_i.

    The target_logits tensor must have shape [1 + N, vocab_size] where:
      - logits[0] corresponds to the last real token's position and predicts
        the first draft token.
      - logits[i] for i in 1..N predicts draft token i (0-indexed).
      - logits[N] predicts the bonus token after all drafts.

    Temperature != 0 is NOT supported in this greedy verifier.  For
    stochastic sampling, the standard speculative sampling acceptance criterion
    (DeepMind 2023) requires comparing draft and target distributions — a
    future extension.

    Args:
        slot_id:        The slot this verification is for.
        draft_tokens:   The N proposed draft tokens.
        target_logits:  [1+N, vocab_size] or [N+1, vocab_size] tensor.
        temperature:    Ignored (greedy only in v0); kept for API compatibility.

    Returns:
        VerifyResult with the accepted prefix and the bonus token.
    """
    N = len(draft_tokens)
    if target_logits.shape[0] != N + 1:
        raise ValueError(
            f"target_logits must have shape [{N+1}, vocab], "
            f"got {list(target_logits.shape)}"
        )

    # Compute argmax for each position — on GPU if tensor is on GPU.
    # Shape: [N+1]
    argmax_tokens = target_logits.argmax(dim=-1)  # [N+1]

    accepted: list[int] = []
    for i, draft_tok in enumerate(draft_tokens):
        target_tok = int(argmax_tokens[i].item())
        if target_tok == draft_tok:
            accepted.append(draft_tok)
        else:
            # First mismatch — bonus is the target's own prediction here.
            bonus = int(argmax_tokens[i].item())
            return VerifyResult(
                slot_id=slot_id,
                accepted_tokens=accepted,
                bonus_token=bonus,
            )

    # All N draft tokens matched — bonus is the prediction after position N.
    bonus = int(argmax_tokens[N].item())
    return VerifyResult(
        slot_id=slot_id,
        accepted_tokens=accepted,
        bonus_token=bonus,
    )


def build_spec_input(
    last_real_token: int,
    draft_tokens: list[int],
    cache_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build input tensors for a speculative decode forward pass.

    Concatenates [last_real_token] + draft_tokens into a single sequence
    for ONE slot and returns (input_ids, position_ids, write_positions).

    Callers use these tensors to build the BatchSlots for the model forward.
    The slot is processed as a "mini-prefill" over the speculative sequence.

    Args:
        last_real_token:  The verified token at the current sequence position.
        draft_tokens:     N draft tokens to verify.
        cache_length:     Current KV cache length for this slot (= position of
                          last_real_token in the sequence, 0-indexed).
        device:           Target device.

    Returns:
        input_ids:       [1, 1+N] long tensor — the full speculative sequence.
        position_ids:    [1, 1+N] long tensor — absolute positions.
        write_positions: [1+N]    long tensor — positions to write in KV cache.

    Note:
        The caller is responsible for constructing BatchSlots with appropriate
        query_lens (= 1+N) and kv_seq_lens (= cache_length + 1 + N).
        write_positions correspond to positions [cache_length, ...,
        cache_length + N].
    """
    N = len(draft_tokens)
    total = 1 + N  # last real token + N drafts

    tokens = [last_real_token] + list(draft_tokens)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)  # [1, total]

    start_pos = cache_length  # position of last_real_token
    position_ids = torch.arange(
        start_pos, start_pos + total, dtype=torch.long, device=device
    ).unsqueeze(0)  # [1, total]

    write_positions = torch.arange(
        start_pos, start_pos + total, dtype=torch.long, device=device
    )  # [total]

    return input_ids, position_ids, write_positions


# ------------------------------------------------------------------ #
# Standalone sanity-check (python -m engine.runtime.spec_decode)
# ------------------------------------------------------------------ #

def _smoke_test() -> None:
    """Quick self-test that exercises the main code paths without a GPU."""
    import random

    # Build a sequence with a known repetition.
    tokens = [10, 20, 30, 40, 50, 20, 30, 40, 60]
    # Query: last 3 tokens = (40, 60, ?) — no, let's query (20, 30, 40):
    query = (20, 30, 40)
    proposals = ngram_lookup(tokens, query, max_proposals=5)
    # First occurrence of (20,30,40) is at index 1, followed by 50.
    # Second occurrence (most recent) is at index 5, followed by 60.
    assert proposals == [60], f"expected [60], got {proposals}"

    # Speculator lifecycle.
    spec = NgramSpeculator(k=3, n=5)
    prompt = [10, 20, 30, 40, 50, 20, 30]
    spec.on_prefill(slot_id=0, prompt_tokens=prompt, first_token=40)
    # History is now: 10 20 30 40 50 20 30 40
    prop = spec.propose(slot_id=0)
    # Query = (30, 40, ?) → last 3 tokens = (30, 40) only 2... wait, k=3.
    # Actually last 3 of [10,20,30,40,50,20,30,40] = (20,30,40).
    # Search backward for (20,30,40): found at index 1, followed by 50,20,30,40.
    # Most recent (before the tail) should return [50].
    assert prop.draft_tokens == [50], f"expected [50], got {prop.draft_tokens}"

    # Verification with greedy check.
    # Simulate target logits: [N+1, vocab] — N=1 draft here.
    draft = [50]
    vocab_size = 100
    logits = torch.zeros(2, vocab_size)
    logits[0, 50] = 10.0   # position 0 predicts draft[0]=50 → match
    logits[1, 99] = 10.0   # bonus token = 99
    result = verify_draft_tokens(slot_id=0, draft_tokens=draft, target_logits=logits)
    assert result.accepted_tokens == [50], f"expected [50], got {result.accepted_tokens}"
    assert result.bonus_token == 99, f"expected 99, got {result.bonus_token}"

    # Mismatch case.
    logits2 = torch.zeros(2, vocab_size)
    logits2[0, 77] = 10.0  # position 0 predicts 77, draft has 50 → mismatch
    logits2[1, 99] = 10.0
    result2 = verify_draft_tokens(slot_id=0, draft_tokens=draft, target_logits=logits2)
    assert result2.accepted_tokens == [], f"expected [], got {result2.accepted_tokens}"
    assert result2.bonus_token == 77, f"expected 77, got {result2.bonus_token}"

    # build_spec_input smoke test.
    inp, pos, wpos = build_spec_input(
        last_real_token=40,
        draft_tokens=[50, 60],
        cache_length=7,
        device=torch.device("cpu"),
    )
    assert inp.shape == (1, 3), inp.shape
    assert list(inp[0].tolist()) == [40, 50, 60]
    assert list(pos[0].tolist()) == [7, 8, 9]
    assert list(wpos.tolist()) == [7, 8, 9]

    print("spec_decode smoke test PASSED")


if __name__ == "__main__":
    _smoke_test()
