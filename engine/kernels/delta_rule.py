"""Helion rewrite of ``chunk_gated_delta_rule`` for Qwen3-Next / Qwen3.5-MoE.

Target hardware: H200 (Hopper, SM 9.0).

Replaces the ``ChunkGatedDeltaRuleFunction`` hotspot from flash-linear-attention
(~50ms in the Phase 1 microbatch profile, the 2nd largest CUDA op behind the
MoE expert matmuls). The goal for v0 is parity with fla's output at a similar
or better wall time; target is <20ms for the chunked recurrence path on the
H200 with batch=16, seq=4096, 32 v-heads, D=128, chunk=64.

API parity
----------
The public entry point :func:`chunk_gated_delta_rule` has the same signature
and semantics as ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``:

    (query, key, value, g, beta, chunk_size=64, initial_state=None,
     output_final_state=False, use_qk_l2norm_in_kernel=False)
    -> (core_attn_out, last_recurrent_state)

with Q/K/V shaped ``[B, T, H, D]`` and g/beta shaped ``[B, T, H]``. This
matches both fla and HF transformers' ``torch_chunk_gated_delta_rule``
reference implementation in ``modeling_qwen3_next.py``.

Algorithm
---------
This is a direct transcription of the HF reference
``torch_chunk_gated_delta_rule`` (which in turn mirrors fla's Triton kernel).
For each (batch, head) the sequence is split into ``C = ceil(T/chunk_size)``
chunks of ``S = chunk_size`` tokens. Per chunk we:

  1. Build a lower-triangular mixing matrix ``T[S, S]`` by forward substitution
     over ``-(k_beta @ k.T) * decay_mask`` (strictly lower-tri).
  2. Precompute ``value_new = T @ v_beta`` and ``k_cumdecay = T @ (k_beta * exp(g))``.
  3. Scan chunks serially, maintaining an fp32 state ``S[D_k, D_v]``:
        local  = softmax-free masked ``q @ k.T * decay_mask`` (strictly lower-tri)
        out    = (q * exp(g)) @ state + local @ (value_new - k_cumdecay @ state)
        state  = state * exp(g[-1]) + (k * exp(g[-1] - g)).T @ (value_new - k_cumdecay @ state)

Output is cast back to bf16; ``last_recurrent_state`` is returned in fp32
matching fla / ``mamba_ssm_dtype=float32``.

v0 strategy
-----------
Helion parallelism: one program per ``(batch * num_heads)`` pair. Inside the
program we iterate over chunks with a plain Python ``for`` loop that Helion
lowers to a sequential Triton loop. The head dimensions (128) and chunk size
(64) are small and fixed, so we use full-dim ``:`` slicing inside the kernel —
this is the documented "small fixed dim" pattern (<=128 is safe for registers).

Compile cost: the first call triggers Helion / Triton compilation, usually
5-30 seconds. Cache the compiled kernel by re-using the same module-level
``@helion.kernel`` object across calls.

Numerical risk
--------------
fp32 accumulation matches fla. Order-of-operations differences (e.g. fused
vs unfused multiplies) can cause ~1e-4 relative divergence vs fla. Downstream
sampling is deterministic, so any divergence can cause *different* token IDs
even if logits are within tolerance. Validate end-to-end perplexity on the
eval set before trusting this in the hot path.

Fallback
--------
If Helion compilation or correctness fails on the H200, set the env var
``PARIS_DISABLE_HELION_DELTA=1`` to force the fla / torch reference path;
:func:`install_delta_rule_monkeypatch` will no-op in that case.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helion import (lazy — we don't want module import to fail on non-GPU hosts)
# --------------------------------------------------------------------------- #

_HELION_AVAILABLE: Optional[bool] = None
_helion = None
_hl = None


def _try_import_helion() -> bool:
    """Import helion lazily. Returns True on success, False on failure."""
    global _HELION_AVAILABLE, _helion, _hl
    if _HELION_AVAILABLE is not None:
        return _HELION_AVAILABLE
    try:
        import helion  # noqa: WPS433 (runtime import)
        import helion.language as hl  # noqa: WPS433

        _helion = helion
        _hl = hl
        _HELION_AVAILABLE = True
    except Exception as e:  # pragma: no cover - host-dependent
        log.warning("helion import failed (%s) — delta_rule will use torch fallback", e)
        _HELION_AVAILABLE = False
    return _HELION_AVAILABLE


# --------------------------------------------------------------------------- #
# Helion kernel: per-(batch, head) chunked recurrence.
#
# Input shapes (all fp32, contiguous, pre-padded so T_pad % S == 0):
#   q, k : [BH, T_pad, Dk]
#   v    : [BH, T_pad, Dv]
#   g    : [BH, T_pad]           # already cumulative-summed per chunk upstream
#   kb   : [BH, T_pad, Dk]       # k * beta
#   vb   : [BH, T_pad, Dv]       # v * beta
#   init : [BH, Dk, Dv]          # initial state (fp32)
# Outputs:
#   out   : [BH, T_pad, Dv] (fp32, caller casts to bf16)
#   state : [BH, Dk, Dv] (fp32)
#
# Helion parallelism: hl.tile over BH only. Dk / Dv / S are compile-time fixed
# and handled with full-dim (`:`) slices.
# --------------------------------------------------------------------------- #


def _build_helion_kernel():
    """Construct (and cache) the Helion kernel object.

    Lazy so that importing this module is cheap and doesn't require a GPU.
    """
    if not _try_import_helion():
        return None

    helion = _helion
    hl = _hl

    # Hopper-targeted config. v0 uses a single conservative config; real
    # autotuning will run on the H200 at deploy time via
    # ``chunk_gated_delta_rule_kernel.autotune(example_args)``.
    hopper_config = helion.Config(
        block_sizes=[1],          # One (batch, head) per program.
        num_warps=4,
        num_stages=3,
        indexing="block_ptr",     # Safe on Hopper; tensor_descriptor needs stride align.
        pid_type="flat",
    )

    @helion.kernel(
        config=hopper_config,
        static_shapes=True,
    )
    def chunk_gated_delta_rule_kernel(
        q: torch.Tensor,       # [BH, T, Dk] fp32
        k: torch.Tensor,       # [BH, T, Dk] fp32
        v: torch.Tensor,       # [BH, T, Dv] fp32
        g: torch.Tensor,       # [BH, T]     fp32 (cumsum-per-chunk already)
        kb: torch.Tensor,      # [BH, T, Dk] fp32 (= k * beta)
        vb: torch.Tensor,      # [BH, T, Dv] fp32 (= v * beta)
        init_state: torch.Tensor,  # [BH, Dk, Dv] fp32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        BH, T, Dk = q.size()
        _, _, Dv = v.size()
        S = 64  # chunk size (must match wrapper)
        C = T // S

        out = torch.empty([BH, T, Dv], dtype=torch.float32, device=q.device)
        final_state = torch.empty([BH, Dk, Dv], dtype=torch.float32, device=q.device)

        for tile_bh in hl.tile(BH, block_size=1):
            # Per-program (b,h) state, held in registers (fp32, [Dk, Dv]).
            state = init_state[tile_bh, :, :].to(torch.float32).reshape(Dk, Dv)

            # Sequential scan over chunks. C is runtime but small (T/64); Helion
            # lowers this to a Triton for-loop rather than unrolling.
            for c in range(C):
                t0 = c * S
                # Load chunk slices. Shapes: [1, S, Dk] and [1, S, Dv], [1, S].
                q_c = q[tile_bh, t0 : t0 + S, :].reshape(S, Dk)
                k_c = k[tile_bh, t0 : t0 + S, :].reshape(S, Dk)
                v_c = v[tile_bh, t0 : t0 + S, :].reshape(S, Dv)
                g_c = g[tile_bh, t0 : t0 + S].reshape(S)
                kb_c = kb[tile_bh, t0 : t0 + S, :].reshape(S, Dk)
                vb_c = vb[tile_bh, t0 : t0 + S, :].reshape(S, Dv)

                # decay_mask[i, j] = exp(g_c[i] - g_c[j]) for i >= j, else 0.
                # Build as outer diff then tril + exp.
                gi = g_c.reshape(S, 1)
                gj = g_c.reshape(1, S)
                diff = gi - gj
                # tril mask (including diagonal)
                row_idx = torch.arange(S, device=q.device).reshape(S, 1)
                col_idx = torch.arange(S, device=q.device).reshape(1, S)
                tril_ij = (row_idx >= col_idx).to(torch.float32)
                strict_tril = (row_idx > col_idx).to(torch.float32)
                decay = torch.exp(diff) * tril_ij  # [S, S]

                # attn (pre-inversion): -(kb @ k.T) * decay, strictly lower triangular.
                kkt = kb_c @ k_c.transpose(0, 1)  # [S, S]
                A = (-(kkt * decay)) * strict_tril  # zero the diag and upper tri

                # Forward substitution: for i in 1..S-1:
                #    A[i, :i] += A[i, :i] @ A[:i, :i]
                # Implemented as an in-place row update. Helion/static_range
                # unrolls since S is fixed (=64).
                # We accumulate row-wise updates into a new matrix ``Tm``.
                Tm = A
                for i in range(1, S):
                    # row i, cols [0:i]. We compute the update then write back.
                    # row_i_new = row_i_old + row_i_old @ Tm[:i, :]  (but only cols <i)
                    # Using full-matrix ops + masking keeps shapes static.
                    row_i = Tm[i : i + 1, :]  # [1, S]
                    # Mask row to cols < i.
                    row_mask = (col_idx < i).to(torch.float32).reshape(1, S)  # [1, S]
                    row_i_masked = row_i * row_mask
                    # Sub-block mask: rows < i and cols < i.
                    sub_rows = (row_idx < i).to(torch.float32)  # [S, 1]
                    sub_mask = sub_rows * row_mask  # [S, S]
                    Tm_sub = Tm * sub_mask
                    update = row_i_masked @ Tm_sub  # [1, S], only cols<i valid
                    new_row = row_i + update * row_mask
                    # Scatter back: replace row i.
                    keep_mask = (row_idx != i).to(torch.float32)  # [S, 1]
                    Tm = Tm * keep_mask + new_row * (1.0 - keep_mask)

                # T = Tm + I
                eye = (row_idx == col_idx).to(torch.float32)
                Tm = Tm + eye

                # value_new = T @ vb
                value_new = Tm @ vb_c  # [S, Dv]
                # k_cumdecay = T @ (kb * exp(g))
                exp_g = torch.exp(g_c).reshape(S, 1)
                k_cumdecay = Tm @ (kb_c * exp_g)  # [S, Dk]

                # Chunk output:
                #   attn_local = (q @ k.T * decay), strict_lower masked zero (strictly lower? no: upper masked)
                #   In reference: masked_fill_(mask_upper_strict, 0) — keep lower tri incl diag.
                # mask = triu(ones, diagonal=1) -> we *zero* those, so keep tril incl diag.
                qk = q_c @ k_c.transpose(0, 1)
                attn_local = qk * decay  # decay already zeroes upper tri (tril mask)
                # v_prime = k_cumdecay @ state
                v_prime = k_cumdecay @ state  # [S, Dv]
                v_new = value_new - v_prime
                # attn_inter = (q * exp(g)) @ state
                attn_inter = (q_c * exp_g) @ state  # [S, Dv]
                core = attn_inter + attn_local @ v_new  # [S, Dv]
                out[tile_bh, t0 : t0 + S, :] = core.reshape(1, S, Dv)

                # State update:
                #   state = state * exp(g[-1]) + (k * exp(g[-1] - g)).T @ v_new
                g_last = g_c[S - 1 : S]  # [1]
                exp_last = torch.exp(g_last).reshape(1, 1)
                decay_k = torch.exp(g_last.reshape(1) - g_c).reshape(S, 1)  # [S, 1]
                k_scaled = k_c * decay_k  # [S, Dk]
                state = state * exp_last + k_scaled.transpose(0, 1) @ v_new

            final_state[tile_bh, :, :] = state.reshape(1, Dk, Dv)

        return out, final_state

    return chunk_gated_delta_rule_kernel


_KERNEL = None


def _get_kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _build_helion_kernel()
    return _KERNEL


# --------------------------------------------------------------------------- #
# Python wrapper matching fla's chunk_gated_delta_rule signature.
# --------------------------------------------------------------------------- #


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inv = torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return x * inv


def _torch_fallback(
    query, key, value, g, beta, chunk_size, initial_state, output_final_state,
    use_qk_l2norm_in_kernel,
):
    """Mirror of HF ``torch_chunk_gated_delta_rule`` — slow reference path."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    B, H, T, Dk = key.shape
    Dv = value.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    Tp = T + pad
    query = query * (Dk ** -0.5)

    vb = value * beta.unsqueeze(-1)
    kb = key * beta.unsqueeze(-1)
    query, key, value, kb, vb = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, kb, vb)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    upper = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((kb @ key.transpose(-1, -2)) * decay).masked_fill(upper, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ vb
    k_cumdecay = attn @ (kb * g.exp().unsqueeze(-1))
    state = (
        torch.zeros(B, H, Dk, Dv, dtype=torch.float32, device=query.device)
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    core = torch.zeros_like(value)
    upper_strict = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(0, Tp // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        a = (q_i @ k_i.transpose(-1, -2) * decay[:, :, i]).masked_fill_(upper_strict, 0)
        v_prime = k_cumdecay[:, :, i] @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
        core[:, :, i] = attn_inter + a @ v_new
        state = (
            state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        state = None
    core = core.reshape(core.shape[0], core.shape[1], -1, core.shape[-1])[:, :, :T]
    core = core.transpose(1, 2).contiguous().to(initial_dtype)
    return core, state


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **_ignored,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Drop-in replacement for ``fla.ops.gated_delta_rule.chunk_gated_delta_rule``.

    Shapes:
      query, key: [B, T, H, Dk]   (Dk = 128 for Qwen3-Next)
      value:      [B, T, H, Dv]   (Dv = 128)
      g, beta:    [B, T, H]
      initial_state: optional [B, H, Dk, Dv] (fp32)

    Returns:
      core_attn_out: [B, T, H, Dv] in the original query dtype (bf16)
      last_recurrent_state: [B, H, Dk, Dv] fp32 or None
    """
    if os.environ.get("PARIS_DISABLE_HELION_DELTA", "") == "1":
        return _torch_fallback(
            query, key, value, g, beta, chunk_size, initial_state,
            output_final_state, use_qk_l2norm_in_kernel,
        )

    kernel = _get_kernel()
    if kernel is None or chunk_size != 64:
        # v0 only autotunes for chunk_size=64 (the Qwen3-Next default).
        return _torch_fallback(
            query, key, value, g, beta, chunk_size, initial_state,
            output_final_state, use_qk_l2norm_in_kernel,
        )

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)

    # -> [B, H, T, D]
    q = query.transpose(1, 2).contiguous().to(torch.float32)
    k = key.transpose(1, 2).contiguous().to(torch.float32)
    v = value.transpose(1, 2).contiguous().to(torch.float32)
    b = beta.transpose(1, 2).contiguous().to(torch.float32)
    g_ = g.transpose(1, 2).contiguous().to(torch.float32)

    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        b = F.pad(b, (0, pad))
        g_ = F.pad(g_, (0, pad))
    Tp = T + pad

    q = q * (Dk ** -0.5)
    kb = k * b.unsqueeze(-1)
    vb = v * b.unsqueeze(-1)

    # Cumulative sum of g within each chunk (matches reference).
    g_chunks = g_.reshape(B, H, -1, chunk_size).cumsum(dim=-1).reshape(B, H, Tp)

    # Collapse (B, H) -> BH.
    BH = B * H
    q_bh = q.reshape(BH, Tp, Dk).contiguous()
    k_bh = k.reshape(BH, Tp, Dk).contiguous()
    v_bh = v.reshape(BH, Tp, Dv).contiguous()
    g_bh = g_chunks.reshape(BH, Tp).contiguous()
    kb_bh = kb.reshape(BH, Tp, Dk).contiguous()
    vb_bh = vb.reshape(BH, Tp, Dv).contiguous()
    if initial_state is None:
        init_bh = torch.zeros(BH, Dk, Dv, dtype=torch.float32, device=q.device)
    else:
        init_bh = initial_state.reshape(BH, Dk, Dv).to(torch.float32).contiguous()

    out_bh, state_bh = kernel(q_bh, k_bh, v_bh, g_bh, kb_bh, vb_bh, init_bh)

    out = out_bh.reshape(B, H, Tp, Dv)[:, :, :T, :]
    out = out.transpose(1, 2).contiguous().to(initial_dtype)
    state = state_bh.reshape(B, H, Dk, Dv) if output_final_state else None
    return out, state


# --------------------------------------------------------------------------- #
# Monkeypatch installer.
# --------------------------------------------------------------------------- #


def install_delta_rule_monkeypatch() -> bool:
    """Replace transformers' / fla's ``chunk_gated_delta_rule`` with ours.

    Must be called AFTER transformers has been imported but BEFORE the model
    is constructed (``Qwen3NextGatedDeltaNet.__init__`` captures the function
    reference at construction time).

    Returns True if the patch was applied, False if Helion is unavailable or
    the opt-out env var is set.
    """
    if os.environ.get("PARIS_DISABLE_HELION_DELTA", "") == "1":
        log.info("PARIS_DISABLE_HELION_DELTA=1 — skipping delta_rule monkeypatch")
        return False

    patched = False
    try:
        import transformers.models.qwen3_next.modeling_qwen3_next as mq
        mq.chunk_gated_delta_rule = chunk_gated_delta_rule
        # Flip the fast-path guard so existing constructors pick us up.
        mq.is_fast_path_available = True
        patched = True
        log.info("patched transformers.qwen3_next.chunk_gated_delta_rule -> helion")
    except Exception as e:
        log.warning("qwen3_next monkeypatch failed: %s", e)

    try:
        import fla.ops.gated_delta_rule as fla_mod  # type: ignore
        fla_mod.chunk_gated_delta_rule = chunk_gated_delta_rule
        log.info("patched fla.ops.gated_delta_rule.chunk_gated_delta_rule -> helion")
        patched = True
    except Exception:
        # fla may not be installed; that's fine — transformers path is enough.
        pass

    return patched
