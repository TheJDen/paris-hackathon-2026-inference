import torch
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet, apply_mask_to_padding_states

def _gdn_forward_slotted(
    self: Qwen3_5MoeGatedDeltaNet,
    hidden_states: torch.Tensor,
    cache_params: transformers.Cache | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs
):

    slotcache = kwargs["slotcache"]
    slots: torch.Tensor = kwargs["slots"]
    decoding: bool = kwargs["decoding"]

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    # Set up dimensions for reshapes later
    batch_size, seq_len, _ = hidden_states.shape

    # We have cached `conv_state` / `recurrent_state` to continue from. The two cached modes
    # (single-token decode and chunk-tokens continuation) share the state read here; they only
    # diverge in how the conv input is assembled and which kernel consumes the states below,
    # which we gate locally on `seq_len`.

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)


    if decoding:
        conv, rec = slotcache.read_gdn(self.layer_idx, slots)
        # Single-token cached decode: the fused per-step kernel updates the conv state in-place.
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        # Multi-token forward (prefill, or chunked-tokens decode when the cache has prior state).
        # Cached chunked-tokens decode: prepend the cached conv context so the causal conv
        # sees the correct left-context rather than zero-padding. Dropped from the output
        # at the end of this branch.
        conv = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=kwargs.get("seq_idx"),
            )
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [
            self.key_dim,
            self.key_dim,
            self.value_dim,
        ],
        dim=-1,
    )

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if decoding:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=rec,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            # The chunked FLA kernel takes a single `cu_seqlens` arg; for packed self-attention this matches q-side lengths.
            cu_seqlens=kwargs.get("cu_seq_lens_q"),
        )

    slotcache.update_gdn(self.layer_idx, slots, conv, last_recurrent_state)

    # reshape input data into 2D tensor
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output

def patch_gdn():
    Qwen3_5MoeGatedDeltaNet.forward = _gdn_forward_slotted
