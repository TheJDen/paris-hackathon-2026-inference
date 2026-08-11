import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)


class SlotCache:
    def __init__(self, cfg: Qwen3_5MoeTextConfig, num_slots, max_len, device):
        layer_types = cfg.layer_types
        full = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        gdn  = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
        n_kv = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        self.k = {l: torch.empty(num_slots, max_len, n_kv, head_dim, dtype=torch.bfloat16, device=device) for l in full}
        self.v = {l: torch.empty(num_slots, max_len, n_kv, head_dim, dtype=torch.bfloat16, device=device) for l in full}
        nv = cfg.linear_num_value_heads
        hk = cfg.linear_key_head_dim
        hv = cfg.linear_value_head_dim
        self.rec = {l: torch.empty(num_slots, nv, hk, hv, dtype=torch.bfloat16, device=device) for l in gdn}
        k_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
        v_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
        conv_dim = k_dim * 2 + v_dim
        conv_width = cfg.linear_conv_kernel_dim
        hk = cfg.linear_key_head_dim
        hv = cfg.linear_value_head_dim
        self.conv = {l: torch.empty(num_slots, conv_dim, conv_width, dtype=torch.bfloat16, device=device) for l in gdn}
        self.lens = torch.zeros(num_slots, dtype=torch.int32, device=device)

    def begin(self, slots: torch.Tensor, seq_lens: torch.Tensor):
        self.lens[slots] = seq_lens

    def advance(self, slots: torch.Tensor, seq_lens: torch.Tensor):
        self.lens[slots] += seq_lens

    def read_gdn(self, layer_idx, slots):
        return self.conv[layer_idx][slots], self.rec[layer_idx][slots]

    def update_gdn(self, layer_idx, slots, conv, rec):
        self.conv[layer_idx][slots] = conv.to(self.conv[layer_idx].dtype)
        self.rec[layer_idx][slots] = rec.to(self.rec[layer_idx].dtype)
