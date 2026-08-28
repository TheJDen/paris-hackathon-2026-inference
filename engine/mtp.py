import re
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeRMSNorm

MTP_EXPERT_RE = re.compile(r"^mtp\.layers\.0\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")  

class MTPHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = torch.nn.ModuleList([Qwen3_5MoeDecoderLayer(cfg, 3)]) # 3rd layer is full_attention in base model
        self.layers[0].self_attn.layer_idx = 0 # still only has 1 (0th) cache slot
        self.fc = torch.nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.pre_fc_norm_hidden = Qwen3_5MoeRMSNorm(cfg.hidden_size)
        self.pre_fc_norm_embedding = Qwen3_5MoeRMSNorm(cfg.hidden_size)
        self.norm = Qwen3_5MoeRMSNorm(cfg.hidden_size)

    @classmethod
    def from_tensors(cls, cfg, mtp_raw, device="cuda"):
        with torch.device(device):
            self = cls(cfg).to(torch.bfloat16)

        params = dict(self.state_dict())
        loaded = set()

        E = cfg.num_experts
        gate = torch.stack([mtp_raw[f"mtp.layers.0.mlp.experts.{i}.gate_proj.weight"] for i in range(E)])  # [E, I, H]
        up   = torch.stack([mtp_raw[f"mtp.layers.0.mlp.experts.{i}.up_proj.weight"]   for i in range(E)])  # [E, I, H]
        down = torch.stack([mtp_raw[f"mtp.layers.0.mlp.experts.{i}.down_proj.weight"] for i in range(E)])  # [E, H, I]
        params["layers.0.mlp.experts.gate_up_proj"].data.copy_(torch.cat([gate, up], dim=1))  # [E, 2I, H] gate-first
        params["layers.0.mlp.experts.down_proj"].data.copy_(down)
        loaded |= {"layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.experts.down_proj"}     

        for name, tensor in mtp_raw.items():
            if MTP_EXPERT_RE.match(name):
                continue
            key = name[len("mtp."):]
            params[key].data.copy_(tensor)     
            loaded.add(key)

        missing = set(params) - loaded
        assert not missing, f"MTP never filled: {missing}"
        return self

    def forward(self, hidden_t, token_emb, rotary_emb, cache, position_ids, cache_position):
        norm_e = self.pre_fc_norm_embedding(token_emb)
        norm_h = self.pre_fc_norm_hidden(hidden_t)
        z = self.fc(torch.cat([norm_e, norm_h], dim=-1))
        pe = rotary_emb(z, position_ids)
        h = self.layers[0](
                z,
                past_key_values=cache,
                position_embeddings=pe,
                position_ids=position_ids,
                cache_position=cache_position,
                use_cache=True
                )
        return h

