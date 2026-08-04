import huggingface_hub
import json
import os
import safetensors
import torch
import tqdm
import transformers
from transformers.models import Qwen3_5MoeForCausalLM

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"

def load(model_id: str = MODEL_ID) -> Qwen3_5MoeForCausalLM:
    model_dir = huggingface_hub.snapshot_download(model_id)
    tcfg = transformers.AutoConfig.from_pretrained(model_dir).get_text_config()
    arch = transformers.Qwen3_5MoeForCausalLM
    with torch.device("cuda"):
        attn_impl = "flash_attention_2" if transformers.utils.import_utils.is_flash_attn_2_available() else "sdpa" 
        print(f"attn_impl: {attn_impl}")
        model = arch._from_config(tcfg, dtype=torch.bfloat16, attn_implementation=attn_impl)
    model.eval()

    _load_weights(model, model_dir)

    return model

def _load_weights(model, model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    weight_map = json.load(open(idx))["weight_map"]
    shards = sorted(set(weight_map.values()))

    main_tensors = dict(model.state_dict())
    loaded_main = set()
    mtp_tensors = {}
    for shard in tqdm.tqdm(shards, desc="Loading shards..."):
        with safetensors.safe_open(os.path.join(model_dir, shard), framework="pt", device="cuda") as f:
            for name in f.keys():
                main_key = _main_key(name)
                if main_key is not None:
                    main_tensors[main_key].data.copy_(f.get_tensor(name))
                    loaded_main.add(main_key)
    missing_main = set(main_tensors) - loaded_main
    assert not missing_main, f"never filled {missing_main}"
    return mtp_tensors

def _main_key(name: str):
    if name == "lm_head.weight":
        return name
    elif name.startswith("model.language_model."):
        return name.replace(".language_model", "")
    return None
