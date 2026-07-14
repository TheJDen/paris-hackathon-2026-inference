import dataclasses
import huggingface_hub
import json
import os
import safetensors
import torch
import tqdm
import transformers

@dataclasses.dataclass
class CompletionRequest:
    messages: list[dict]
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0

@dataclasses.dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"

def sample_next(logits, temperature, top_p):
    temp = temperature.view(-1, 1)
    greedy = temp == 0
    safe = torch.where(greedy, torch.ones_like(temp), temp)
    scaled = logits / safe

    s_logits, s_idx = torch.sort(scaled, descending=True, dim=-1)
    cum = torch.softmax(s_logits, dim=-1).cumsum(dim=-1)
    remove = cum > top_p.view(-1, 1)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    s_logits = s_logits.masked_fill(remove, float("-inf"))
    scaled = torch.full_like(scaled, float("-inf")).scatter(-1, s_idx, s_logits)

    probs = torch.softmax(scaled, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    greedy_tok = torch.argmax(logits, dim=-1, keepdim=True)
    return torch.where(greedy, greedy_tok, sampled)

def load(model_id: str = MODEL_ID):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_dir = huggingface_hub.snapshot_download(model_id)
    cfg = transformers.AutoConfig.from_pretrained(model_dir)
    arch = getattr(transformers, cfg.architectures[0])   # Qwen3_5MoeForConditionalGeneration
    with torch.device("cuda"):
        attn_impl = "flash_attention_2" if transformers.utils.import_utils.is_flash_attn_2_available() else "sdpa" 
        print(f"attn_impl: {attn_impl}")
        model = arch._from_config(cfg, dtype=torch.bfloat16, attn_implementation=attn_impl)
    model.eval()

    _load_weights(model, model_dir)

    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    return model, tokenizer, stop_ids

def _load_weights(model, model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    weight_map = json.load(open(idx))["weight_map"]
    shards = sorted(set(weight_map.values()))

    gpu_tensors = dict(model.state_dict())
    loaded_names = set()
    with tqdm.tqdm(total=len(gpu_tensors), desc="Loading weights", unit="tensor") as pbar:
        for shard in shards:
            with safetensors.safe_open(os.path.join(model_dir, shard), framework="pt", device="cuda") as f:
                for name in f.keys():
                    if name not in gpu_tensors:
                        continue
                    gpu_tensors[name].data.copy_(f.get_tensor(name))
                    loaded_names.add(name)
                    pbar.update(1)
            pbar.set_postfix_str(shard[-25:])
    missing_names = set(gpu_tensors) - loaded_names 
    assert not missing_names, f"never filled {missing_names}"
