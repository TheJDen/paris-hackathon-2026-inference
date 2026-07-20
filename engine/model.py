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

@dataclasses.dataclass
class ModelBundle:
    model: transformers.Qwen3_5MoeForCausalLM
    tokenizer: transformers.PreTrainedTokenizerFast
    stop_ids: set

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

def load(model_id: str = MODEL_ID) -> ModelBundle:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_dir = huggingface_hub.snapshot_download(model_id)
    tcfg = transformers.AutoConfig.from_pretrained(model_dir).get_text_config()
    arch = transformers.Qwen3_5MoeForCausalLM
    with torch.device("cuda"):
        attn_impl = "flash_attention_2" if transformers.utils.import_utils.is_flash_attn_2_available() else "sdpa" 
        print(f"attn_impl: {attn_impl}")
        model = arch._from_config(tcfg, dtype=torch.bfloat16, attn_implementation=attn_impl)
    model.eval()

    _load_weights(model, model_dir)

    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}
    return ModelBundle(model, tokenizer, stop_ids)

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
