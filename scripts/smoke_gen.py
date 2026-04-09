"""Single-GPU smoke generate.

Loads Qwen3.5-35B-A3B (text branch) on a single H200, runs a tiny chat
completion, and prints the result. This is the first end-to-end check
that the model + tokenizer + chat-template + thinking-disabled flow all
work *before* we touch the server, batching, or TP.

Usage:
    python scripts/smoke_gen.py
    python scripts/smoke_gen.py --prompt "What is the capital of France?"
    python scripts/smoke_gen.py --device cuda:0 --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import logging
import time

import torch

from engine.model.qwen3_next import (
    MODEL_ID,
    load_model,
    num_full_attention_layers,
    num_linear_attention_layers,
)
from engine.tokenizer.chat_template import THINK_OPEN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prompt", default="What is 2+2? Answer with just the number.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    print(f"loading {args.model} on {args.device}...")
    t0 = time.perf_counter()
    loaded = load_model(args.model, device=args.device, attn_impl=args.attn_impl)
    dt = time.perf_counter() - t0
    n_full = num_full_attention_layers(loaded.text_config)
    n_lin = num_linear_attention_layers(loaded.text_config)
    print(f"loaded in {dt:.1f}s — {n_full} full-attn + {n_lin} linear-attn layers")

    mem = torch.cuda.memory_allocated(loaded.device) / 1024**3
    print(f"GPU memory after load: {mem:.1f} GB")

    # Render the chat-templated prompt with thinking disabled.
    messages = [{"role": "user", "content": args.prompt}]
    try:
        text = loaded.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = loaded.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    print("--- rendered prompt ---")
    print(text)
    print("--- end prompt ---")
    if THINK_OPEN in text:
        print(f"WARNING: rendered prompt contains {THINK_OPEN!r} — chat template did not honor enable_thinking=False")

    inputs = loaded.tokenizer(text, return_tensors="pt").to(loaded.device)
    prompt_len = inputs["input_ids"].shape[-1]
    print(f"prompt tokens: {prompt_len}")

    print("generating...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = loaded.model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,  # ignored when do_sample=False but transformers complains otherwise
            top_p=1.0,
            pad_token_id=loaded.tokenizer.eos_token_id,
        )
    dt = time.perf_counter() - t0
    new_tokens = out.shape[-1] - prompt_len
    text_out = loaded.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    tps = new_tokens / dt if dt > 0 else 0.0
    print(f"--- generated ({new_tokens} tokens, {dt:.2f}s, {tps:.1f} tok/s) ---")
    print(text_out)
    print("--- end ---")
    if THINK_OPEN in text_out:
        print(f"WARNING: generated output contains {THINK_OPEN!r}")


if __name__ == "__main__":
    main()
