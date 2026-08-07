"""Decode throughput vs batch size (concurrency), single GPU, one model load.

Answers the DP-vs-TP/EP question: how much does aggregate decode tok/s grow as
you concentrate more requests on one GPU? If it saturates early, 8 DP replicas at
low per-replica batch already sit near peak and beat a comm-taxed single logical
model. If it keeps climbing to 64+, concentrating batch (what TP/EP do) has
headroom -- but only if the collective comm is cheaper than the gain.

    python scripts/decode_scaling.py     # sweeps batch 1..64, vLLM, tp=1
"""
import json
import os
import sys
import time

os.environ["PATH"] = os.path.dirname(sys.executable) + ":/usr/local/cuda/bin:" + os.environ.get("PATH", "")

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-35B-A3B")
GEN = int(os.environ.get("GEN_TOKENS", "256"))
BATCHES = [1, 2, 4, 8, 16, 32, 64]


def main():
    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL, tensor_parallel_size=1, dtype="float16",
              gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.85")),
              max_model_len=2048)
    prompt = "Write a long, detailed story about a robot learning to paint."

    def gen(bs, toks):
        sp = SamplingParams(max_tokens=toks, temperature=0.0, ignore_eos=True)
        t0 = time.perf_counter()
        outs = llm.generate([prompt] * bs, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        return dt, ntok

    gen(8, 16)  # warmup: trigger CUDA-graph capture / JIT

    res = {"model": MODEL, "gen_tokens": GEN, "rows": []}
    for bs in BATCHES:
        dt, ntok = gen(bs, GEN)
        row = {"batch": bs, "agg_tok_s": round(ntok / dt, 1),
               "per_req_tok_s": round(ntok / dt / bs, 1), "gen_s": round(dt, 2)}
        res["rows"].append(row)
        print(f"batch={bs:>3} | agg={row['agg_tok_s']:>8} tok/s | per-req={row['per_req_tok_s']:>6} tok/s | {row['gen_s']}s")

    base = res["rows"][0]["agg_tok_s"]
    for r in res["rows"]:
        r["scale_vs_linear"] = round(r["agg_tok_s"] / (base * r["batch"]), 2)
    json.dump(res, open("benchmark_result.json", "w"), indent=2)
    print("RESULT:", json.dumps(res))


if __name__ == "__main__":
    main()
