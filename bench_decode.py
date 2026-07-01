"""vLLM offline decode-throughput benchmark.

Generates a batch of completions and reports decode tokens/sec. Writes
benchmark_result.json (collected by the cloudbench harness).

    python bench_decode.py --model Qwen/Qwen2.5-0.5B-Instruct --tokens 128 --batch-size 32

Notes
-----
* Run it with a Python that has vLLM installed (e.g. the GB10 env at
  ~/venv-vllm, or the cloud Docker image).
* It puts the venv's bin and /usr/local/cuda/bin on PATH so vLLM can find
  `ninja`/`nvcc` for its just-in-time kernel compile (needed on new GPUs like
  the GB10 / Blackwell, sm_121).
* `tensor_parallel_size` defaults to $NGPUS so the same script scales 1x -> 8x.
"""

from __future__ import annotations

import os
import sys

# vLLM JIT-compiles kernels on first run; make sure the build tools resolve.
os.environ["PATH"] = (
    os.path.dirname(sys.executable) + ":/usr/local/cuda/bin:" + os.environ.get("PATH", "")
)

import argparse  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tp", type=int, default=int(os.environ.get("NGPUS", "1")))
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float,
                    default=float(os.environ.get("GPU_MEM_UTIL", "0.6")))
    # CUDA graphs boost decode throughput but capture can be flaky on brand-new
    # GPUs. Default to ON (real perf); pass --enforce-eager to disable them.
    ap.add_argument("--enforce-eager", action="store_true",
                    default=os.environ.get("ENFORCE_EAGER", "0") == "1")
    a, _ = ap.parse_known_args()

    from vllm import LLM, SamplingParams

    t_load = time.perf_counter()
    llm = LLM(
        model=a.model,
        tensor_parallel_size=a.tp,
        dtype="float16",
        gpu_memory_utilization=a.gpu_mem_util,
        max_model_len=a.max_model_len,
        enforce_eager=a.enforce_eager,
    )
    load_s = time.perf_counter() - t_load

    prompts = ["Write a short story about a robot learning to paint."] * a.batch_size
    sp = SamplingParams(max_tokens=a.tokens, temperature=0.0, ignore_eos=True)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0

    out_toks = sum(len(o.outputs[0].token_ids) for o in outs)
    result = {
        "model": a.model,
        "tensor_parallel": a.tp,
        "cuda_graphs": not a.enforce_eager,
        "batch_size": a.batch_size,
        "max_tokens": a.tokens,
        "load_seconds": round(load_s, 2),
        "generate_seconds": round(dt, 3),
        "output_tokens": out_toks,
        "decode_tokens_per_s": round(out_toks / dt, 1),
        "requests_per_s": round(a.batch_size / dt, 2),
    }
    pathlib.Path("benchmark_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
