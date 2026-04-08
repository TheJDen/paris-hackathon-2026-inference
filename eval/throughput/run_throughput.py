"""Throughput benchmark for inference engine implementations.

Measures output tokens/sec at various concurrency levels using fixed
1024-token input / 1024-token output workloads against an OpenAI-compatible
chat completions endpoint.

Prompts are generated at runtime using random token IDs from the model
vocabulary (excluding special tokens), matching vLLM's RandomDataset approach.

Anti-gaming measures:
- Token counts are verified by re-tokenizing responses with the Qwen tokenizer
- All prompts are generated at runtime (nothing pre-computed)
- Correctness spot-checks are embedded among random prompts

Usage:
    python -m eval.throughput.run_throughput --base-url http://localhost:8000
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time

import aiohttp
import numpy as np
from tabulate import tabulate
from transformers import AutoTokenizer

DEFAULT_CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64]
DEFAULT_NUM_REQUESTS = 64
DEFAULT_NUM_PROMPTS = 128
WARMUP_REQUESTS = 2
INPUT_TOKENS = 1024
OUTPUT_TOKENS = 1024
MODEL_ID = "Qwen/Qwen3.5-35B-A3B"

# Spot-check questions with verifiable answers.
# Kept simple to avoid false failures from sampling variance.
SPOT_CHECKS = [
    {
        "content": "What is 7 * 8? Only output the number, nothing else.",
        "answer": 56,
    },
    {
        "content": "What is 100 + 250 + 50? Only output the number, nothing else.",
        "answer": 400,
    },
    {
        "content": "What is 1000 - 337? Only output the number, nothing else.",
        "answer": 663,
    },
    {
        "content": "What is 15 * 20? Only output the number, nothing else.",
        "answer": 300,
    },
    {
        "content": "What is 144 / 12? Only output the number, nothing else.",
        "answer": 12,
    },
    {
        "content": "What is 99 + 1? Only output the number, nothing else.",
        "answer": 100,
    },
    {
        "content": "What is 25 * 4? Only output the number, nothing else.",
        "answer": 100,
    },
    {
        "content": "What is 500 - 123? Only output the number, nothing else.",
        "answer": 377,
    },
]


def generate_prompts(tokenizer, count: int, target_tokens: int) -> list[str]:
    """Generate random-vocab prompts at runtime.

    Uses allowed (non-special) tokens with sequential pattern through the
    vocabulary, matching vLLM's RandomDataset approach. Iterative decode-encode
    adjustment ensures exact target token length.
    """
    rng = np.random.default_rng()  # system entropy, not reproducible
    all_tokens = np.arange(tokenizer.vocab_size)
    prohibited = set(tokenizer.all_special_ids)
    allowed_tokens = np.array([t for t in all_tokens if t not in prohibited])

    prompts = []
    for i in range(count):
        offset = int(rng.integers(0, len(allowed_tokens)))
        token_ids = allowed_tokens[
            (offset + i + np.arange(target_tokens)) % len(allowed_tokens)
        ].tolist()
        # Iterative decode-encode adjustment for exact length
        for _retry in range(10):
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) == target_tokens:
                break
            elif len(token_ids) < target_tokens:
                extra = allowed_tokens[
                    rng.integers(0, len(allowed_tokens), size=target_tokens - len(token_ids))
                ].tolist()
                token_ids.extend(extra)
            else:
                token_ids = token_ids[:target_tokens]
        prompts.append(tokenizer.decode(token_ids, skip_special_tokens=True))
    return prompts


def verify_token_count(tokenizer, text: str) -> int:
    """Count tokens using the actual tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def check_spot_answer(response_text: str, expected: float) -> bool:
    """Check if the response contains the expected numeric answer."""
    numbers = re.findall(r'[\d,]+\.?\d*', response_text.replace(",", ""))
    for num_str in numbers:
        try:
            val = float(num_str)
            if abs(val - expected) < 0.01:
                return True
        except ValueError:
            continue
    return False


def count_prompt_tokens(tokenizer, messages: list[dict]) -> int:
    """Count prompt tokens by applying the chat template locally."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    tokenizer,
    spot_check_answer: float | None = None,
) -> dict:
    """Send a single chat completion request and measure timing."""
    # Use temperature=0 for spot-checks (deterministic), 1.0 for throughput prompts
    temp = 0.0 if spot_check_answer is not None else 1.0
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
    }

    # Count prompt tokens locally via chat template
    verified_prompt = count_prompt_tokens(tokenizer, messages)

    async with semaphore:
        t_start = time.perf_counter()
        try:
            async with session.post(url, json=payload) as resp:
                body = await resp.json()
                t_end = time.perf_counter()

                if resp.status != 200:
                    return {"success": False, "error": f"HTTP {resp.status}", "latency": t_end - t_start}

                usage = body.get("usage", {})
                reported_prompt = usage.get("prompt_tokens", 0)
                reported_completion = usage.get("completion_tokens", 0)
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                verified_completion = verify_token_count(tokenizer, content)

                result = {
                    "success": True,
                    "latency": t_end - t_start,
                    "verified_tokens": verified_prompt + verified_completion,
                    "verified_prompt_tokens": verified_prompt,
                    "verified_completion_tokens": verified_completion,
                    "reported_prompt_tokens": reported_prompt,
                    "reported_completion_tokens": reported_completion,
                }

                if spot_check_answer is not None:
                    result["spot_check_pass"] = check_spot_answer(content, spot_check_answer)
                    result["spot_check_response"] = content[:200]

                return result
        except Exception as e:
            t_end = time.perf_counter()
            return {"success": False, "error": str(e), "latency": t_end - t_start}


async def run_concurrency_level(
    session: aiohttp.ClientSession,
    url: str,
    prompts: list[str],
    concurrency: int,
    num_requests: int,
    max_tokens: int,
    tokenizer,
    spot_checks: list[dict],
) -> dict:
    """Run benchmark at a single concurrency level."""
    semaphore = asyncio.Semaphore(concurrency)

    # Warmup
    warmup_tasks = [
        send_request(session, url, prompts[i % len(prompts)], max_tokens, semaphore, tokenizer)
        for i in range(WARMUP_REQUESTS)
    ]
    await asyncio.gather(*warmup_tasks)

    # Build request list: mostly random prompts, with spot-checks injected
    request_args = []
    spot_indices = set()
    if spot_checks:
        spot_indices = set(random.sample(range(num_requests), min(len(spot_checks), num_requests)))

    spot_iter = iter(spot_checks)
    prompt_idx = 0
    for i in range(num_requests):
        if i in spot_indices:
            sc = next(spot_iter, None)
            if sc:
                request_args.append((sc["content"], max_tokens, sc["answer"]))
                continue
        request_args.append((prompts[prompt_idx % len(prompts)], max_tokens, None))
        prompt_idx += 1

    # Benchmark
    wall_start = time.perf_counter()
    tasks = [
        send_request(session, url, prompt, mt, semaphore, tokenizer, spot_answer)
        for prompt, mt, spot_answer in request_args
    ]
    results = await asyncio.gather(*tasks)
    wall_end = time.perf_counter()

    wall_time = wall_end - wall_start
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        return {
            "concurrency": concurrency,
            "throughput_tok_per_sec": 0,
            "successful_requests": 0,
            "failed_requests": len(failed),
            "wall_time_sec": wall_time,
            "spot_checks_passed": 0,
            "spot_checks_total": 0,
        }

    total_verified = sum(r["verified_tokens"] for r in successful)
    total_prompt = sum(r["verified_prompt_tokens"] for r in successful)
    total_completion = sum(r["verified_completion_tokens"] for r in successful)
    total_reported_prompt = sum(r["reported_prompt_tokens"] for r in successful)
    total_reported_completion = sum(r["reported_completion_tokens"] for r in successful)
    throughput = total_verified / wall_time

    # Check for token count discrepancies (>5% difference)
    prompt_discrepancy = abs(total_reported_prompt - total_prompt) > max(total_prompt * 0.05, 10)
    completion_discrepancy = abs(total_reported_completion - total_completion) > max(total_completion * 0.05, 10)

    # Spot-check results
    spot_results = [r for r in successful if "spot_check_pass" in r]
    spot_passed = sum(1 for r in spot_results if r["spot_check_pass"])

    return {
        "concurrency": concurrency,
        "throughput_tok_per_sec": round(throughput, 2),
        "total_tokens": total_verified,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "reported_prompt_tokens": total_reported_prompt,
        "reported_completion_tokens": total_reported_completion,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "wall_time_sec": round(wall_time, 2),
        "token_discrepancy": prompt_discrepancy or completion_discrepancy,
        "spot_checks_passed": spot_passed,
        "spot_checks_total": len(spot_results),
    }


async def run_benchmark(
    base_url: str,
    prompts: list[str],
    concurrency_levels: list[int],
    num_requests: int,
    max_tokens: int,
    tokenizer,
) -> list[dict]:
    """Run the full throughput benchmark across all concurrency levels."""
    url = f"{base_url}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=600)

    spot_pool = SPOT_CHECKS.copy()
    random.shuffle(spot_pool)

    results = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, concurrency in enumerate(concurrency_levels):
            sc_start = (i * 2) % len(spot_pool)
            level_spots = [spot_pool[(sc_start + j) % len(spot_pool)] for j in range(2)]

            print(f"  Concurrency={concurrency} ({num_requests} requests)...", end=" ", flush=True)
            result = await run_concurrency_level(
                session, url, prompts, concurrency, num_requests, max_tokens,
                tokenizer, level_spots,
            )

            # Status line
            parts = [
                f"{result['throughput_tok_per_sec']} tok/s",
                f"({result['successful_requests']}/{num_requests} ok, {result['wall_time_sec']}s)",
            ]
            if result["token_discrepancy"]:
                parts.append("[WARN: token count mismatch]")
            if result["spot_checks_total"] > 0:
                sc_status = f"spot={result['spot_checks_passed']}/{result['spot_checks_total']}"
                if result["spot_checks_passed"] < result["spot_checks_total"]:
                    sc_status = f"[WARN: {sc_status}]"
                parts.append(sc_status)
            print(" ".join(parts))
            results.append(result)

    return results


def print_results(results: list[dict], baseline_path: str | None = None):
    """Print results as a formatted table, optionally comparing to baseline."""
    baseline = {}
    if baseline_path and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        baseline = {r["concurrency"]: r["throughput_tok_per_sec"]
                    for r in baseline_data["results"] if "concurrency" in r}

    headers = [
        "Concurrency",
        "tok/s (total)",
        "Requests",
        "Wall (s)",
        "Spot Checks",
    ]
    if baseline:
        headers.append("Baseline tok/s")
        headers.append("% of Baseline")

    rows = []
    for r in results:
        total_req = r["successful_requests"] + r["failed_requests"]
        spot = f"{r['spot_checks_passed']}/{r['spot_checks_total']}"

        row = [
            r["concurrency"],
            r["throughput_tok_per_sec"],
            f"{r['successful_requests']}/{total_req}",
            r["wall_time_sec"],
            spot,
        ]
        if baseline:
            bl = baseline.get(r["concurrency"], 0)
            row.append(bl if bl else "—")
            pct = (r["throughput_tok_per_sec"] / bl * 100) if bl else 0
            row.append(f"{pct:.1f}%" if bl else "—")
        rows.append(row)

    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))

    # Print warnings summary
    discrepancies = [r for r in results if r.get("token_discrepancy")]
    spot_failures = [r for r in results if r["spot_checks_passed"] < r["spot_checks_total"]]

    if discrepancies:
        print(f"\n[WARNING] Token count mismatch detected at concurrency: "
              f"{[r['concurrency'] for r in discrepancies]}")
        for r in discrepancies:
            print(f"  c={r['concurrency']}: "
                  f"prompt reported={r['reported_prompt_tokens']} verified={r['total_prompt_tokens']}, "
                  f"completion reported={r['reported_completion_tokens']} verified={r['total_completion_tokens']}")

    if spot_failures:
        print(f"\n[WARNING] Correctness spot-checks failed in some levels.")
        for r in spot_failures:
            print(f"  c={r['concurrency']}: {r['spot_checks_passed']}/{r['spot_checks_total']} passed")


def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark for inference engines")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--num-requests", type=int, default=DEFAULT_NUM_REQUESTS,
                        help="Number of requests per concurrency level")
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS,
                        help="Number of random prompts to generate")
    parser.add_argument("--input-tokens", type=int, default=INPUT_TOKENS,
                        help="Input sequence length in tokens")
    parser.add_argument("--max-tokens", type=int, default=OUTPUT_TOKENS,
                        help="Max output tokens per request")
    parser.add_argument("--concurrency", type=int, nargs="+", default=DEFAULT_CONCURRENCY_LEVELS,
                        help="Concurrency levels to test")
    parser.add_argument("--output", default=None, help="Output file for results JSON")
    parser.add_argument("--baseline", default="baseline/results/throughput_baseline.json",
                        help="Path to baseline results for comparison")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    # Load tokenizer for prompt generation and verification
    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Generate all prompts at runtime (like vLLM's RandomDataset)
    print(f"Generating {args.num_prompts} random prompts of {args.input_tokens} tokens...")
    prompts = generate_prompts(tokenizer, args.num_prompts, args.input_tokens)
    print(f"Generated {len(prompts)} prompts")

    print(f"Config: {args.num_requests} requests/level, ISL={args.input_tokens}, OSL={args.max_tokens}")
    print(f"Concurrency levels: {args.concurrency}")
    print(f"Target: {base_url}\n")

    print("Running throughput benchmark:")
    results = asyncio.run(run_benchmark(
        base_url, prompts, args.concurrency, args.num_requests, args.max_tokens, tokenizer
    ))

    print_results(results, args.baseline)

    output = {
        "model": MODEL_ID,
        "config": {
            "input_tokens": args.input_tokens,
            "output_tokens": args.max_tokens,
            "num_requests_per_level": args.num_requests,
            "num_prompts": args.num_prompts,
            "spot_checks_per_level": 2,
        },
        "results": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print(f"\nJSON output:\n{json.dumps(output, indent=2)}")


if __name__ == "__main__":
    main()
