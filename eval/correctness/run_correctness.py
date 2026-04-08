"""Correctness evaluation using GSM8K-CoT via lm-evaluation-harness.

Runs 200 GSM8K chain-of-thought problems against an OpenAI-compatible
chat completions endpoint and reports exact-match accuracy.

Usage:
    python -m eval.correctness.run_correctness --base-url http://localhost:8000
"""

import argparse
import json
import os
import random
import sys
import subprocess


TASK = "gsm8k_cot"
LIMIT = 200


def run_eval(base_url: str, output_dir: str, num_concurrent: int, limit: int, seed: int) -> dict:
    """Run lm-evaluation-harness against the target server."""
    base_url = base_url.rstrip("/")
    model_args = (
        f"model=Qwen/Qwen3.5-35B-A3B,"
        f"base_url={base_url}/v1/chat/completions,"
        f"num_concurrent={num_concurrent},"
        f"tokenizer_backend=huggingface,"
        f"timeout=600"
    )

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", model_args,
        "--tasks", TASK,
        "--limit", str(limit),
        "--apply_chat_template",
        "--gen_kwargs", "temperature=0,top_p=1.0",
        "--seed", str(seed),
        "--output_path", output_dir,
        "--log_samples",
    ]

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\nlm-eval exited with code {result.returncode}")
        sys.exit(1)

    # Find and parse the results file
    results_file = find_results_file(output_dir)
    if results_file:
        with open(results_file) as f:
            return json.load(f)
    return {}


def find_results_file(output_dir: str) -> str | None:
    """Find the lm-eval results JSON file in the output directory."""
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f == "results.json":
                return os.path.join(root, f)
    return None


def print_results(results: dict, baseline_path: str | None = None):
    """Print accuracy results, optionally comparing to baseline."""
    if not results:
        print("No results found.")
        return

    task_results = results.get("results", {}).get(TASK, {})
    flexible = task_results.get("exact_match,flexible-extract")
    strict = task_results.get("exact_match,strict-match")
    accuracy = flexible or strict

    if accuracy is None:
        for key, val in task_results.items():
            if "exact_match" in key and "stderr" not in key:
                accuracy = val
                break

    if accuracy is not None:
        print(f"\n{'='*60}")
        print(f"  GSM8K-CoT Results ({LIMIT} problems)")
        print(f"{'='*60}")
        if flexible is not None:
            print(f"  Exact match (flexible extract): {flexible:.4f} ({flexible*100:.1f}%)")
        if strict is not None:
            print(f"  Exact match (strict match):     {strict:.4f} ({strict*100:.1f}%)")

        if baseline_path and os.path.exists(baseline_path):
            with open(baseline_path) as f:
                baseline = json.load(f)
            bl_accuracy = baseline.get("accuracy")
            if bl_accuracy is not None:
                print(f"  vLLM Baseline (flexible):       {bl_accuracy:.4f} ({bl_accuracy*100:.1f}%)")
                diff = (flexible or accuracy) - bl_accuracy
                print(f"  Difference:                     {diff:+.4f} ({diff*100:+.1f}%)")
        print(f"{'='*60}\n")
    else:
        print("\nCould not extract accuracy metric from results.")
        print(f"Available metrics: {list(task_results.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Correctness eval using GSM8K-CoT")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--limit", type=int, default=LIMIT, help="Number of problems to evaluate")
    parser.add_argument("--num-concurrent", type=int, default=8,
                        help="Number of concurrent requests to the server")
    parser.add_argument("--output", default=None, help="Output file for summary JSON")
    parser.add_argument("--output-dir", default="results/correctness",
                        help="Directory for lm-eval output")
    parser.add_argument("--baseline", default="baseline/results/correctness_baseline.json",
                        help="Path to baseline results for comparison")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for problem selection (random if not set)")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    # Randomize seed at eval time to prevent answer memorization
    seed = args.seed if args.seed is not None else random.randint(0, 999999)

    print(f"Target: {base_url}")
    print(f"Task: {TASK} ({args.limit} problems)")
    print(f"Concurrent requests: {args.num_concurrent}")
    print(f"Seed: {seed}\n")

    results = run_eval(base_url, args.output_dir, args.num_concurrent, args.limit, seed)
    print_results(results, args.baseline)

    if args.output and results:
        task_results = results.get("results", {}).get(TASK, {})
        accuracy = None
        for key, val in task_results.items():
            if "exact_match" in key:
                accuracy = val
                break

        summary = {
            "task": TASK,
            "limit": args.limit,
            "seed": seed,
            "accuracy": accuracy,
            "full_results": task_results,
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {args.output}")


if __name__ == "__main__":
    main()
