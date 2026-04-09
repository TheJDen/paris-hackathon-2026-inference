"""Quick throughput sweep for the per-change iteration loop.

Reuses `eval.throughput.run_throughput` so we measure exactly what the
official scoring measures (runtime random prompts via the same
iterative-decode-encode adjustment, same spot-check injection, same
tokenizer-verified tok/s). The differences from the full eval are:

  * Default to the **high-weight concurrency levels only** — c=16, c=32,
    c=64. These three carry 16/22 = 73% of the final score weight.
  * Default to **fewer requests per level** (16 instead of 64) so the
    sweep finishes in about a minute on a working engine. The full
    64-requests-per-level sweep is only run on demand or before merging.
  * Always print the **partial weighted score** using the official
    weights from `eval.score`, so each iteration tells you what the
    submission score would look like.
  * Pulls the engine's CLI region table at the end so you can diff it
    against the previous run.

Usage:
    # default — c=16,32,64, 16 reqs/level, ~1 min
    python -m bench.quick_throughput --base-url http://localhost:8765

    # bigger sweep that mirrors the official eval
    python -m bench.quick_throughput --base-url http://localhost:8765 \\
        --concurrency 1 2 4 8 16 32 64 --num-requests 64

    # save the result for later diffing
    python -m bench.quick_throughput --base-url http://localhost:8765 \\
        --output profiles/throughput_phase1_<sha>.json
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import sys
import time
from typing import Any
import urllib.error
import urllib.request

# Reuse the official harness internals so we measure the same thing it does.
from eval.score import CONCURRENCY_WEIGHTS
from eval.throughput.run_throughput import (
    INPUT_TOKENS,
    MODEL_ID,
    OUTPUT_TOKENS,
    generate_prompts,
    run_benchmark,
)

DEFAULT_QUICK_LEVELS = [16, 32, 64]
DEFAULT_QUICK_REQUESTS = 16
DEFAULT_NUM_PROMPTS = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quick throughput sweep for iteration (high-weight concurrency only)",
    )
    p.add_argument("--base-url", default="http://localhost:8765")
    p.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=DEFAULT_QUICK_LEVELS,
        help="concurrency levels to sweep (default: 16 32 64)",
    )
    p.add_argument(
        "--num-requests",
        type=int,
        default=DEFAULT_QUICK_REQUESTS,
        help="requests per concurrency level (default: 16; full eval uses 64)",
    )
    p.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    p.add_argument("--input-tokens", type=int, default=INPUT_TOKENS)
    p.add_argument("--max-tokens", type=int, default=OUTPUT_TOKENS)
    p.add_argument("--model", default=MODEL_ID, help="tokenizer source (id or local path)")
    p.add_argument("--output", default=None, help="save JSON results here")
    p.add_argument(
        "--regions-out",
        default=None,
        help="save engine region table here (requires server /metrics/regions)",
    )
    p.add_argument(
        "--no-fetch-regions",
        action="store_true",
        help="skip the /metrics/regions fetch at the end",
    )
    return p.parse_args()


def fetch_regions(base_url: str) -> str | None:
    """Pull the CLI region table from the engine, if it exposes one."""
    url = f"{base_url.rstrip('/')}/metrics/regions"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.URLError:
        return None
    except Exception:
        return None


def fetch_metrics(base_url: str) -> dict | None:
    url = f"{base_url.rstrip('/')}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def weighted_score(results: list[dict[str, Any]]) -> tuple[float, int, list[tuple[int, float, int]]]:
    """Sum tok/s × weight across the levels we ran.

    Returns (total_weighted, total_weight, per_level_rows).
    """
    rows: list[tuple[int, float, int]] = []
    total_weighted = 0.0
    total_weight = 0
    for r in results:
        c = r.get("concurrency")
        if c is None:
            continue
        tps = float(r.get("throughput_tok_per_sec", 0.0))
        w = CONCURRENCY_WEIGHTS.get(c, 0)
        total_weighted += tps * w
        total_weight += w
        rows.append((c, tps, w))
    return total_weighted, total_weight, rows


def print_summary(results: list[dict], elapsed_s: float, all_levels: bool = False) -> None:
    weighted, total_weight, rows = weighted_score(results)
    print()
    print(f"{'concurrency':>12} {'tok/s':>15} {'wall_s':>10} {'reqs ok':>10} {'spot':>8} {'weight':>8}")
    print("-" * 75)
    for r in results:
        c = r["concurrency"]
        tps = r["throughput_tok_per_sec"]
        wall = r["wall_time_sec"]
        ok = f"{r['successful_requests']}/{r['successful_requests']+r['failed_requests']}"
        spot = f"{r['spot_checks_passed']}/{r['spot_checks_total']}"
        w = CONCURRENCY_WEIGHTS.get(c, 0)
        print(f"{c:>12} {tps:>15.2f} {wall:>10.2f} {ok:>10} {spot:>8} {w:>7}x")
    print("-" * 75)

    # Score the *levels we actually ran* and compare to the full-eval max.
    print(f"  partial weighted score (this run): {weighted:.0f}")
    print(f"  weight covered by this run:        {total_weight}/22")
    if all_levels and total_weight == 22:
        print(f"  FULL WEIGHTED SCORE = {weighted:.0f}")
    elif total_weight > 0:
        # Extrapolate the score to all 7 levels by assuming the missing
        # levels' tok/s scales linearly. Useful as a rough "where would we
        # land in a full sweep" hint, NOT the official number.
        extrap = weighted * (22 / total_weight)
        print(f"  rough extrapolation to all 7 levels: ~{extrap:.0f}  (NOT the official score)")
    print(f"  elapsed: {elapsed_s:.1f}s")


def main() -> None:
    args = parse_args()

    print(f"target: {args.base_url}")
    print(f"levels: {args.concurrency}  reqs/level: {args.num_requests}  isl={args.input_tokens} osl={args.max_tokens}")

    print("loading tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"generating {args.num_prompts} random prompts of {args.input_tokens} tokens...")
    prompts = generate_prompts(tokenizer, args.num_prompts, args.input_tokens)

    print("sweeping...")
    t0 = time.perf_counter()
    results = asyncio.run(
        run_benchmark(
            args.base_url,
            prompts,
            list(args.concurrency),
            args.num_requests,
            args.max_tokens,
            tokenizer,
        )
    )
    elapsed = time.perf_counter() - t0
    all_levels = sorted(args.concurrency) == [1, 2, 4, 8, 16, 32, 64]
    print_summary(results, elapsed, all_levels=all_levels)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        weighted, total_weight, rows = weighted_score(results)
        out = {
            "model": args.model,
            "config": {
                "input_tokens": args.input_tokens,
                "output_tokens": args.max_tokens,
                "num_requests_per_level": args.num_requests,
                "num_prompts": args.num_prompts,
                "concurrency": list(args.concurrency),
            },
            "results": results,
            "partial_weighted_score": weighted,
            "weight_covered": total_weight,
            "elapsed_s": elapsed,
            "captured_at": _dt.datetime.utcnow().isoformat() + "Z",
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nresults saved → {args.output}")

    if not args.no_fetch_regions:
        regions = fetch_regions(args.base_url)
        if regions:
            print("\n" + "=" * 75)
            print("engine region table (post-sweep)")
            print("=" * 75)
            print(regions)
            if args.regions_out:
                os.makedirs(os.path.dirname(args.regions_out) or ".", exist_ok=True)
                with open(args.regions_out, "w") as f:
                    f.write(regions)
                print(f"regions saved → {args.regions_out}")
        metrics = fetch_metrics(args.base_url)
        if metrics:
            print("\nlive /metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
