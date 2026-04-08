"""Compute final hackathon score from correctness and throughput results.

Correctness is a gate (>= 85% GSM8K exact match). Throughput is scored
as a weighted sum of verified tok/s across concurrency levels, with higher
concurrency weighted more heavily.

Usage:
    python -m eval.score \
        --correctness results/correctness.json \
        --throughput results/throughput.json
"""

import argparse
import json
import sys

CORRECTNESS_GATE = 0.85

CONCURRENCY_WEIGHTS = {
    1: 1,
    2: 1,
    4: 2,
    8: 2,
    16: 4,
    32: 4,
    64: 8,
}


def main():
    parser = argparse.ArgumentParser(description="Compute hackathon score")
    parser.add_argument("--correctness", required=True, help="Path to correctness results JSON")
    parser.add_argument("--throughput", required=True, help="Path to throughput results JSON")
    args = parser.parse_args()

    # Load correctness
    with open(args.correctness) as f:
        correctness = json.load(f)

    accuracy = correctness.get("accuracy")
    if accuracy is None:
        full = correctness.get("full_results", {})
        accuracy = full.get("exact_match,flexible-extract", full.get("exact_match,strict-match"))

    if accuracy is None:
        print("ERROR: Could not extract accuracy from correctness results.")
        sys.exit(1)

    print(f"Correctness: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Gate:        >= {CORRECTNESS_GATE*100:.0f}%")

    if accuracy < CORRECTNESS_GATE:
        print(f"\nFAILED correctness gate. Score = 0")
        sys.exit(0)

    print("PASSED correctness gate.\n")

    # Load throughput
    with open(args.throughput) as f:
        throughput = json.load(f)

    results = throughput.get("results", [])

    print(f"{'Concurrency':>12} {'tok/s':>15} {'Weight':>8} {'Weighted':>10}")
    print("-" * 50)

    total_weighted = 0.0
    total_weight = 0
    for r in results:
        c = r.get("concurrency")
        if c is None:
            continue
        verified = r.get("throughput_tok_per_sec", 0)
        weight = CONCURRENCY_WEIGHTS.get(c, 1)
        weighted = verified * weight
        total_weighted += weighted
        total_weight += weight
        print(f"{c:>12} {verified:>15.2f} {weight:>8}x {weighted:>10.2f}")

    print("-" * 50)
    print(f"{'FINAL SCORE':>12} {'':>15} {total_weight:>8}  {total_weighted:>10.2f}")
    print(f"\nScore: {total_weighted:.2f}")


if __name__ == "__main__":
    main()
