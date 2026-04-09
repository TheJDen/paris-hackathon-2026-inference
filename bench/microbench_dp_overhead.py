"""Micro-benchmark: measure the per-request latency overhead introduced by
the DP proxy (server/dp_proxy.py) vs hitting a single backend directly.

The proxy itself does no GPU work; this benchmark quantifies its pure
networking / routing overhead so we can confirm it is negligible compared to
the engine's generation latency.

Method:
  1. Send N tiny chat-completion requests to the proxy (--proxy-url).
  2. Send the same N requests to one backend directly (--backend-url).
  3. Compute per-request latency distributions for both.
  4. Report: median and p99 for proxy and direct, plus the overhead delta.

All requests are sent sequentially (concurrency=1) so the latency numbers
reflect pure wall-clock round-trip time, not throughput. This is intentional:
we want to isolate the proxy hop, not stress-test the engine.

Usage:
    # default: 100 requests, compare proxy on 8765 vs rank-0 backend on 7001
    python -m bench.microbench_dp_overhead

    # custom
    python -m bench.microbench_dp_overhead \\
        --proxy-url http://localhost:8765 \\
        --backend-url http://localhost:7001 \\
        --n 200 \\
        --max-tokens 4

Note: both the proxy AND at least one backend must be running before you
execute this script. Use scripts/start_dp.sh (or start a single backend
manually with --stub for a quick local sanity check).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Sequence

import aiohttp


# ---------------------------------------------------------------------------
# Tiny chat payload (designed for minimal engine work; we care about latency
# of the request pipeline, not generation quality).
# ---------------------------------------------------------------------------

_PAYLOAD = {
    "model": "Qwen/Qwen3.5-35B-A3B",
    "messages": [{"role": "user", "content": "Say 'ok'."}],
    "max_tokens": 4,
    "temperature": 0.0,
}


async def _send_one(session: aiohttp.ClientSession, url: str) -> float:
    """POST one completion and return wall-clock latency in milliseconds."""
    t0 = time.perf_counter()
    async with session.post(
        url,
        json=_PAYLOAD,
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        _ = await resp.read()  # consume body so the connection is recycled
        if resp.status not in (200, 201):
            raise RuntimeError(f"unexpected status {resp.status} from {url}")
    return (time.perf_counter() - t0) * 1000.0  # ms


async def _run_sequential(url: str, n: int, label: str) -> list[float]:
    """Send N requests sequentially and return a list of latencies (ms)."""
    print(f"  [{label}] sending {n} requests to {url} ...", flush=True)
    latencies: list[float] = []
    async with aiohttp.ClientSession() as session:
        for i in range(n):
            try:
                ms = await _send_one(session, url)
                latencies.append(ms)
            except Exception as exc:
                print(f"    request {i+1}/{n} FAILED: {exc}")
    return latencies


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    sorted_data = sorted(data)
    idx = max(0, min(len(sorted_data) - 1, int(round(p / 100.0 * (len(sorted_data) - 1)))))
    return sorted_data[idx]


def _report(label: str, latencies: list[float]) -> dict[str, float]:
    if not latencies:
        print(f"  {label}: NO DATA")
        return {}
    p50 = _percentile(latencies, 50)
    p99 = _percentile(latencies, 99)
    mean = statistics.mean(latencies)
    print(
        f"  {label:20s}  n={len(latencies):4d}  "
        f"mean={mean:7.1f}ms  p50={p50:7.1f}ms  p99={p99:7.1f}ms"
    )
    return {"mean_ms": mean, "p50_ms": p50, "p99_ms": p99, "n": len(latencies)}


async def main_async(args: argparse.Namespace) -> None:
    proxy_url = args.proxy_url.rstrip("/") + "/v1/chat/completions"
    backend_url = args.backend_url.rstrip("/") + "/v1/chat/completions"

    print(f"DP proxy overhead benchmark")
    print(f"  proxy   : {proxy_url}")
    print(f"  backend : {backend_url}")
    print(f"  n       : {args.n} requests each")
    print()

    # Warm up both endpoints (1 request each, not counted).
    print("Warming up ...")
    async with aiohttp.ClientSession() as session:
        for url in (proxy_url, backend_url):
            try:
                await _send_one(session, url)
            except Exception as exc:
                print(f"  WARNING: warm-up to {url} failed: {exc}")
    print()

    print("Proxy run:")
    proxy_lat = await _run_sequential(proxy_url, args.n, "proxy")
    print()

    print("Direct backend run:")
    direct_lat = await _run_sequential(backend_url, args.n, "direct")
    print()

    print("=" * 60)
    print("Results")
    print("=" * 60)
    proxy_stats = _report("proxy", proxy_lat)
    direct_stats = _report("direct backend", direct_lat)

    if proxy_stats and direct_stats:
        print()
        print("Proxy overhead (proxy - direct):")
        p50_overhead = proxy_stats["p50_ms"] - direct_stats["p50_ms"]
        p99_overhead = proxy_stats["p99_ms"] - direct_stats["p99_ms"]
        mean_overhead = proxy_stats["mean_ms"] - direct_stats["mean_ms"]
        print(f"  mean overhead : {mean_overhead:+.1f} ms")
        print(f"  p50 overhead  : {p50_overhead:+.1f} ms")
        print(f"  p99 overhead  : {p99_overhead:+.1f} ms")

        # Fraction of total latency attributable to the proxy hop.
        if direct_stats["p50_ms"] > 0:
            frac = p50_overhead / direct_stats["p50_ms"] * 100
            print(f"  p50 overhead as % of direct p50: {frac:.1f}%")

    if args.output:
        import os
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        result = {
            "proxy_url": args.proxy_url,
            "backend_url": args.backend_url,
            "n": args.n,
            "proxy": proxy_stats,
            "direct": direct_stats,
            "overhead": {
                "mean_ms": proxy_stats.get("mean_ms", float("nan")) - direct_stats.get("mean_ms", float("nan")),
                "p50_ms": proxy_stats.get("p50_ms", float("nan")) - direct_stats.get("p50_ms", float("nan")),
                "p99_ms": proxy_stats.get("p99_ms", float("nan")) - direct_stats.get("p99_ms", float("nan")),
            } if proxy_stats and direct_stats else {},
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nresults saved → {args.output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure per-request latency overhead of the DP=8 proxy vs a direct backend call"
    )
    p.add_argument(
        "--proxy-url",
        default="http://localhost:8765",
        help="base URL of the DP proxy (default: http://localhost:8765)",
    )
    p.add_argument(
        "--backend-url",
        default="http://localhost:7001",
        help="base URL of a single backend to compare against (default: http://localhost:7001)",
    )
    p.add_argument(
        "--n",
        type=int,
        default=100,
        help="number of requests to send to each endpoint (default: 100)",
    )
    p.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="save JSON results to this path (optional)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
