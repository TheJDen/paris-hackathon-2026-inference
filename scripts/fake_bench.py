"""Tiny but REAL GPU benchmark — matmul throughput in TFLOPS.

Proves the torch+CUDA path end-to-end on any provider. It auto-installs torch
if it's missing (so it works on bare CUDA images like Nebius's), measures
fp16 matmul throughput, and writes benchmark_result.json + an artifact.

    python3 scripts/fake_bench.py --size 4096 --iters 50 --dtype float16

For real benches, prefer baking torch into a Docker image
(SYNC_MODE=docker REGISTRY_IMAGE=...) instead of installing at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys
import time


def _have_pip() -> bool:
    return subprocess.run([sys.executable, "-m", "pip", "--version"],
                          capture_output=True).returncode == 0


def ensure_torch():
    try:
        import torch  # noqa: WPS433
        return torch
    except ImportError:
        pass
    print("[fake_bench] torch not found — installing once (this can take ~1-2 min)...",
          flush=True)

    # 1) Make sure pip exists: ensurepip, then apt (bare images like Nebius's
    #    worker-node have no pip at all).
    if not _have_pip():
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"],
                       capture_output=True)
    if not _have_pip():
        subprocess.run("sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip",
                       shell=True)
    if not _have_pip():
        raise RuntimeError("could not obtain pip (no ensurepip, apt failed)")

    # 2) Install torch with whatever flags THIS pip accepts. Order matters:
    #    --user works on old pip (ubuntu22); --user+--break-system-packages is
    #    needed for new pip under PEP 668 as a non-root user (ubuntu24/Nebius).
    last = ""
    for extra in (["--user"], ["--user", "--break-system-packages"],
                  ["--break-system-packages"], []):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                            *extra, "torch", "numpy"], capture_output=True, text=True)
        if r.returncode == 0:
            break
        last = (r.stderr or r.stdout or "").strip()

    # 3) Make a fresh --user install importable in this already-running process.
    import importlib
    import site
    try:
        site.main()
    except Exception:  # noqa: BLE001
        pass
    importlib.invalidate_caches()
    try:
        import torch  # noqa: WPS433
        return torch
    except ImportError as exc:
        raise RuntimeError(f"torch install failed; last pip error:\n{last[:600]}") from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=4096, help="square matrix dim")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dtype", default="float16")
    args, _ = ap.parse_known_args()

    torch = ensure_torch()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    n = args.size

    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(3):  # warmup
        c = a @ b
    if dev == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        c = a @ b  # noqa: F841
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    flops = 2 * (n ** 3) * args.iters
    result = {
        "device": dev,
        "gpu_name": torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "matrix_size": n,
        "iters": args.iters,
        "dtype": args.dtype,
        "seconds": round(dt, 4),
        "tflops": round(flops / dt / 1e12, 2),
        "iters_per_s": round(args.iters / dt, 2),
        "hostname": platform.node(),
    }

    pathlib.Path("benchmark_result.json").write_text(json.dumps(result, indent=2))
    art = pathlib.Path(os.environ.get("CB_ARTIFACTS_DIR", "artifacts"))
    art.mkdir(parents=True, exist_ok=True)
    (art / "fake_bench.json").write_text(json.dumps(result, indent=2))

    print("[fake_bench]", json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
