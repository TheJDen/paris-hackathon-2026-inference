"""Tiny example benchmark — proves the cloudbench path end-to-end.

It does no real work; it just records environment info, writes
``benchmark_result.json`` (which the harness downloads) and drops a file in
the artifacts dir. Use it as a template: replace with your real
``bench_decode.py`` / ``bench_prefill.py`` / ``serve_bench.py``.

    python scripts/hello_bench.py [--tokens N] [--batch-size N] [...]

Unknown args are accepted and recorded, so the same script works for any of
the benchmark commands in configs/benchmarks.yaml.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path


def collect() -> dict:
    info = {
        "hello": "cloudbench",
        "python": platform.python_version(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "argv": sys.argv[1:],
        "unix_time": int(time.time()),
        "torch": None,
        "cuda_available": False,
    }
    # Report GPU info if torch happens to be installed (optional).
    try:
        import torch  # noqa: WPS433
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:  # noqa: BLE001 - torch is optional for the smoke path
        pass
    return info


def main() -> int:
    info = collect()

    # 1) Result file the harness collects from the workdir.
    Path("benchmark_result.json").write_text(json.dumps(info, indent=2))

    # 2) Something in the artifacts dir (downloaded too).
    artifacts = Path(os.environ.get("CB_ARTIFACTS_DIR", "artifacts"))
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "hello.txt").write_text("cloudbench example artifact\n")

    print("hello_bench OK ->", json.dumps(info))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
