"""Modal entrypoint for cloudbench's Modal provider.

Run indirectly via the harness:

    make bench-cloud PROVIDER=modal BENCH=decode_smoke

The cloudbench Modal provider invokes `modal run scripts/modal_bench.py`
with these env vars set:

    CB_COMMAND   benchmark command to run
    CB_GPU       GPU type (e.g. H200)
    CB_NGPUS     number of GPUs (0 = CPU)
    CB_TIMEOUT   timeout in seconds
    CB_APP_NAME  resource/app name

It runs the command on a Modal GPU function, then prints a JSON envelope
between sentinels so the provider can parse stdout/stderr/exit_code and any
benchmark_result.json the command produced.

NOTE: image build steps and multi-GPU support depend on your Modal account
and the project's dependencies — adjust `IMAGE` below as needed.
"""

from __future__ import annotations

import json
import os
import subprocess

import modal

RESULT_BEGIN = "__CLOUDBENCH_RESULT_BEGIN__"
RESULT_END = "__CLOUDBENCH_RESULT_END__"

CB_COMMAND = os.environ.get("CB_COMMAND", "python -c \"print('no command')\"")
CB_GPU = os.environ.get("CB_GPU", "H200")
CB_NGPUS = int(os.environ.get("CB_NGPUS", "1"))
CB_TIMEOUT = int(os.environ.get("CB_TIMEOUT", "600"))
CB_APP_NAME = os.environ.get("CB_APP_NAME", "paris-bench-modal")
CB_ARTIFACTS_DIR = os.environ.get("CB_ARTIFACTS_DIR", "artifacts")
# Cap on artifacts returned via the envelope (profiler reports get big fast).
CB_ARTIFACTS_MAX_MB = int(os.environ.get("CB_ARTIFACTS_MAX_MB", "64"))


def _gpu_spec():
    if CB_NGPUS <= 0 or CB_GPU.lower() in ("", "none"):
        return None
    if CB_NGPUS == 1:
        return CB_GPU
    return f"{CB_GPU}:{CB_NGPUS}"


# Adjust this image to match your benchmark's dependencies.
IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "tqdm", "torch")  # torch (CUDA build) for GPU benches
    .add_local_dir(".", remote_path="/workspace", ignore=[
        ".git", ".venv", "results", "__pycache__", "*.pyc",
    ])
)

app = modal.App(CB_APP_NAME)


@app.function(gpu=_gpu_spec(), timeout=CB_TIMEOUT, image=IMAGE)
def run_bench(command: str) -> dict:
    """Run the benchmark command inside the Modal container and capture output."""
    os.makedirs(f"/workspace/{CB_ARTIFACTS_DIR}", exist_ok=True)
    proc = subprocess.run(
        command, shell=True, cwd="/workspace",
        capture_output=True, text=True,
    )
    benchmark_result = None
    result_path = "/workspace/benchmark_result.json"
    if os.path.exists(result_path):
        try:
            with open(result_path) as fh:
                benchmark_result = json.load(fh)
        except (json.JSONDecodeError, OSError):
            benchmark_result = None

    # Tar the artifacts dir (profiler reports, chrome traces) back through the
    # envelope, size-capped — Modal has no instance to scp from.
    artifacts_tar_b64 = None
    artifacts_note = None
    art_path = f"/workspace/{CB_ARTIFACTS_DIR}"
    if os.path.isdir(art_path) and os.listdir(art_path):
        import base64
        import io
        import tarfile
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(art_path, arcname=CB_ARTIFACTS_DIR)
        size_mb = buf.tell() / 1e6
        if size_mb <= CB_ARTIFACTS_MAX_MB:
            artifacts_tar_b64 = base64.b64encode(buf.getvalue()).decode()
        else:
            artifacts_note = (
                f"artifacts {size_mb:.0f}MB exceed CB_ARTIFACTS_MAX_MB="
                f"{CB_ARTIFACTS_MAX_MB}MB; use a Modal Volume for large profiles")

    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "benchmark_result": benchmark_result,
        "artifacts_tar_b64": artifacts_tar_b64,
        "artifacts_note": artifacts_note,
    }


@app.local_entrypoint()
def main():
    envelope = run_bench.remote(CB_COMMAND)
    print(RESULT_BEGIN)
    print(json.dumps(envelope))
    print(RESULT_END)
