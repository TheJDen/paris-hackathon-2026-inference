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
import signal
import subprocess
import threading
import time

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


def _source_dir() -> str:
    """What add_local_dir ships to /workspace.

    Default: the working tree (".") — good for iterating on uncommitted WIP.
    CB_SHIP_COMMIT=1: a clean export of the current git commit (HEAD), so bench-cloud
    validates COMMITTED code, not uncommitted WIP. (add_local_dir copies from disk, so
    WIP otherwise leaks in — which is how a half-written edit once broke the smoke.)
    The uncommitted smoke script is overlaid on the export so diagnostics stay good while
    the *engine* under test is exactly the commit.
    """
    if not os.environ.get("CB_SHIP_COMMIT"):
        return "."
    export = "/tmp/cb-commit-export"
    subprocess.run(
        f"rm -rf {export} && mkdir -p {export} && git archive HEAD | tar -x -C {export}",
        shell=True, check=True)
    subprocess.run(
        f"cp scripts/engine_smoke_modal.sh {export}/scripts/engine_smoke_modal.sh",
        shell=True, check=True)
    print(f"[modal] CB_SHIP_COMMIT=1 — shipping git HEAD from {export}", flush=True)
    return export


SOURCE_DIR = _source_dir()


def _gpu_spec():
    if CB_NGPUS <= 0 or CB_GPU.lower() in ("", "none"):
        return None
    if CB_NGPUS == 1:
        return CB_GPU
    return f"{CB_GPU}:{CB_NGPUS}"


# Engine image: CUDA devel base (has nvcc to build kernels) + engine deps + the
# fast kernels. Modal builds this once and caches it. See engine-kernel-deps notes
# for the gotchas baked in here (FLA from git; uninstall `kernels` after building
# causal_conv1d or it breaks the transformers import).
IMAGE = (
    # torch 2.8 ships Triton 3.4 = where `tl.make_tensor_descriptor` (modern TMA API)
    # became stable — the reason for the bump. Its base image is Python 3.11 (mature
    # wheels). NB: do NOT use 2.10 here — that base ships Python 3.14, and Modal's own
    # pinned aiohttp has a stale Cython `_websocket.c` referencing the removed
    # `ob_digit` CPython internal, so its client-deps step fails to compile a wheel.
    modal.Image.from_registry(
        "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel",
        # Modal auto-injects its own client bootstrap (`COPY modal_requirements.txt`
        # + `pip install --upgrade pip`) right after FROM, BEFORE any of our layers.
        # If the base's Python is PEP-668 externally-managed, that injected pip fails —
        # and our .env below can't help a step that runs before it. setup_dockerfile_commands
        # run FIRST (before Modal's bootstrap), so strip the EXTERNALLY-MANAGED marker here.
        setup_dockerfile_commands=[
            "RUN find / -name EXTERNALLY-MANAGED -delete 2>/dev/null || true",
        ],
    )
    .env({"PIP_BREAK_SYSTEM_PACKAGES": "1"})  # belt-and-suspenders for our own pips
    .apt_install("git", "build-essential", "ninja-build", "curl")
    # NB: all pip via run_commands (not .pip_install) — the latter injects an
    # `upgrade pip` step that ignores PIP_BREAK_SYSTEM_PACKAGES and fails on PEP-668.
    .run_commands(
        "pip install --break-system-packages fastapi 'uvicorn[standard]' "
        "'transformers>=5.12' 'pydantic>=2' huggingface_hub safetensors einops",
        "pip install --break-system-packages --no-deps "
        "git+https://github.com/fla-org/flash-linear-attention",
        "pip install --break-system-packages --no-build-isolation flash-attn",
        "pip install --break-system-packages kernels "
        "&& pip install --break-system-packages --no-build-isolation causal_conv1d "
        "&& pip uninstall -y kernels kernels-data",
        # eval/throughput driver deps (appended as its own layer so the flash-attn/fla
        # layers above stay cached).
        "pip install --break-system-packages tabulate aiohttp numpy",
    )
    .env({"HF_HOME": "/models"})  # point the HF cache at the mounted Volume
    .add_local_dir(SOURCE_DIR, remote_path="/workspace", ignore=[
        ".git", ".venv", "results", "__pycache__", "*.pyc",
    ])
)

# Persistent cache for the ~70GB model — survives across runs and won't fit in the
# function's ephemeral disk.
HF_CACHE = modal.Volume.from_name("paris-hf-cache", create_if_missing=True)

app = modal.App(CB_APP_NAME)


@app.function(gpu=_gpu_spec(), timeout=CB_TIMEOUT, image=IMAGE,
              volumes={"/models": HF_CACHE})
def run_bench(command: str, cb_timeout: int) -> dict:
    """Run the benchmark command inside the Modal container.

    Streams output live (visible via `modal app logs` / the local `modal run`
    tail) instead of buffering, and self-limits ~90s under the Modal function
    timeout via a watchdog: on deadline we terminate the command *gracefully*,
    then still commit the volume and stage whatever artifacts exist. That turns a
    Modal hard-kill (which returns nothing) into a partial return — so a slow load
    can't silently eat the whole run and lose the trace.
    """
    os.makedirs(f"/workspace/{CB_ARTIFACTS_DIR}", exist_ok=True)

    # cb_timeout is passed as an arg because Modal does NOT forward the local env to
    # the container — reading CB_TIMEOUT from os.environ here would hit the 600 default.
    deadline = max(30, cb_timeout - 90)  # leave headroom to commit + tar + return
    # start_new_session -> the shell + all children (bash, uvicorn, driver) share a
    # process group we can signal as a unit; SIGTERM to just the wrapper would orphan
    # them and skip bash's EXIT trap that stages server.log.
    proc = subprocess.Popen(
        command, shell=True, cwd="/workspace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        start_new_session=True,
    )
    timed_out = {"v": False}

    def _signal_group(sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _watchdog() -> None:
        start = time.monotonic()
        while proc.poll() is None:
            if time.monotonic() - start > deadline:
                timed_out["v"] = True
                print(f"[modal] watchdog: {deadline}s deadline hit — SIGTERM the "
                      "process group and staging partial results", flush=True)
                _signal_group(signal.SIGTERM)  # lets bash EXIT trap stage server.log
                try:
                    proc.wait(20)
                except subprocess.TimeoutExpired:
                    _signal_group(signal.SIGKILL)
                return
            time.sleep(2)

    threading.Thread(target=_watchdog, daemon=True).start()

    out_chunks: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)  # -> container stdout, forwarded live
        out_chunks.append(line)
    proc.wait()
    stdout_text = "".join(out_chunks)

    # Persist the Triton cache (and anything else written to /models) so the next
    # run reuses it. Best-effort — never fail the envelope over a commit hiccup.
    try:
        HF_CACHE.commit()
    except Exception as e:  # noqa: BLE001
        print(f"[modal] volume commit failed (non-fatal): {e}", flush=True)

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
        "exit_code": 124 if timed_out["v"] else proc.returncode,
        "stdout": stdout_text,
        "stderr": "",  # merged into stdout (stderr=STDOUT) so ordering is preserved
        "benchmark_result": benchmark_result,
        "artifacts_tar_b64": artifacts_tar_b64,
        "artifacts_note": artifacts_note,
        "timed_out": timed_out["v"],
    }


@app.local_entrypoint()
def main():
    envelope = run_bench.remote(CB_COMMAND, CB_TIMEOUT)
    print(RESULT_BEGIN)
    print(json.dumps(envelope))
    print(RESULT_END)
