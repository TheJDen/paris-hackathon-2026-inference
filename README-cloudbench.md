# cloudbench

A provider-independent benchmark harness for the Paris hackathon inference repo.

> **BENCH** selects *what* to run. **PROVIDER** selects *where* to run it.

You run the same benchmarks against your laptop, Sesterce, Nebius, or Modal
without remembering any provider-specific CLI/API commands. The harness:

1. provisions the cloud hardware,
2. syncs your code (rsync) or pulls a Docker image,
3. runs the benchmark command,
4. downloads logs/results,
5. **destroys the resource by default**,

with cleanup guaranteed via `try/finally`. It never touches resources it did
not create, and names everything `paris-bench-${USER}-${timestamp}-${bench}`.

## Quick start

```bash
cp .env.example .env          # fill in provider credentials
make bench-list               # see available benchmarks

make test-smoke                                   # local CPU smoke test
make bench-local  BENCH=decode_smoke              # run a bench locally
make bench-cloud  PROVIDER=sesterce BENCH=decode_smoke
make bench-cloud  PROVIDER=sesterce BENCH=single_h200_decode
make bench-cloud  PROVIDER=nebius   BENCH=single_h200_decode
make bench-cloud  PROVIDER=nebius   BENCH=e2e_8xh200
make bench-cloud  PROVIDER=modal    BENCH=decode_smoke
```

Override any benchmark command on the fly:

```bash
make bench-cloud PROVIDER=sesterce BENCH=single_h200_decode BENCH_CMD="python my_new_test.py"
```

Equivalent CLI (the Makefile is a thin wrapper):

```bash
python -m cloudbench.cli list
python -m cloudbench.cli run --provider local    --bench decode_smoke
python -m cloudbench.cli run --provider sesterce --bench single_h200_decode
python -m cloudbench.cli run --provider nebius   --bench e2e_8xh200
python -m cloudbench.cli run --provider modal    --bench decode_smoke
```

## Environments — keeping the harness out of your project deps

There are **two separate environments**, by design, so neither pollutes the other:

| | Control plane (the harness) | Workload (the benchmark) |
|---|---|---|
| What | `cloudbench` orchestration: provision, sync, ssh, reap | `torch` / `vllm` / your bench code |
| Deps | **pure stdlib** (PyYAML optional) | the project's full ML stack |
| Runs on | your laptop | the GPU instance (or locally for CPU benches) |
| How it's launched | `uv run --no-project --with pyyaml` (isolated) | `uv run` (project env) |

The harness is a **zero-dependency control plane**. It never imports torch or
lm-eval — it just shells out to `ssh`/`rsync`/`nebius`/`modal`/`docker`. PyYAML
is optional (there's a stdlib fallback parser), so the harness even runs under a
bare `python3` with nothing installed:

```bash
python3 -m cloudbench.cli list        # works with no venv, no install, no deps
```

The Makefile wires this up automatically:

- `make bench-cloud`, `make bench-list`, `make reap` → **isolated** env
  (`uv run --no-project --with pyyaml`). No ML deps are installed to orchestrate
  a cloud run.
- `make test-smoke` → isolated env + `pytest` (still no ML deps).
- `make bench-local` → the **project** env (`uv run`), because the workload
  actually executes on your machine and needs torch etc.

### Provider CLIs (nebius, modal) — install them outside the project too

The harness shells out to these tools; it never imports them. Keep them out of
the project venv:

| Tool | What it is | Install (isolated) | Auth |
|------|-----------|--------------------|------|
| **nebius** | standalone binary (Go), *not* pip | official install script → lands on `PATH` | `nebius init` |
| **modal** | Python package | `uv tool install modal` (own isolated env) or `uvx modal …` | `modal token new` |
| **rsync / ssh / scp / docker** | system binaries | your OS package manager | — |

So:

```bash
# Nebius CLI — standalone binary, zero Python pollution:
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash   # verify URL in Nebius docs
nebius init

# Modal — installed in its OWN uv-managed tool env, never the project venv:
uv tool install modal      # puts `modal` on PATH, isolated from .venv
modal token new
```

`uv tool install` is the key: it gives `modal` a dedicated environment, so
`uv sync` / your project `.venv` stay clean. If a tool lives somewhere
non-standard, point the harness at it with `NEBIUS_BIN` / `MODAL_BIN` in `.env`
(see `.env.example`). Nothing about these CLIs ever enters `pyproject.toml`.

### Project Python with uv

```bash
uv sync                 # create .venv + lockfile from pyproject (Python 3.12, pinned in .python-version)
uv sync --group dev     # also install pytest (for working on the harness)
uv run python ...       # run inside the project env
uv run cloudbench list  # console script, available after a project sync
```

Dependency layout in `pyproject.toml`:

- `[project.dependencies]` — the eval/ML stack (unchanged).
- `[project.optional-dependencies].cloudbench = ["pyyaml"]` — optional, *not*
  required; install with `uv sync --extra cloudbench` if you want real YAML.
- `[dependency-groups].dev = ["pytest", "pyyaml"]` — dev-only, never shipped.

So adding the harness added **nothing** to your runtime dependencies.

## Intended development loop

1. Add or edit code locally.
2. `make test-smoke`.
3. `make bench-local BENCH=decode_smoke` if it can run on your machine.
4. `make bench-cloud PROVIDER=sesterce BENCH=decode_smoke` to verify the cloud path.
5. Iterate mostly with `PROVIDER=sesterce BENCH=single_h200_decode` (or `single_h200_prefill`).
6. Occasionally compare with `PROVIDER=nebius` on 1xH200.
7. Only run `PROVIDER=nebius BENCH=e2e_8xh200` for distributed validation / final numbers.
8. Use `KEEP_INSTANCE=1` only for debugging cloud setup issues.
9. Check `results/cloud/...` for logs and metadata.

## Lifecycle flags

Pass inline, e.g. `DRY_RUN=1 make bench-cloud PROVIDER=nebius BENCH=e2e_8xh200`.

| Flag | Effect |
|------|--------|
| `KEEP_INSTANCE=1` | Leave the cloud resource running and print SSH instructions. |
| `DRY_RUN=1` | Print planned actions; create nothing. |
| `SYNC_MODE=rsync` | (default) rsync repo source to the instance. |
| `SYNC_MODE=docker` | Pull and run inside a Docker image instead. |
| `REGISTRY_IMAGE=...` | Use a prebuilt image (implies `SYNC_MODE=docker`). |
| `BENCH_CMD="..."` | Override the benchmark command for one run. |
| `PROFILE=nsys\|ncu` | Wrap the command in a profiler (see [Profiling](#profiling-ncu--nsys--chrome-traces)). |
| `PROFILE_ARGS="..."` | Extra flags passed to the profiler. |
| `ARTIFACTS_DIR=...` | Dir collected after a run (default `artifacts`). |

> Stray resource after a crash? Run **`make reap`** to destroy anything the
> harness created but didn't get to clean up. See [Safety](#safety--never-leak-a-gpu).

## Output layout

Each run writes:

```
results/cloud/<timestamp>-<provider>-<bench>/
  metadata.json          provider, bench, command, git commit + dirty summary,
                         start/end/duration, gpu, ngpus, instance id, exit code,
                         cleanup status, sync mode, ...
  stdout.log
  stderr.log
  benchmark_result.json  if the benchmark produced one (downloaded from remote)
  provider.json          resource IDs, instance type, region, GPU count, SSH info
```

A benchmark "produces a result" by writing `benchmark_result.json` into the repo
root (local/Modal) or the remote workdir (`/root/bench`) — the harness picks it up.

## Benchmarks

Defined in [`configs/benchmarks.yaml`](configs/benchmarks.yaml). Add an entry:

```yaml
my_bench:
  command: python bench_thing.py --foo 1
  gpu: H200
  ngpus: 1
  timeout_minutes: 30
  description: What it measures.
```

Then `make bench-cloud PROVIDER=sesterce BENCH=my_bench`.

## Profiling (ncu / nsys / Chrome traces)

Anything written into the **`artifacts/`** dir (in the workdir) is automatically
downloaded into `results/cloud/<run>/artifacts/` after the run — on every
provider. That's the one rule: **profile output goes in `artifacts/`.**

### Launcher profilers — let the harness wrap the command

```bash
# Nsight Systems (timeline) — best for end-to-end + multi-GPU
PROFILE=nsys make bench-cloud PROVIDER=sesterce BENCH=single_h200_decode

# Nsight Compute (per-kernel) — single-process / single-GPU
PROFILE=ncu  make bench-cloud PROVIDER=sesterce BENCH=single_h200_decode

# Extra profiler flags:
PROFILE=nsys PROFILE_ARGS="--gpu-metrics-device=all" make bench-cloud ...
PROFILE=ncu  PROFILE_ARGS="--section SpeedOfLight"    make bench-cloud ...
```

The harness rewrites your command to, e.g.:

```
nsys profile --force-overwrite true -o artifacts/<resource>_nsys \
  --trace=cuda,nvtx,cublas,cudnn,osrt  python bench_decode.py ...
```

producing `artifacts/<resource>_nsys.nsys-rep` (or `_ncu.ncu-rep`), which is
collected automatically. The exact command is recorded in `metadata.json`
(`command` = wrapped, `base_command` = original, `profile` = mode). Open with
`nsys-ui report.nsys-rep` / `ncu-ui report.ncu-rep` locally.

### Chrome / Perfetto event JSON — your code emits it

The harness can't generate a torch trace; your benchmark writes it, into
`artifacts/`:

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    run_decode_step()
prof.export_chrome_trace("artifacts/trace.json")   # <-- collected automatically
```

View at `chrome://tracing` or [ui.perfetto.dev](https://ui.perfetto.dev).
(`nsys` reports can also be converted: `nsys export --type=json report.nsys-rep`.)

### Things to know

- **Tools must be in the image/instance.** `nsys`/`ncu` ship with the CUDA
  toolkit; make sure your `SESTERCE_IMAGE_ID` / `NEBIUS_IMAGE_ID` / Docker image
  has them on `PATH`.
- **Permissions.** `ncu` needs GPU perf-counter access — fine as root (the
  default remote user), otherwise set `NVreg_RestrictProfilingToAdminUsers=0`.
  In `SYNC_MODE=docker`, the harness adds `--cap-add=SYS_ADMIN` automatically
  when profiling.
- **`ncu` is slow** (it replays kernels). Bump `timeout_minutes` on the bench,
  and prefer `PROFILE_ARGS="--launch-count N"` / `--kernel-name` to scope it.
- **Multi-GPU:** use `nsys`, not `ncu` — `ncu` wraps a single process and won't
  see `torchrun` workers (the harness warns you if you try).
- **Modal:** artifacts come back through the function return, size-capped
  (`CB_ARTIFACTS_MAX_MB`, default 64). For large profiles use a real GPU
  provider (Sesterce/Nebius) or a Modal Volume.

## Providers

| Provider | Where | Notes |
|----------|-------|-------|
| `local` | this machine | No provisioning/cleanup. Good for CPU benches + the smoke test. |
| `sesterce` | Sesterce REST API | **Default for frequent 1xH200 iteration.** Endpoints centralized at the top of `cloudbench/providers/sesterce.py`. |
| `nebius` | Nebius CLI | 1xH200 sanity checks + 8xH200 distributed. Assumes CLI auth configured. `ngpus=8` uses the 8x platform / GPU cluster. |
| `modal` | Modal serverless | **Smoke tests only** (limited credits). Runs `scripts/modal_bench.py`. Serverless, so nothing to keep/destroy. |

Configure providers via `.env` (see `.env.example` for the full variable list).

### Patching provider APIs

The cloud SDK/API shapes may drift. To make that painless:

- **Sesterce**: `ENDPOINTS` and `FIELDS` dicts at the top of `providers/sesterce.py`.
- **Nebius**: the `CLI` verb dict at the top of `providers/nebius.py`.
- **Modal**: image build + GPU spec in `scripts/modal_bench.py`.

All four providers share the SSH/rsync/scp helpers in `providers/_remote.py` and
the create → wait → run → download → destroy lifecycle in `providers/__init__.py`
(`CloudProvider`), so adapters only implement create/wait/ssh-info/destroy.

## Safety — never leak a GPU

Leaving a GPU reserved is treated as the worst outcome (re-running work is
cheap). The harness defends against it in layers:

1. **Guaranteed teardown.** Cleanup runs in `try/finally` — a failed *or
   interrupted* benchmark still tears down hardware.
2. **Optimistic ownership.** The resource is "owned" from the moment creation
   is *attempted* and a ledger breadcrumb is written **before** the create
   call — so even a kill mid-create still triggers a destroy. If the instance
   id was never captured, the harness recovers it by name and destroys it.
3. **Signal handling.** `SIGTERM` (`kill`) and `SIGHUP` (closed terminal / SSH
   drop) are converted into a clean shutdown so `finally` cleanup runs. Plain
   `Ctrl-C` (`SIGINT`) already does.
4. **Interrupt-shielded destroy.** While a destroy is in flight, `SIGINT`/
   `SIGTERM`/`SIGHUP` are ignored, so a panicked second `Ctrl-C` cannot abort
   teardown and leak the GPU. (Let it finish — don't force-kill.)
5. **Retrying, idempotent destroy.** Destroy is retried up to 5× with backoff;
   an "already gone" (404/not-found) counts as success.
6. **Persistent ledger + `reap`.** Every created resource is recorded in
   `results/cloud/.active/`. The record is removed only after a confirmed
   destroy. Anything left there is a possible leak.

### What survives a *hard* kill

`kill -9` (SIGKILL), a power cut, or a closed laptop lid **cannot** run any
in-process cleanup — nothing can. For those, the ledger is the backstop:

```bash
make reap            # destroy everything still in the ledger
make reap-dry        # show what WOULD be destroyed first
```

The harness also **warns loudly at the start of every run** if the ledger
still lists resources, so orphans are caught early. `make reap` skips
`KEEP_INSTANCE` resources by default (`--include-kept` to force).

### Recommended belt-and-suspenders: provider-side TTL

Software can't help if the whole machine dies and you never run `reap`. For a
true hands-off guarantee, also set a **max-lifetime / auto-termination** on the
provider side (where supported) so the instance self-destructs after, say,
`timeout_minutes + buffer`. Wire this into `create_instance()` for your
provider — it's the only thing that protects you with zero local process alive.

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | benchmark passed, resource destroyed |
| `130` | interrupted (Ctrl-C/signal) — resource was still torn down |
| `3` | **cleanup failed** — a resource may still be running; run `make reap` |
| other | benchmark's own non-zero exit code |
```
