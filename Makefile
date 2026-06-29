# cloudbench — provider-independent benchmark harness.
#
#   make test-smoke
#   make bench-local  BENCH=decode_smoke
#   make bench-cloud  PROVIDER=sesterce BENCH=single_h200_decode
#   make bench-cloud  PROVIDER=nebius   BENCH=e2e_8xh200
#   make bench-cloud  PROVIDER=modal    BENCH=decode_smoke
#   make bench-cloud  PROVIDER=sesterce BENCH=single_h200_decode BENCH_CMD="python my_new_test.py"
#
# Lifecycle env vars (pass inline, e.g. KEEP_INSTANCE=1 make bench-cloud ...):
#   KEEP_INSTANCE=1   leave the cloud resource running + print SSH instructions
#   DRY_RUN=1         print planned actions without creating resources
#   SYNC_MODE=rsync|docker
#   REGISTRY_IMAGE=...  pull a prebuilt image instead of rsyncing source
#   PROFILE=nsys|ncu  wrap the benchmark in a profiler
#
# The harness is a zero-dependency control plane: it runs under a bare
# `python3` (PyYAML optional, stdlib fallback otherwise) and never installs
# the project's ML stack. The benchmark COMMAND, for a local run, executes in
# whatever Python environment is active in your shell — so activate your
# workload venv first if your bench needs torch/vllm. Cloud runs execute the
# command on the remote instance, so nothing ML is needed locally at all.

BENCH     ?= decode_smoke
PROVIDER  ?= local
GPU       ?=
NGPUS     ?=
BENCH_CMD ?=

UV ?= uv
PY ?= python3

# Zero-dependency harness launcher. No venv, no sync, no project deps.
CB ?= $(PY) -m cloudbench.cli

OVERRIDE = $(if $(BENCH_CMD),--bench-cmd "$(BENCH_CMD)",)

# Docker bench image (torch + your deps baked in). Override DOCKER_IMAGE with a
# registry tag you can push to. Cloud H200s are x86_64 -> build linux/amd64.
DOCKER_IMAGE    ?= paris-bench:latest
DOCKER_BASE     ?= pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
DOCKER_PLATFORM ?= linux/amd64

.PHONY: sync test-smoke bench-local bench-cloud bench-list reap reap-dry docker-build docker-push

# Optional: materialize the project's ML env (.venv) for local workloads.
sync:
	$(UV) sync

# Harness self-test: isolated env + pytest, no ML deps installed.
test-smoke:
	$(UV) run --no-project --with pyyaml --with pytest python -m cloudbench.cli run --provider local --bench smoke

# Run a benchmark locally. The command runs in YOUR active shell env — if your
# bench needs torch, `source .venv/bin/activate` (or `uv run make bench-local`).
bench-local:
	$(CB) run --provider local --bench $(BENCH) $(OVERRIDE)

# Orchestrate a remote run. Harness is zero-dep; the workload runs remotely.
bench-cloud:
	$(CB) run --provider $(PROVIDER) --bench $(BENCH) $(OVERRIDE)

bench-list:
	$(CB) list

# Destroy any cloud resources left in the active ledger (orphans from a hard
# kill). Run this if a benchmark was force-killed or your machine crashed.
reap:
	$(CB) reap

reap-dry:
	$(CB) reap --dry-run

# Build the torch bench image (for the cloud, build linux/amd64 — needs buildx
# if you're on arm64). Then push it and pass it as REGISTRY_IMAGE.
docker-build:
	docker build --platform $(DOCKER_PLATFORM) --build-arg BASE=$(DOCKER_BASE) \
		-f docker/Dockerfile -t $(DOCKER_IMAGE) .

docker-push:
	docker push $(DOCKER_IMAGE)
