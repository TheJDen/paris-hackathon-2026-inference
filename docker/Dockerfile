# Runtime image for OUR inference engine (NOT the vLLM baseline).
#
# Build on x86_64 — cloud GPUs are amd64 AND the fast-kernel wheels are amd64.
# Do NOT build this on the arm64 GB10; build on an x86 host or a rented instance:
#   make docker-build DOCKER_IMAGE=ghcr.io/<you>/paris-engine:latest
#   make docker-push  DOCKER_IMAGE=ghcr.io/<you>/paris-engine:latest
#
# Use it:
#   make bench-cloud PROVIDER=nebius BENCH=engine_smoke \
#     SYNC_MODE=docker REGISTRY_IMAGE=ghcr.io/<you>/paris-engine:latest
#
# Only DEPENDENCIES are baked in. Your source is rsynced to /workspace at run
# time, so editing engine code does NOT require an image rebuild.

# devel (not runtime) base: needs nvcc + headers to build causal_conv1d.
# 2.7 base = Triton 3.3, which FLA git-main REQUIRES (its autotuner needs `do_bench`;
# older Triton crashes `import fla` on model load). Validated on Modal H200. Don't
# downgrade below 2.7 without re-checking FLA import.
ARG BASE=pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Engine Python deps (torch/cuda already in the base image).
COPY docker/requirements-bench.txt /tmp/requirements-bench.txt
RUN pip install --no-cache-dir -r /tmp/requirements-bench.txt

# --- Fast kernels: order + gotchas matter (see engine-kernel-deps notes) ---
# FLA: the PyPI wheel is broken (ships no fla/ops or fla/modules) -> install from
# git. Pure python/triton, no compile. --no-deps to not disturb the base torch/triton.
RUN pip install --no-cache-dir --no-deps "git+https://github.com/fla-org/flash-linear-attention"

# flash-attn: on x86 there are prebuilt wheels for common torch/cuda combos, so
# this usually pulls a wheel (no long compile). --no-build-isolation uses base torch.
RUN pip install --no-cache-dir --no-build-isolation flash-attn

# causal_conv1d needs `kernels` at BUILD time, but leaving `kernels` installed
# BREAKS the transformers model import at runtime (LayerRepository version
# incompatibility). So: install kernels transiently, build causal_conv1d, then
# REMOVE kernels + kernels-data before the image is finalized.
RUN pip install --no-cache-dir kernels \
    && pip install --no-cache-dir --no-build-isolation causal_conv1d \
    && pip uninstall -y kernels kernels-data

# Sanity: model class imports and all three fast-kernel flags are True at build time.
RUN python -c "import transformers; transformers.Qwen3_5MoeForConditionalGeneration; \
from transformers.utils.import_utils import is_flash_linear_attention_available as a, is_causal_conv1d_available as b, is_flash_attn_2_available as c; \
assert a() and b() and c(), (a(), b(), c()); print('kernels OK:', a(), b(), c())"
