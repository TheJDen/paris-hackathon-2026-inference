# Benchmark runtime image for cloudbench — torch + CUDA + your deps baked in.
#
# Build (cloud GPUs are x86_64, so build for linux/amd64):
#   make docker-build DOCKER_IMAGE=ghcr.io/<you>/paris-bench:latest
#   make docker-push  DOCKER_IMAGE=ghcr.io/<you>/paris-bench:latest
#
# Use it (skips the slow per-run `pip install torch`):
#   make bench-cloud PROVIDER=nebius BENCH=single_h200_decode \
#     SYNC_MODE=docker REGISTRY_IMAGE=ghcr.io/<you>/paris-bench:latest
#
# Only DEPENDENCIES are baked in. Your source is rsynced and mounted at
# /workspace at run time, so editing code does NOT require a rebuild.

ARG BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM ${BASE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# torch/cuda are already in the base image. Add the rest of your bench deps to
# docker/requirements-bench.txt, then rebuild + push.
COPY docker/requirements-bench.txt /tmp/requirements-bench.txt
RUN pip install --no-cache-dir -r /tmp/requirements-bench.txt
