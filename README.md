# Inference Engine Hackathon

Build an inference engine from scratch for **Qwen/Qwen3.5-35B-A3B** on **8xH200**.

This repo provides the evaluation harness: a correctness benchmark, a throughput benchmark, and vLLM baseline numbers to compare against.

## Rules

### Objective

Implement a high-throughput inference engine for Qwen3.5-35B-A3B that serves an OpenAI-compatible chat completions API. You are scored on **throughput** (output tokens/sec), with **correctness as a hard requirement**.

### Submission

Your submission consists of:

1. **A start script** — a single script that launches your server and makes it ready on a given port. Must exit cleanly and leave the server running.
2. **Source code** — your full inference engine implementation.
3. **Documentation** — a brief writeup explaining your approach, architecture decisions, and any optimizations used.
4. **Results** — a JSON file containing the throughput results for your engine (we will verify the results ourselves, however to save us time, we would appreciate if you included the results in your submission to do preliminary scoring)

### Scoring

**Correctness is a gate.** Your engine must pass the GSM8K-CoT correctness evaluation (exact match >= 87.5%) to be eligible for throughput scoring. Submissions that fail correctness receive a score of 0.

**Throughput determines rank.** We measure verified output tokens/sec at concurrency levels 1, 2, 4, 8, 16, 32, 64. Higher concurrency levels carry higher weight in the final score:

| Concurrency | Weight |
|---|---|
| 1 | 1x |
| 2 | 1x |
| 4 | 2x |
| 8 | 2x |
| 16 | 4x |
| 32 | 4x |
| 64 | 8x |

**Final score** = weighted sum of verified tok/s across all concurrency levels (if correctness gate is passed).

### What's Allowed

- **Precision:** BF16 only. No FP8, INT8, INT4, or any other reduced precision.
- **Parallelism:** Any parallelism strategy is allowed — tensor parallel, pipeline parallel, expert parallel, data parallel, or any combination.
- **Inference optimizations:** Allowed, as long as they do not affect correctness or output accuracy. Examples of what's allowed:
  - FlashAttention or other fused attention kernels
  - Continuous batching / dynamic batching
  - KV cache optimizations (paged attention, etc.)
  - Prefix caching
  - CUDA graphs
  - Custom CUDA/Triton kernels
  - Speculative decoding (if output matches non-speculative)
  - Operator fusion
- **Not allowed:** Any optimization that changes model outputs compared to a BF16 reference implementation (e.g., quantization, pruning, distillation, approximate attention that drops tokens). **If you're unsure whether an optimization is allowed, ask the organizers.**
- **Language:** Any language. Python, C++, Rust, CUDA — whatever you want.
- **Libraries:** You may use low-level libraries (cuBLAS, cuDNN, NCCL, Triton, etc.) but not high-level inference frameworks (vLLM, SGLang, TensorRT-LLM, etc.). The point is to build the engine yourself.

### Hardware

- **8x NVIDIA H200** (141 GB HBM3e each, NVLink interconnect)
- No other hardware is available. Your engine must run entirely on this node.

## Model

| Property | Value |
|---|---|
| Model | [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) |
| Architecture | Hybrid Gated DeltaNet + Sparse MoE |
| Total params | 35B |
| Active params | 3B per token |
| Experts | 256 total, 9 active (8 routed + 1 shared) |
| Layers | 40 |
| Context length | 262,144 tokens |
| Hardware | 8x NVIDIA H200 (141GB each) |
| Parallelism | Tensor Parallel = 8 (you are free to use any other deployment you want) |

**Important:** Thinking mode must be disabled. Your server must not emit `<think>` tags in output.

## API Specification

Your server must implement the following endpoints:

### `GET /health`

Returns HTTP 200 when the server is ready.

### `POST /v1/chat/completions`

OpenAI-compatible chat completions (non-streaming only).

**Request:**

```json
{
  "model": "Qwen/Qwen3.5-35B-A3B",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "max_tokens": 1024,
  "temperature": 0.0,
  "top_p": 1.0
}
```

| Field | Required | Default | Description |
|---|---|---|---|
| `model` | Yes | — | Model name (can be ignored by server, but must be accepted) |
| `messages` | Yes | — | Array of `{role, content}` objects |
| `max_tokens` | Yes | — | Maximum output tokens |
| `temperature` | No | 0.0 | Sampling temperature |
| `top_p` | No | 1.0 | Nucleus sampling parameter |

**Response:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "Qwen/Qwen3.5-35B-A3B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "2+2 equals 4."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 8,
    "total_tokens": 33
  }
}
```

**Requirements:**
- Must handle at least 64 concurrent requests
- `usage` field with token counts is mandatory
- `finish_reason` must be `"stop"` (max_tokens reached) or `"length"`

## Quick Start

### 1. Install the harness

```bash
uv venv && uv pip install -e .
```

### 2. Check your server is conformant

```bash
python -m eval.check_server --base-url http://localhost:8000
```

### 3. Run correctness eval (GSM8K-CoT, 200 problems)

```bash
python -m eval.correctness.run_correctness --base-url http://localhost:8000
```

### 4. Run throughput benchmark

```bash
python -m eval.throughput.run_throughput --base-url http://localhost:8000
```

## Evaluation Details

### Correctness: GSM8K-CoT

- **What:** 200 grade-school math problems requiring chain-of-thought reasoning
- **Why:** Math is extremely sensitive to implementation bugs — wrong attention masks, bad KV cache, quantization errors, or sampling bugs will cause accuracy to collapse
- **How:** Uses [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with `local-chat-completions` backend
- **Metric:** Exact match accuracy on the final numeric answer
- **Gate:** >= 87.5% exact match required to qualify for throughput scoring
- **Settings:** temperature=0, top_p=1.0, 8 concurrent requests
- **Seed:** Randomized at eval time

### Throughput

- **What:** Verified output tokens/sec at concurrency levels 1, 2, 4, 8, 16, 32, 64
- **Workload:** 1024 input tokens, 1024 output tokens per request, 64 requests per concurrency level
- **Warmup:** 2 requests discarded before measurement at each level
- **Prompts:** Generated at runtime using random token IDs from the full vocabulary (excluding special tokens), matching vLLM's `RandomDataset` approach with iterative decode-encode length adjustment. No pre-computed prompts — all generated fresh each run.
- **Token verification:** Output tokens are re-counted using the Qwen tokenizer — server-reported `usage.completion_tokens` is compared and discrepancies are flagged. Verified counts are used for scoring.
- **Spot checks:** 2 math questions per concurrency level are injected among random prompts to verify the server is producing correct outputs, not garbage

## Baseline Numbers

Generated using vLLM v0.19.0 on 8xH200 with TP=8, BF16 weights.

### Correctness: GSM8K-CoT (200 problems)

| Metric | Score |
|---|---|
| Exact match (flexible extract) | **91.5%** |
| Exact match (strict match) | **91.0%** |

### Throughput (1024 input / 1024 output tokens, 64 requests per level)

| Concurrency | tok/s (prompt+completion) | Wall Time (s) |
|---|---|---|
| 1 | 984 | 86.1 |
| 2 | 1,753 | 47.5 |
| 4 | 3,038 | 27.3 |
| 8 | 4,749 | 17.9 |
| 16 | 6,446 | 13.6 |
| 32 | 11,144 | 7.5 |
| 64 | 12,810 | 6.5 |

Run `./baseline/run_baseline.sh` to reproduce.

## Repository Structure

```
eval/
  check_server.py                  # Health check + API conformance
  correctness/
    run_correctness.py             # GSM8K-CoT evaluation wrapper
  throughput/
    run_throughput.py              # Async throughput benchmark (generates prompts at runtime)
baseline/
  run_baseline.sh                  # Generate vLLM baseline numbers
  results/                         # Baseline results
```
