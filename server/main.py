"""Server entrypoint.

Usage:
    python -m server.main --model Qwen/Qwen3.5-35B-A3B --port 8000 [--stub]

In Phase 0 you will almost always want `--stub` so the server boots without
a GPU and you can verify the OpenAI shape against `eval/check_server.py`.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from engine.runtime.engine import Engine
from engine.runtime.metrics import metrics
from engine.runtime.profiling import enable_torch_profiler
from server.app import create_app


log = logging.getLogger("server.main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="paris-hackathon-2026-inference server")
    p.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument(
        "--batch-window-ms",
        type=float,
        default=5.0,
        help="time the batcher waits to gather more requests before launching",
    )
    p.add_argument("--device", default="cuda:0", help="device for the (Phase 1) single-GPU engine")
    p.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="HF attention implementation flag",
    )
    p.add_argument(
        "--stub",
        action="store_true",
        help="run without loading a model — returns canned text (Phase 0 smoke test)",
    )
    p.add_argument(
        "--metrics-interval",
        type=float,
        default=5.0,
        help="seconds between live metrics flushes to stdout (0 = off)",
    )
    p.add_argument(
        "--profile-torch",
        action="store_true",
        help="arm torch.profiler so record_function regions are emitted",
    )
    p.add_argument(
        "--profile-torch-after-batches",
        type=int,
        default=0,
        help="capture a one-shot torch profile of the FIRST batch after N warmup "
        "batches whose size meets --profile-torch-min-batch-size (0 disables). "
        "Output lands in profiles/torch_<tag>_b<bs>_n<n>_<sha>_<ts>.{json.gz,txt,summary.json}",
    )
    p.add_argument(
        "--profile-torch-min-batch-size",
        type=int,
        default=1,
        help="minimum batch size for the one-shot profile capture; the engine "
        "skips smaller batches (e.g. per-level warmups) and waits for the next one",
    )
    p.add_argument("--profile-tag", default="run", help="tag for profile artifact names")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    if args.profile_torch or args.profile_torch_after_batches > 0:
        enable_torch_profiler(tag=args.profile_tag)

    log.info(
        "building engine model=%s stub=%s tp=%d max_batch=%d max_model_len=%d",
        args.model, args.stub, args.tp, args.max_batch, args.max_model_len,
    )
    engine = Engine.build(
        args.model,
        stub=args.stub,
        tp=args.tp,
        max_batch=args.max_batch,
        max_model_len=args.max_model_len,
        device=args.device,
        attn_impl=args.attn_impl,
        batch_window_ms=args.batch_window_ms,
        profile_torch_after_batches=args.profile_torch_after_batches,
        profile_torch_min_batch_size=args.profile_torch_min_batch_size,
        profile_torch_tag=args.profile_tag,
    )

    if args.metrics_interval > 0:
        metrics.start_flusher(interval_s=args.metrics_interval)

    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
