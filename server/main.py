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
    p.add_argument(
        "--ep",
        type=int,
        default=1,
        help="expert parallel size. When >1, launch with torchrun "
        "--nproc_per_node=EP. Each rank loads full weights but only runs its "
        "slice of experts in the MoE forward; routed contribution is all-reduced. "
        "Only rank 0 binds --port; other ranks run an EP worker loop.",
    )
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
    p.add_argument(
        "--cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable CUDA graph capture for the decode hot path (default: on). "
        "Use --no-cuda-graphs to disable (e.g. for profiling eager kernels or "
        "debugging). First call per bucket pays ~1-3s capture cost; subsequent "
        "calls replay the graph with near-zero CPU dispatch overhead.",
    )
    p.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="wrap the inner transformer model with torch.compile (Inductor, "
        "reduce-overhead mode). Fuses RMSNorm/SiLU/elementwise ops; expects a "
        "30-90s compile cost on first warm-up before the server takes traffic. "
        "Use --no-torch-compile to fall back to eager (default: on). "
        "NOTE: when --torch-compile is on, --cuda-graphs is automatically "
        "disabled to avoid conflict (reduce-overhead owns CUDA graphs internally).",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    # TP / EP aware: under torchrun, WORLD_SIZE / RANK / LOCAL_RANK
    # are set by the launcher. Only rank 0 binds HTTP. Non-zero ranks load
    # the same engine and enter a passive worker loop so they participate
    # in the MoE / RowParallelLinear all_reduce collectives driven by rank 0.
    import os as _os_main
    _ep_world = int(_os_main.environ.get("WORLD_SIZE", "1"))
    _ep_rank = int(_os_main.environ.get("RANK", "0"))
    _ep_local_rank = int(_os_main.environ.get("LOCAL_RANK", "0"))
    if (args.ep > 1 or args.tp > 1) and _ep_world > 1:
        args.device = f"cuda:{_ep_local_rank}"

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
        ep=args.ep,
        max_batch=args.max_batch,
        max_model_len=args.max_model_len,
        device=args.device,
        attn_impl=args.attn_impl,
        batch_window_ms=args.batch_window_ms,
        profile_torch_after_batches=args.profile_torch_after_batches,
        profile_torch_min_batch_size=args.profile_torch_min_batch_size,
        profile_torch_tag=args.profile_tag,
        enable_cuda_graphs=args.cuda_graphs,
        torch_compile=args.torch_compile,
    )

    if args.metrics_interval > 0 and _ep_rank == 0:
        metrics.start_flusher(interval_s=args.metrics_interval)

    # TP > 1: workers re-exec'd by torchrun enter a lockstep loop that
    # receives broadcast forward inputs from rank 0 and runs the local
    # (sharded) inner_model.forward so collectives match up.
    if args.tp > 1 and _ep_world > 1 and _ep_rank != 0:
        from engine.runtime.tp_worker import tp_worker_loop
        log.info("TP worker rank=%d local_rank=%d: entering lockstep loop",
                 _ep_rank, _ep_local_rank)
        try:
            tp_worker_loop(engine, _ep_world, _ep_rank)
        except KeyboardInterrupt:
            pass
        return

    # Legacy EP-only worker (TP==1, EP>1): no input broadcast yet — left
    # as the passive barrier placeholder.
    if args.ep > 1 and args.tp == 1 and _ep_rank != 0:
        log.info("EP worker rank=%d: model loaded, entering passive barrier loop", _ep_rank)
        try:
            import torch.distributed as _dist
            while True:
                _dist.barrier()
        except KeyboardInterrupt:
            pass
        return

    # Rank 0 path: monkey-patch inner_model.forward so every forward broadcasts
    # its inputs to TP workers before running locally.
    if args.tp > 1 and _ep_world > 1 and _ep_rank == 0:
        from engine.runtime.tp_worker import make_rank0_forward, broadcast_shutdown
        import atexit as _atexit
        runner = engine.runner
        inner = runner.inner_model
        orig_fwd = inner.forward
        inner.forward = make_rank0_forward(orig_fwd, _ep_world)
        log.info("TP rank0: patched inner_model.forward for lockstep broadcast (world=%d)", _ep_world)
        _atexit.register(broadcast_shutdown, _ep_world)

    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
