"""TP worker lockstep.

Rank 0 drives the scheduler and calls ``inner_model.forward(...)``. Worker
ranks (1..N-1) need to participate in the all_reduce / all_gather collectives
inside ``RowParallelLinear`` / MoE layers, otherwise rank 0 hangs on the very
first collective.

Mechanism (super simple, super dumb, works):

1. Rank 0's ``inner_model.forward`` is monkey-patched. On entry it broadcasts
   a small Python object describing the kwargs (which are tensors, their
   shapes + dtypes, and any non-tensor scalars/None values), then broadcasts
   each tensor in a fixed order. Then it runs the original forward locally.

2. Worker ranks call :func:`tp_worker_loop`, which loops:
     a. Receive the meta object via ``broadcast_object_list`` from src=0.
     b. Allocate empty tensor buffers per the meta and ``broadcast`` each.
     c. Call the *unpatched* ``inner_model.forward(**kwargs)`` so the worker
        participates in the same all_reduces. Output is discarded.
     d. ``"shutdown"`` cmd breaks the loop.

Caveats / known limitations:

* **Recursive forwards**: if the inner model recursively calls
  ``inner_model.forward`` from inside the patched forward (e.g. for a draft /
  speculative path), the broadcast would fire again and deadlock. We guard
  against this with a thread-local re-entrancy flag.
* **KV cache state per rank**: each rank owns its own KV cache. The cache
  update path inside the model runs on every rank with the SAME inputs (we
  broadcast them), so the per-rank caches stay byte-identical for replicated
  state and correctly sharded for column/row-parallel state. We do NOT
  broadcast the cache itself.
* **Non-tensor kwargs**: forwarded inside the meta object as-is (must be
  picklable). Booleans, ints, None, simple dicts are fine.
* **Device**: workers must already have ``torch.cuda.set_device(local_rank)``
  in effect (``load_model`` does this) so the empty buffers land on the
  right GPU.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import torch
import torch.distributed as dist


log = logging.getLogger("engine.runtime.tp_worker")


# Re-entrancy guard so a recursive call to inner_model.forward (e.g. from
# inside a layer that calls back into the top-level model) doesn't trigger
# a second broadcast. Per-thread because uvicorn worker threads can each
# drive forwards independently — but only one should be broadcasting at a
# time anyway (the engine serializes scheduler steps).
_in_patched_forward = threading.local()


def _is_in_patched_forward() -> bool:
    return getattr(_in_patched_forward, "active", False)


def _dtype_name(dtype: torch.dtype) -> str:
    # "torch.float16" -> "float16"
    return str(dtype).split(".", 1)[-1]


def _dtype_from_name(name: str) -> torch.dtype:
    return getattr(torch, name)


def _build_meta(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Walk kwargs and extract a picklable description.

    Returns a dict mapping kwarg name -> ("tensor", shape_tuple, dtype_name)
    or ("value", value) for non-tensors.
    """
    meta: dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            meta[k] = ("tensor", tuple(v.shape), _dtype_name(v.dtype))
        else:
            meta[k] = ("value", v)
    return meta


def make_rank0_forward(orig_forward, world_size: int):
    """Wrap an inner_model.forward so rank 0 broadcasts inputs to TP peers.

    The returned callable has the same signature as ``orig_forward`` (it is
    bound — it accepts ``*args, **kwargs`` and forwards them).
    """

    def patched(*args, **kwargs):
        # Re-entrancy: only the OUTERMOST call broadcasts.
        if _is_in_patched_forward() or world_size <= 1:
            return orig_forward(*args, **kwargs)

        # We do not support positional args here — the engine always calls
        # the model with kwargs (input_ids=, position_ids=, etc.). If a
        # caller passes positional args, fold them into kwargs is hard
        # without the original signature; assert loudly instead.
        if args:
            raise RuntimeError(
                "tp_worker: rank0 forward wrapper requires kwargs-only calls; "
                f"got {len(args)} positional args"
            )

        meta = _build_meta(kwargs)
        # Send command + meta as a single object.
        dist.broadcast_object_list([("forward", meta)], src=0)
        # Now broadcast each tensor in the order they appear in meta.
        for k, entry in meta.items():
            if entry[0] == "tensor":
                t = kwargs[k]
                if not t.is_contiguous():
                    t = t.contiguous()
                    kwargs[k] = t
                dist.broadcast(t, src=0)

        _in_patched_forward.active = True
        try:
            return orig_forward(**kwargs)
        finally:
            _in_patched_forward.active = False

    return patched


def tp_worker_loop(engine, world_size: int, rank: int) -> None:
    """Worker rank main loop. Runs forever until shutdown command received.

    Workers re-create the kwargs from the broadcast meta + tensors, then call
    the *raw* (unpatched) forward so they participate in collectives.
    """
    runner = engine.runner
    inner_model = runner.inner_model
    device = runner.device
    # Workers should call the original forward. We don't want to ever patch
    # the workers — they don't drive broadcasts, they receive them.
    raw_forward = inner_model.forward

    log.info(
        "tp_worker_loop start rank=%d world=%d device=%s",
        rank, world_size, device,
    )

    n = 0
    while True:
        obj_list: list[Any] = [None]
        dist.broadcast_object_list(obj_list, src=0)
        msg = obj_list[0]
        if msg is None:
            log.warning("tp_worker rank=%d got None message, treating as shutdown", rank)
            return
        cmd = msg[0]
        if cmd == "shutdown":
            log.info("tp_worker rank=%d shutdown", rank)
            return
        if cmd != "forward":
            log.warning("tp_worker rank=%d unknown cmd=%r, ignoring", rank, cmd)
            continue

        meta = msg[1]
        kwargs: dict[str, Any] = {}
        # IMPORTANT: iterate in the SAME order as rank 0 (dict order is
        # preserved across pickle for both ends since Py3.7).
        for k, entry in meta.items():
            if entry[0] == "tensor":
                _, shape, dtype_name = entry
                buf = torch.empty(shape, dtype=_dtype_from_name(dtype_name), device=device)
                dist.broadcast(buf, src=0)
                kwargs[k] = buf
            else:
                # ("value", v)
                kwargs[k] = entry[1]

        with torch.inference_mode():
            try:
                raw_forward(**kwargs)
            except Exception:
                log.exception("tp_worker rank=%d forward failed", rank)
                raise
        n += 1
        if n % 256 == 0:
            log.info("tp_worker rank=%d processed %d forwards", rank, n)


def broadcast_shutdown(world_size: int) -> None:
    """Rank 0 calls this on engine teardown to release the workers."""
    if world_size <= 1:
        return
    try:
        dist.broadcast_object_list([("shutdown", None)], src=0)
    except Exception:
        log.exception("broadcast_shutdown failed (workers may already be gone)")
