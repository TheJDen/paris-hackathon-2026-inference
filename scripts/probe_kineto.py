"""Does the engine's execution CONTEXT suppress capture? (cheap, no full model)

Kernels + structure all capture fine in isolation. The last untested difference from
the engine is the context forwards run under: inference_mode / no_grad / a non-default
CUDA stream. Profile the same work under each; whichever flips to EMPTY is the cause.
"""
import contextlib
import os

import torch


def make_profiler():
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False)


def report(p, tag):
    ka = p.key_averages()
    cuda_ev = [e for e in ka
               if getattr(e, "self_device_time_total", 0) or getattr(e, "cuda_time_total", 0)]
    launches = sum(getattr(e, "count", 0) for e in cuda_ev)
    print(f"[{tag}] {'REAL ' if cuda_ev else 'EMPTY'} kernel_types={len(cuda_ev)} "
          f"captured_launches={launches}", flush=True)


def work():
    x = torch.randn(512, 512, device="cuda")
    w = torch.randn(512, 512, device="cuda")
    for _ in range(100):
        x = torch.nn.functional.silu(x @ w)
    torch.cuda.synchronize()


def profile_under(ctx_factory, tag):
    p = make_profiler(); p.start()
    with ctx_factory():
        work()
    p.stop()
    report(p, tag)


torch.zeros(1, device="cuda")
torch.cuda.synchronize()
print("torch:", torch.__version__, flush=True)

profile_under(contextlib.nullcontext, "plain")
profile_under(torch.no_grad, "no_grad")
profile_under(torch.inference_mode, "inference_mode")

# non-default CUDA stream
def _stream_ctx():
    return torch.cuda.stream(torch.cuda.Stream())
profile_under(_stream_ctx, "side_stream")

# inference_mode + side stream (closest to a real inference engine)
@contextlib.contextmanager
def _infer_and_stream():
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        yield
profile_under(_infer_and_stream, "inference_mode+side_stream")

print("=== done ===", flush=True)
