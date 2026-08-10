import os
import pathlib
import time

import torch


class Profiler:
    def __init__(self, out_dir: str):
        self.dir = out_dir
        self._profiler = None

    def start(self, record_shapes=False, with_stack=False):
        if self._profiler is not None:
            print("Profiler already in progress")
            return
        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=record_shapes,
            with_stack=with_stack
        )
        self._profiler.start()

    def stop(self) -> str | None:
        if self._profiler is None:
            print("Profiler not running")
            return None
        self._profiler.stop()
        pathlib.Path(self.dir).mkdir(parents=True, exist_ok=True)
        out = os.path.join(self.dir, f"trace-{int(time.time())}.json.gz")
        self._profiler.export_chrome_trace(out) # view in ui.perfetto.dev / chrome://tracing
        print(f"trace: {out}")
        self._profiler = None
        return out


