from collections.abc import Callable

import torch


class EventFuture:
    def __init__(self, *, event: torch.cuda.Event, out: torch.Tensor):
        self.event = event
        self.event.record()
        self.out_buffer = out

    def result(self):
        self.event.synchronize()
        return self.out_buffer

    def done(self) -> bool:
        return self.event.query()

class EventExecutor:
    def __init__(self, *, num_events: int, out_capacity: int, dtype):
        self.events = [torch.cuda.Event() for _ in range(num_events)]
        self.ring = torch.empty(
            num_events,
            out_capacity,
            dtype=dtype,
            pin_memory=True
        )
        self.i = 0
        self.n = num_events 

    def submit(self, work: Callable[[], torch.Tensor]) -> EventFuture:
        event, out_buffer = self.events[self.i], self.ring[self.i]
        self.i = (self.i + 1) % self.n
        result = work()
        out_buffer[:result.shape[0]].copy_(result, non_blocking=True)
        return EventFuture(event=event, out=out_buffer[:result.shape[0]])

class CudaGraphs:
    def __init__(self, keys: list[int]):
        self.graphs = {key: torch.cuda.CUDAGraph() for key in keys}
        self.pool = torch.cuda.graph_pool_handle()

    def capture(self, key: int, thunk: Callable[[], None]):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                thunk()
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(self.graphs[key], pool=self.pool):
            thunk()

    def replay(self, key: int):
        self.graphs[key].replay()
