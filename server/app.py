"""FastAPI app exposing the OpenAI-compatible chat completions endpoint.

The app holds a single `Engine` instance constructed at startup. The
Engine is the stable boundary — Phase 0 plugs in a stub, Phase 1+ swaps
in the real generate path with no server-side changes required.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response

from engine.runtime.engine import Engine
from engine.runtime.metrics import metrics
from engine.runtime.profiling import print_region_stats, timer
from server.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    Usage,
)


def create_app(engine: Engine) -> FastAPI:
    app = FastAPI(title="paris-hackathon-2026-inference", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def get_metrics() -> dict:
        return metrics.snapshot()

    @app.get("/metrics/regions", response_class=Response)
    async def get_regions(sort_by: str = "total") -> Response:
        return Response(content=timer.format_table(sort_by=sort_by), media_type="text/plain")

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        if req.max_tokens is None or req.max_tokens <= 0:
            raise HTTPException(status_code=400, detail="max_tokens must be > 0")

        messages = [m.model_dump() for m in req.messages]
        try:
            result = await engine.generate(
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
        except NotImplementedError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:  # surface engine failures as 500
            raise HTTPException(status_code=500, detail=f"engine error: {e}")

        return ChatCompletionResponse(
            id=Engine.make_request_id(),
            created=Engine.now_unix(),
            model=req.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=result.text),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
        )

    @app.on_event("startup")
    async def _on_startup() -> None:
        # Spawn the batcher loop on the server event loop so the first
        # request doesn't pay the cold-start cost.
        if not engine.stub:
            await engine.start()

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        if not engine.stub:
            await engine.stop()
        # Dump the final region stats so the operator sees the breakdown
        # of the run that just ended without having to curl.
        print_region_stats()

    return app
