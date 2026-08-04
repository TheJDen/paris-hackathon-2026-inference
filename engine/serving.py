import asyncio
import engine.engine
import engine.records
import fastapi
import pydantic
import queue
import time

from contextlib import asynccontextmanager


class ChatMessage(pydantic.BaseModel):
    role: str
    content: str

class ChatCompletionRequest(pydantic.BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    model_config = pydantic.ConfigDict(extra="ignore")

class Choice(pydantic.BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(pydantic.BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(pydantic.BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    created: int
    choices: list[Choice]
    usage: Usage


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    async_engine = engine.engine.AsyncEngine()
    async_engine.start()
    app.state.async_engine = async_engine
    yield

app = fastapi.FastAPI(lifespan=lifespan)

@app.get("/health")
async def health(request: fastapi.Request):
    async_engine: engine.engine.AsyncEngine = request.app.state.async_engine
    if not async_engine.ready:
        return fastapi.Response(status_code=503)
    return fastapi.Response(status_code=200)

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: fastapi.Request):
    async_engine: engine.engine.AsyncEngine = request.app.state.async_engine

    cc_req = engine.records.CompletionRequest(
            messages = [m.model_dump() for m in req.messages],
            max_tokens = req.max_tokens,
            temperature = req.temperature,
            top_p = req.top_p,
    )

    try:
        completion = await asyncio.wait_for(async_engine.generate(cc_req), timeout=600)
    except queue.Full:
        raise fastapi.HTTPException(
                status_code=503,
                detail="Engine queue full"
                )
    except asyncio.TimeoutError:
        raise fastapi.HTTPException(
                status_code=504,
                detail="Generation timed out, the whole server is cooked"
                )
    except asyncio.CancelledError:
        raise fastapi.HTTPException(
                status_code=504,
                detail="Generation cancelled, the whole server is cooked"
                )

    return ChatCompletionResponse(
        id=f"chatcmpl-{completion.request_id}",
        model=req.model,
        created=int(time.time()),
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=completion.text),
                finish_reason=completion.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.prompt_tokens + completion.completion_tokens
        ),
    )

@app.post("/start_profile")
def start_profile(request: fastapi.Request):
    try:
        request.app.state.async_engine.queue_start_profile()
    except queue.Full:
        raise fastapi.HTTPException(503, "engine queue full; retry")
    except RuntimeError as e:
        raise fastapi.HTTPException(404, str(e))
    return {"status" :"started"}


@app.post("/stop_profile")
def stop_profile(request: fastapi.Request):
    try:
        request.app.state.async_engine.queue_stop_profile()
    except queue.Full:
          raise fastapi.HTTPException(503, "engine queue full; retry")
    except RuntimeError as e:
        raise fastapi.HTTPException(404, str(e))
    return {"status" :"stopping"}

