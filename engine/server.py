import asyncio
import fastapi
import pydantic
import queue
import time

from contextlib import asynccontextmanager

import engine.client

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
    client = engine.client.Client()
    client.start()
    app.state.client = client

    yield

    await client.shutdown()

app = fastapi.FastAPI(lifespan=lifespan)

@app.get("/health")
async def health(request: fastapi.Request): # technically we can return 200 bc lifespan but this doesn't hurt to have if we move to pods
    client: engine.client.Client = request.app.state.client
    if not client.ready:
        return fastapi.Response(status_code=503)
    return fastapi.Response(status_code=200)

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: fastapi.Request):
    client: engine.client.Client = request.app.state.client

    cc_req = engine.client.ChatCompletionRequest(
            messages = [m.model_dump() for m in req.messages],
            max_tokens = req.max_tokens,
            temperature = req.temperature,
            top_p = req.top_p,
    )

    try:
        request_id, future = client.submit(cc_req)
    except queue.Full:
        raise fastapi.HTTPException(
                status_code=503,
                detail="Engine queue full"
                )

    try:
        result = await asyncio.wait_for(future, timeout=600)
    except asyncio.TimeoutError:
        client.cancel_id(request_id)
        raise fastapi.HTTPException(
                status_code=504,
                detail="Generation timed out"
                )
    except asyncio.CancelledError:
        client.cancel_id(request_id)
        raise

    return ChatCompletionResponse(
        id=f"chatcmpl-{result['request_id']}",
        model=req.model,
        created=int(time.time()),
        choices=[
            Choice(
                index=0,
                message=ChatMessage(role="assistant", content=result["text"]),
                finish_reason=result["finish_reason"],
            )
        ],
        usage=Usage(**result["usage"]),
    )

@app.post("/start_profile")
def start_profile(request: fastapi.Request):
    try:
        request.app.state.client.queue_start_profile()
    except queue.Full:
        raise fastapi.HTTPException(503, "engine queue full; retry")
    except RuntimeError as e:
        raise fastapi.HTTPException(404, str(e))
    return {"status" :"started"}


@app.post("/stop_profile")
def stop_profile(request: fastapi.Request):
    try:
        request.app.state.client.queue_stop_profile()
    except queue.Full:
          raise fastapi.HTTPException(503, "engine queue full; retry")
    except RuntimeError as e:
        raise fastapi.HTTPException(404, str(e))
    return {"status" :"stopping"}

