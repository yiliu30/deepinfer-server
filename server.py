"""Anthropic Messages gateway for DeepInfer, usable by Claude Code.

Claude Code talks the Anthropic Messages API. This module converts each
request with deepseek-recipe, feeds the rendered prompt to a DeepInfer engine
pool, and streams model tokens back as Anthropic SSE events.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from anyio import CancelScope
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.types import Receive, Scope, Send

from deepseek_recipe import (
    ConversionOptions,
    ConversationRequest,
    DeepseekV41Encoding,
    InferenceChunk,
    InferenceFinishReason,
    MessagesRequest,
    MessagesResponse,
    PromptUsage,
    StreamProcessor,
    Tokenizer,
)

from engine_pool import EnginePool


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


MODEL_DIR = _required("GATEWAY_MODEL")
REPO_DIR = _required("DEEPINFER_REPO")
DEVICES = [int(x) for x in os.environ.get("GATEWAY_GPUS", "0,1,2,3").split(",") if x.strip()]
EP = int(os.environ.get("GATEWAY_EP", len(DEVICES)))
DP = int(os.environ.get("GATEWAY_DP", len(DEVICES)))
MAX_MODEL_LEN = int(os.environ.get("GATEWAY_MAX_MODEL_LEN", "1048576"))
MAX_BATCHED = int(os.environ.get("GATEWAY_MAX_NUM_BATCHED_TOKENS", "32768"))
MAX_SEQS = int(os.environ.get("GATEWAY_MAX_NUM_SEQS", "4"))
GPU_UTIL = float(os.environ.get("GATEWAY_GPU_MEMORY_UTILIZATION", "0.85"))
LOOKAHEAD = int(os.environ.get("GATEWAY_LOOKAHEAD_TOKENS", "1"))
SAMPLER_TOPK = int(os.environ.get("GATEWAY_SAMPLER_FORCE_TOPK", "4096"))
SERVED_MODEL = os.environ.get("GATEWAY_SERVED_MODEL", "DeepSeek-V4.1-Flash")
API_KEY = os.environ.get("GATEWAY_API_KEY", "")
THINKING_DEFAULT = os.environ.get("GATEWAY_THINKING", "1") not in {"0", "false", "False"}

os.environ.setdefault("MASTER_PORT", "24555")

TOKENIZER = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
ENCODING = DeepseekV41Encoding().with_tokenizer(TOKENIZER)
OPTIONS = ConversionOptions(default_thinking_mode=THINKING_DEFAULT)

POOL = EnginePool(
    {
        "model": MODEL_DIR,
        "repo": REPO_DIR,
        "devices": DEVICES,
        "ep": EP,
        "dp": DP,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_BATCHED,
        "max_num_seqs": MAX_SEQS,
        "gpu_memory_utilization": GPU_UTIL,
        "lookahead_tokens": LOOKAHEAD,
        "sampler_force_topk": SAMPLER_TOPK,
    }
)

READY = threading.Event()
START_ERROR: str | None = None


class RequestError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Prepared:
    conversation_request: ConversationRequest
    prompt: str
    response_model: str
    stream: bool


def prepare(body: bytes) -> Prepared:
    try:
        data = json.loads(body)
    except ValueError as error:
        raise RequestError(f"invalid JSON body: {error}") from error
    if not isinstance(data, dict):
        raise RequestError("request body must be a JSON object")
    if not data.get("model"):
        data["model"] = SERVED_MODEL
    requested_model = str(data["model"])
    # deepseek-recipe resolves prompts by model name; accept any Claude Code
    # model id and map it to the served checkpoint.
    data["model"] = SERVED_MODEL
    try:
        request = MessagesRequest(data)
        converted = request.convert(OPTIONS)
    except Exception as error:
        raise RequestError(str(error)) from error
    rendered = ENCODING.render_conversation(converted.conversation)
    if rendered.image_sources:
        raise RequestError("image inputs are not enabled on this gateway yet")
    return Prepared(converted, rendered.prompt, requested_model, bool(converted.stream))


def _finish_reason(reason: str) -> InferenceFinishReason:
    reason = (reason or "").lower()
    if "length" in reason or "max" in reason:
        return InferenceFinishReason.Length
    if "abort" in reason or "filter" in reason:
        return InferenceFinishReason.ContentFilter
    return InferenceFinishReason.Stop


async def infer(prepared: Prepared) -> AsyncIterator[InferenceChunk]:
    converted = prepared.conversation_request
    prompt_ids = ENCODING.encode(converted.conversation)
    options = converted.inference_options
    max_tokens = converted.resolve_max_tokens(len(prompt_ids), MAX_MODEL_LEN)
    sampling_params = {
        "temperature": float(options.temperature if options.temperature is not None else 1.0),
        "top_p": float(options.top_p if options.top_p is not None else 1.0),
        "max_tokens": int(max_tokens),
    }
    if options.seed is not None:
        sampling_params["seq_seed"] = int(options.seed)

    request_id, token_queue = POOL.submit(prompt_ids, sampling_params)
    try:
        yield InferenceChunk.ready(prompt_usage=PromptUsage(prompt_tokens=len(prompt_ids)))
        while True:
            item = await run_in_threadpool(token_queue.get)
            if item is None:
                yield InferenceChunk.finish(InferenceFinishReason.Stop)
                break
            yield InferenceChunk.tokens([item["token_id"]])
            if item["finished"]:
                yield InferenceChunk.finish(_finish_reason(item["finish_reason"]))
                break
    finally:
        POOL.release(request_id, abort=True)


def sse_frame(event: str | None, data: str) -> str:
    prefix = f"event: {event}\n" if event is not None else ""
    return f"{prefix}data: {data}\n\n"


async def response_body(prepared: Prepared) -> AsyncIterator[str]:
    response_id = str(uuid.uuid4())
    model = prepared.response_model or SERVED_MODEL
    generator = MessagesRequest.chunk_generator(
        prepared.conversation_request, response_id, model
    )
    processor = StreamProcessor(
        generator, prepared.conversation_request.parsing_options, TOKENIZER
    )
    response = None if prepared.stream else MessagesResponse(response_id, model, int(time.time()), 0, 0)
    inference = None
    try:
        inference = aiter(infer(prepared))
        while True:
            try:
                chunk = await anext(inference)
            except StopAsyncIteration:
                chunks = processor.finish()
            else:
                chunks = processor.push(chunk)
            for output in chunks:
                if prepared.stream:
                    yield sse_frame(MessagesResponse.sse_event_name(output), output.to_json())
                else:
                    response.accumulate(output)
            if processor.finished:
                break
        if prepared.stream:
            done = MessagesResponse.sse_done_data()
            if done is not None:
                yield sse_frame(None, done)
        else:
            yield response.to_json()
    finally:
        try:
            close = getattr(inference, "aclose", None)
            if close is not None:
                with CancelScope(shield=True):
                    await close()
        finally:
            processor.close()


class InferenceStreamingResponse(StreamingResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                await self.body_iterator.aclose()


def _error(message: str, status_code: int, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": kind, "message": message}},
    )


def _check_auth(request: Request) -> None:
    if not API_KEY:
        return
    key = request.headers.get("x-api-key", "")
    if not key:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            key = auth[7:]
    if key != API_KEY:
        raise RequestError("invalid API key", 401, )


def _start_pool() -> None:
    global START_ERROR
    try:
        POOL.start()
        READY.set()
    except Exception:
        START_ERROR = traceback.format_exc()


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_start_pool, name="engine-start", daemon=True).start()
    yield
    POOL.close()


app = FastAPI(title="DeepInfer Claude Code Gateway", lifespan=lifespan)


def _gate() -> JSONResponse | None:
    if START_ERROR is not None:
        return _error(f"engine failed to start:\n{START_ERROR}", 503, "api_error")
    if not READY.is_set():
        return _error("engine is still loading the model", 503, "overloaded_error")
    if POOL.error is not None:
        return _error(f"engine error:\n{POOL.error}", 503, "api_error")
    return None


@app.get("/health")
async def health():
    return {
        "ready": READY.is_set() and START_ERROR is None and POOL.error is None,
        "model": SERVED_MODEL,
        "gpus": DEVICES,
        "ep": EP,
        "dp": DP,
        "max_model_len": MAX_MODEL_LEN,
        "start_error": START_ERROR,
        "engine_error": POOL.error,
    }


@app.get("/v1/models")
async def models():
    return {
        "data": [
            {
                "type": "model",
                "id": SERVED_MODEL,
                "display_name": SERVED_MODEL,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ],
        "has_more": False,
        "first_id": SERVED_MODEL,
        "last_id": SERVED_MODEL,
    }


@app.post("/v1/messages")
async def messages(request: Request):
    try:
        _check_auth(request)
    except RequestError as error:
        return _error(str(error), error.status_code)
    gate = _gate()
    if gate is not None:
        return gate
    try:
        prepared = await run_in_threadpool(prepare, await request.body())
    except RequestError as error:
        return _error(str(error), error.status_code)
    output = response_body(prepared)
    if prepared.stream:
        return InferenceStreamingResponse(output, media_type="text/event-stream")
    try:
        content = "".join([part async for part in output])
    except Exception:
        return _error(traceback.format_exc(), 500, "api_error")
    return Response(content, media_type="application/json")


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    try:
        _check_auth(request)
    except RequestError as error:
        return _error(str(error), error.status_code)
    try:
        prepared = await run_in_threadpool(prepare, await request.body())
        prompt_ids = ENCODING.encode(prepared.conversation_request.conversation)
    except RequestError as error:
        return _error(str(error), error.status_code)
    return {"input_tokens": len(prompt_ids)}
