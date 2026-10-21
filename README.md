# DeepInfer Claude Code gateway

Anthropic Messages API server backed by a DeepInfer inference engine. It lets
Claude Code use a locally served DeepSeek model through `ANTHROPIC_BASE_URL`.

## Layout

- `engine_pool.py` - one DeepInfer engine process per GPU, driven in lockstep;
  expert-parallel collectives stay aligned even when only some ranks have work.
- `server.py` - FastAPI gateway: `/v1/messages` (JSON and SSE),
  `/v1/messages/count_tokens`, `/v1/models`, `/health`.
- `start.sh` - loads the DeepInfer environment and starts uvicorn.
- `claude-env.sh` - exports the Claude Code environment and helpers.
- `smoke_test.sh` - non-stream, stream, tool-call, and token-count checks.

## Requirements

- A built DeepInfer checkout and virtualenv with its NVIDIA dependencies.
- A local DeepSeek-V4.1-Flash checkpoint.
- Four GPUs in one NVSwitch domain; expert parallelism is set to four by default.
- `deepseek-recipe` 0.2.0 or newer (see `requirements.txt`).

## Start

```bash
export DEEPINFER_ENV=/path/to/venv/bin/activate
export DEEPINFER_REPO=/path/to/DeepInfer
export GATEWAY_MODEL=/path/to/DeepSeek-V4.1-Flash-1017
export GATEWAY_GPUS=0,1,2,3
./start.sh
```

Model loading takes a few minutes. `/health` reports `"ready": true` once all
ranks are up. Other knobs: `PORT` (default `8000`), `HOST`, `MASTER_PORT`,
`GATEWAY_MAX_MODEL_LEN` (default `65536`), `GATEWAY_EP`, `GATEWAY_DP`,
`GATEWAY_GPU_MEMORY_UTILIZATION`, `GATEWAY_THINKING`, `GATEWAY_API_KEY`.

Keep the four GPU indices on one NUMA/NIC rail set. A mixed set can make DeepEP
pick mismatched RDMA devices and fail during startup.

## Use With Claude Code

```bash
export GATEWAY_BASE_URL=http://gateway-host:8000
source claude-env.sh
cc
```

`claude-env.sh` points all Claude model aliases at the served model, sets the
context window to the engine's `max_model_len`, and adds `gateway-health`.
Requests are rewritten to the served checkpoint, so any Claude model id works.

## Notes

- The gateway speaks the Anthropic Messages API only; OpenAI `/v1/chat/completions`
  and `/v1/responses` are not implemented.
- Text input only for now; image blocks return HTTP 400.
- DeepInfer has no prefix cache, so each turn re-prefills its full context.
- The first request on each rank pays kernel JIT cost; later turns are faster.
- Verified with Claude Code 2.1.304, including streamed tool calls and a
  multi-turn write/read file task.
