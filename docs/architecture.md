# Architecture

Three pieces cooperate: **DeepInfer** runs the model, **deepseek-recipe** owns
the DeepSeek request/response format, and **deepinfer-server** connects them to
an HTTP client.

## Component view

```mermaid
flowchart TB
    CC["Client<br/>Claude Code or any Anthropic Messages client"]

    subgraph SERVER["deepinfer-server"]
        API["API layer: FastAPI<br/>/v1/messages · /count_tokens · /health"]
        POOL["Engine pool<br/>rank orchestration + token routing"]
    end

    subgraph RECIPE["deepseek-recipe"]
        CONV["Request conversion<br/>MessagesRequest.convert"]
        RENDER["Prompt rendering and tokenization<br/>DeepseekV41Encoding"]
        STREAM["Stream parsing<br/>StreamProcessor"]
    end

    subgraph ENGINE["DeepInfer"]
        RANKS["Engine ranks<br/>EP=4 · DP=4"]
        WEIGHTS["Weights and KV cache<br/>one rank per GPU"]
    end

    CC -->|"POST /v1/messages"| API
    API -->|"request body"| CONV
    CONV -->|"Conversation"| RENDER
    RENDER -->|"prompt token IDs + sampling options"| POOL
    POOL -->|"add_request / step"| RANKS
    RANKS --- WEIGHTS
    RANKS -->|"token IDs"| POOL
    POOL -->|"InferenceChunk.tokens"| STREAM
    STREAM -->|"thinking / text / tool_use events"| API
    API -->|"JSON or SSE"| CC
```

## Request flow

```mermaid
sequenceDiagram
    participant CC as Client
    participant API as deepinfer-server API
    participant R as deepseek-recipe
    participant E as DeepInfer engine pool

    CC->>API: POST /v1/messages
    API->>R: convert request, render prompt, tokenize
    R-->>API: prompt token IDs, sampling and parsing options
    API->>E: add_request on one data-parallel rank
    loop engine steps, all ranks in lockstep
        E-->>API: generated token IDs
        API->>R: StreamProcessor.push(tokens)
        R-->>API: Anthropic SSE events
        API-->>CC: content_block_delta and friends
    end
    E-->>API: finished
    API->>R: StreamProcessor.finish()
    R-->>API: message_delta, message_stop
    API-->>CC: end of stream
```

## Responsibilities

| Component | Owns | Does not do |
| --- | --- | --- |
| DeepInfer | Model execution, weights, KV cache, expert/data parallelism, sampling, speculative decoding | HTTP transport, Anthropic protocol details |
| deepseek-recipe | Messages/Chat/Responses conversion, DeepSeek prompt rendering, tokenization, streaming output parsing, tool calls, thinking blocks | Model inference, request routing, HTTP transport |
| deepinfer-server | HTTP endpoints, SSE framing, rank orchestration, request-to-rank routing, readiness and error reporting | Model kernels, prompt format details |

## Why the split

- deepseek-recipe has no inference backend and no HTTP layer; it turns protocol
  objects into DeepSeek prompts and model output back into protocol objects.
- DeepInfer has no client API; it exposes request-level engine calls
  (`add_request`, `step`, `abort_request`) but no transport.
- deepinfer-server is the thin layer that adds transport, multi-rank scheduling,
  and streaming, while leaving model and format concerns to the other two.

## GPU layout

The engine pool starts one process per GPU. Each process owns one DeepInfer
engine rank and one visible GPU, all ranks join one expert-parallel group, and
the pool calls `step` on every rank for each scheduling round. Ranks without
requests execute a placeholder step so expert-parallel collectives always have
all participants.
