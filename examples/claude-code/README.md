# Claude Code

Claude Code speaks the Anthropic Messages API, so it can use the server
directly through `ANTHROPIC_BASE_URL`.

## Prerequisites

Start the server first and confirm it is ready:

```bash
curl -s http://gateway-host:8000/health
```

## Setup

```bash
export GATEWAY_BASE_URL=http://gateway-host:8000
source examples/claude-code/claude-env.sh
cc
```

`claude-env.sh` points every Claude model alias at the served model and adds
two helpers: `cc` (Claude Code with permissions bypassed) and
`gateway-health`.

Optional overrides before sourcing:

- `GATEWAY_MODEL_ID` - model id shown to Claude Code (default
  `DeepSeek-V4.1-Flash`).
- `GATEWAY_MAX_CONTEXT_TOKENS` - assumed context window; keep it equal to the
  server's `GATEWAY_MAX_MODEL_LEN` (default `65536`).
- `GATEWAY_EFFORT` - reasoning effort: `low`, `high`, or `max` (default
  `max`).
- `GATEWAY_NO_PROXY` - extra hosts to bypass for HTTP proxies.

## Notes

- Claude Code prints a harmless `[claude-code:unrecognized_model]` telemetry
  line for the custom model id.
- Setting `CLAUDE_CODE_MAX_CONTEXT_TOKENS` matters: without it Claude Code
  assumes 200k for an unrecognized id and can overfill the server before
  auto-compact fires.
- Codex is not wired here because the server does not expose OpenAI-style
  endpoints.
