# Claude Code against a DeepInfer-backed Anthropic Messages gateway.
#
#   export GATEWAY_BASE_URL=http://gateway-host:8000
#   source claude-env.sh
#   cc
#
# The gateway is started by start.sh from a DeepInfer checkout; see README.md.

: "${GATEWAY_BASE_URL:?set GATEWAY_BASE_URL, e.g. http://gateway-host:8000}"

export PATH="$HOME/.local/bin:$PATH"
export no_proxy="127.0.0.1,localhost${GATEWAY_NO_PROXY:+,$GATEWAY_NO_PROXY}"
export NO_PROXY="$no_proxy"
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

DS_MODEL="${GATEWAY_MODEL_ID:-DeepSeek-V4.1-Flash}"
export ANTHROPIC_BASE_URL="$GATEWAY_BASE_URL"
export ANTHROPIC_MODEL="$DS_MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$DS_MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$DS_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$DS_MODEL"
export ANTHROPIC_SMALL_FAST_MODEL="$DS_MODEL"

# The gateway accepts any non-empty key unless GATEWAY_API_KEY is set on the
# server side.
export ANTHROPIC_API_KEY=dummy
export ANTHROPIC_AUTH_TOKEN=dummy

# Keep Claude Code's assumed window in sync with the engine's max_model_len.
# Without this it assumes 200k for an unrecognized model id and overfills the
# server before auto-compact fires.
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="${GATEWAY_MAX_CONTEXT_TOKENS:-65536}"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

# DeepSeek-V4.1 uses a numeric reasoning budget: low=50, high=75, max=100.
export CLAUDE_CODE_EFFORT_LEVEL="${GATEWAY_EFFORT:-max}"

alias cc="IS_SANDBOX=1 claude --allow-dangerously-skip-permissions --permission-mode bypassPermissions"
alias ct="copilot --allow-all"
alias gateway-health="curl -s \"$GATEWAY_BASE_URL/health\"; echo"

# Codex is intentionally not wired here: the gateway exposes the Anthropic
# Messages API only, not OpenAI /v1/chat/completions or /v1/responses.
