#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"

echo "=== health ==="
curl -s "$BASE/health"
echo

echo "=== non-stream text ==="
curl -s -X POST "$BASE/v1/messages" \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-sonnet-4-5","max_tokens":64,"messages":[{"role":"user","content":"Reply with exactly: hello from deepinfer"}]}'
echo

echo "=== stream text ==="
curl -sN -X POST "$BASE/v1/messages" \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-sonnet-4-5","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"Count from 1 to 5."}]}' | tail -20

echo "=== tool call ==="
curl -s -X POST "$BASE/v1/messages" \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-sonnet-4-5","max_tokens":256,"messages":[{"role":"user","content":"What is the weather in Vancouver? Use the tool."}],"tools":[{"name":"get_weather","description":"Get weather for a city","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],"tool_choice":{"type":"auto"}}'
echo

echo "=== count tokens ==="
curl -s -X POST "$BASE/v1/messages/count_tokens" \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-sonnet-4-5","messages":[{"role":"user","content":"Hello"}]}'
echo
