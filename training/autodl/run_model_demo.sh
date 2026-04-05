#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$PROJECT_ROOT/.venv/bin/python"}

COMMAND=${COMMAND:-ask-model}
QUERY=${QUERY:-A1001 到哪了}
SHOP_ID=${SHOP_ID:-demo-shop}
ROUTER_MODEL=${ROUTER_MODEL:-router-lora}
ANSWER_MODEL=${ANSWER_MODEL:-answer-lora}
ROUTER_BASE_URL=${ROUTER_BASE_URL:-http://127.0.0.1:8000/v1}
ANSWER_BASE_URL=${ANSWER_BASE_URL:-http://127.0.0.1:8001/v1}
ROUTER_API_KEY=${ROUTER_API_KEY:-EMPTY}
ANSWER_API_KEY=${ANSWER_API_KEY:-EMPTY}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-120}
ROUTER_MAX_TOKENS=${ROUTER_MAX_TOKENS:-256}
ANSWER_MAX_TOKENS=${ANSWER_MAX_TOKENS:-256}
PRODUCT_ID=${PRODUCT_ID:-}
ORDER_ID=${ORDER_ID:-}

ARGS=(
  -m app.cli
  "$COMMAND"
  --shop-id "$SHOP_ID"
  --router-model "$ROUTER_MODEL"
  --answer-model "$ANSWER_MODEL"
  --router-base-url "$ROUTER_BASE_URL"
  --answer-base-url "$ANSWER_BASE_URL"
  --router-api-key "$ROUTER_API_KEY"
  --answer-api-key "$ANSWER_API_KEY"
  --timeout-seconds "$TIMEOUT_SECONDS"
  --router-max-tokens "$ROUTER_MAX_TOKENS"
  --answer-max-tokens "$ANSWER_MAX_TOKENS"
)

if [[ "$COMMAND" != "chat-model" ]]; then
  ARGS+=("$QUERY")
fi

if [[ -n "$PRODUCT_ID" ]]; then
  ARGS+=(--product-id "$PRODUCT_ID")
fi

if [[ -n "$ORDER_ID" ]]; then
  ARGS+=(--order-id "$ORDER_ID")
fi

if [[ "${JSON_OUTPUT:-0}" == "1" ]]; then
  ARGS+=(--json)
fi

if [[ "${SHOW_DEBUG:-0}" == "1" ]]; then
  ARGS+=(--show-debug)
fi

"$PYTHON_BIN" "${ARGS[@]}"
