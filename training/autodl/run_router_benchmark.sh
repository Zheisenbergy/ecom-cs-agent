#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$PROJECT_ROOT/.venv/bin/python"}

MODEL_NAME=${MODEL_NAME:-router-lora}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
API_KEY=${API_KEY:-EMPTY}
OUTPUT_PATH=${OUTPUT_PATH:-"$PROJECT_ROOT/training/datasets/router_benchmark.autodl.json"}

"$PYTHON_BIN" -m app.cli benchmark-router \
  --input "$PROJECT_ROOT/training/datasets/router_sft.dev.generated.jsonl" \
  --model "$MODEL_NAME" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --output "$OUTPUT_PATH"
