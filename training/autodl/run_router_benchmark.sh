#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$PROJECT_ROOT/.venv/bin/python"}

MODEL_NAME=${MODEL_NAME:-router-lora}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
API_KEY=${API_KEY:-EMPTY}
BENCHMARK_VARIANT=${BENCHMARK_VARIANT:-seed}

if [[ -z "${INPUT_PATH:-}" ]]; then
  case "$BENCHMARK_VARIANT" in
    mixed)
      INPUT_PATH="$PROJECT_ROOT/training/datasets/router_sft.dev.mixed.generated.jsonl"
      ;;
    holdout)
      INPUT_PATH="$PROJECT_ROOT/training/datasets/router_sft.dev.holdout.jsonl"
      ;;
    *)
      INPUT_PATH="$PROJECT_ROOT/training/datasets/router_sft.dev.generated.jsonl"
      ;;
  esac
fi

if [[ -z "${OUTPUT_PATH:-}" ]]; then
  case "$BENCHMARK_VARIANT" in
    mixed)
      OUTPUT_PATH="$PROJECT_ROOT/training/datasets/router_benchmark.autodl.mixed.json"
      ;;
    holdout)
      OUTPUT_PATH="$PROJECT_ROOT/training/datasets/router_benchmark.autodl.holdout.json"
      ;;
    *)
      OUTPUT_PATH="$PROJECT_ROOT/training/datasets/router_benchmark.autodl.json"
      ;;
  esac
fi

"$PYTHON_BIN" -m app.cli benchmark-router \
  --input "$INPUT_PATH" \
  --model "$MODEL_NAME" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --output "$OUTPUT_PATH"
