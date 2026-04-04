#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$PROJECT_ROOT/.venv/bin/python"}

MODEL_NAME=${MODEL_NAME:-answer-lora}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000/v1}
API_KEY=${API_KEY:-EMPTY}
BENCHMARK_VARIANT=${BENCHMARK_VARIANT:-seed}

if [[ -z "${INPUT_PATH:-}" ]]; then
  if [[ "$BENCHMARK_VARIANT" == "mixed" ]]; then
    INPUT_PATH="$PROJECT_ROOT/training/datasets/answer_sft.dev.mixed.generated.jsonl"
  else
    INPUT_PATH="$PROJECT_ROOT/training/datasets/answer_sft.dev.generated.jsonl"
  fi
fi

if [[ -z "${OUTPUT_PATH:-}" ]]; then
  if [[ "$BENCHMARK_VARIANT" == "mixed" ]]; then
    OUTPUT_PATH="$PROJECT_ROOT/training/datasets/answer_benchmark.autodl.mixed.json"
  else
    OUTPUT_PATH="$PROJECT_ROOT/training/datasets/answer_benchmark.autodl.json"
  fi
fi

"$PYTHON_BIN" -m app.cli benchmark-answer \
  --input "$INPUT_PATH" \
  --model "$MODEL_NAME" \
  --base-url "$BASE_URL" \
  --api-key "$API_KEY" \
  --output "$OUTPUT_PATH"
