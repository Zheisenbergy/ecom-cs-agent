#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B-Instruct}
API_KEY=${API_KEY:-EMPTY}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

if [[ -z "${ADAPTER_ALIAS:-}" || -z "${ADAPTER_PATH:-}" ]]; then
  echo "ADAPTER_ALIAS and ADAPTER_PATH are required."
  exit 1
fi

vllm serve "$BASE_MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --enable-lora \
  --lora-modules "$ADAPTER_ALIAS=$ADAPTER_PATH"
