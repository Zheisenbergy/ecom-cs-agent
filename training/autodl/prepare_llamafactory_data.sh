#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$PROJECT_ROOT/.venv/bin/python"}

DATA_VARIANT=${DATA_VARIANT:-seed}

case "$DATA_VARIANT" in
  seed)
    TRAIN_TRACE_PATH=${TRAIN_TRACE_PATH:-"$PROJECT_ROOT/training/datasets/episode_traces.train.seed.generated.jsonl"}
    DEV_TRACE_PATH=${DEV_TRACE_PATH:-"$PROJECT_ROOT/training/datasets/episode_traces.dev.seed.generated.jsonl"}
    ROUTER_DATA_DIR=${ROUTER_DATA_DIR:-"$SCRIPT_DIR/lf_data/router"}
    ANSWER_DATA_DIR=${ANSWER_DATA_DIR:-"$SCRIPT_DIR/lf_data/answer"}
    ;;
  mixed)
    TRAIN_TRACE_PATH=${TRAIN_TRACE_PATH:-"$PROJECT_ROOT/training/datasets/episode_traces.train.mixed.generated.jsonl"}
    DEV_TRACE_PATH=${DEV_TRACE_PATH:-"$PROJECT_ROOT/training/datasets/episode_traces.dev.mixed.generated.jsonl"}
    ROUTER_DATA_DIR=${ROUTER_DATA_DIR:-"$SCRIPT_DIR/lf_data/router_mixed"}
    ANSWER_DATA_DIR=${ANSWER_DATA_DIR:-"$SCRIPT_DIR/lf_data/answer_mixed"}
    ;;
  *)
    echo "Unsupported DATA_VARIANT: $DATA_VARIANT" >&2
    echo "Expected one of: seed, mixed" >&2
    exit 1
    ;;
esac

mkdir -p "$ROUTER_DATA_DIR" "$ANSWER_DATA_DIR"

"$PYTHON_BIN" -m app.cli export-router-lf \
  --input "$TRAIN_TRACE_PATH" \
  --output "$ROUTER_DATA_DIR/router_sft.train.lf.json" \
  --dataset-name ecom_cs_router_sft_train \
  --dataset-info "$ROUTER_DATA_DIR/dataset_info.json"

"$PYTHON_BIN" -m app.cli export-router-lf \
  --input "$DEV_TRACE_PATH" \
  --output "$ROUTER_DATA_DIR/router_sft.dev.lf.json" \
  --dataset-name ecom_cs_router_sft_dev \
  --dataset-info "$ROUTER_DATA_DIR/dataset_info.json"

"$PYTHON_BIN" -m app.cli export-answer-lf \
  --input "$TRAIN_TRACE_PATH" \
  --output "$ANSWER_DATA_DIR/answer_sft.train.lf.json" \
  --dataset-name ecom_cs_answer_sft_train \
  --dataset-info "$ANSWER_DATA_DIR/dataset_info.json"

"$PYTHON_BIN" -m app.cli export-answer-lf \
  --input "$DEV_TRACE_PATH" \
  --output "$ANSWER_DATA_DIR/answer_sft.dev.lf.json" \
  --dataset-name ecom_cs_answer_sft_dev \
  --dataset-info "$ANSWER_DATA_DIR/dataset_info.json"

echo "Prepared LLaMA-Factory datasets:"
echo "  variant: $DATA_VARIANT"
echo "  router: $ROUTER_DATA_DIR"
echo "  answer: $ANSWER_DATA_DIR"
