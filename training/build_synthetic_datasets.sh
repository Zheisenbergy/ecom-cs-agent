#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
CLI_BIN=${CLI_BIN:-"$PROJECT_ROOT/.venv/bin/ecom-cs-agent"}

CONFIG_PATH=${CONFIG_PATH:-"$PROJECT_ROOT/training/datasets/synthesis_templates.default.json"}
TRAIN_SYN_PATH=${TRAIN_SYN_PATH:-"$PROJECT_ROOT/training/datasets/episode_cases.synthetic.train.jsonl"}
DEV_SYN_PATH=${DEV_SYN_PATH:-"$PROJECT_ROOT/training/datasets/episode_cases.synthetic.dev.jsonl"}
TRAIN_MIXED_PATH=${TRAIN_MIXED_PATH:-"$PROJECT_ROOT/training/datasets/episode_cases.train.mixed.jsonl"}
DEV_MIXED_PATH=${DEV_MIXED_PATH:-"$PROJECT_ROOT/training/datasets/episode_cases.dev.mixed.jsonl"}

TRAIN_TRACE_PATH=${TRAIN_TRACE_PATH:-"$PROJECT_ROOT/training/datasets/episode_traces.train.mixed.generated.jsonl"}
DEV_TRACE_PATH=${DEV_TRACE_PATH:-"$PROJECT_ROOT/training/datasets/episode_traces.dev.mixed.generated.jsonl"}

ROUTER_TRAIN_JSONL=${ROUTER_TRAIN_JSONL:-"$PROJECT_ROOT/training/datasets/router_sft.train.mixed.generated.jsonl"}
ROUTER_DEV_JSONL=${ROUTER_DEV_JSONL:-"$PROJECT_ROOT/training/datasets/router_sft.dev.mixed.generated.jsonl"}
ANSWER_TRAIN_JSONL=${ANSWER_TRAIN_JSONL:-"$PROJECT_ROOT/training/datasets/answer_sft.train.mixed.generated.jsonl"}
ANSWER_DEV_JSONL=${ANSWER_DEV_JSONL:-"$PROJECT_ROOT/training/datasets/answer_sft.dev.mixed.generated.jsonl"}

ROUTER_TRAIN_LF=${ROUTER_TRAIN_LF:-"$PROJECT_ROOT/training/datasets/router_sft.train.mixed.lf.json"}
ROUTER_DEV_LF=${ROUTER_DEV_LF:-"$PROJECT_ROOT/training/datasets/router_sft.dev.mixed.lf.json"}
ANSWER_TRAIN_LF=${ANSWER_TRAIN_LF:-"$PROJECT_ROOT/training/datasets/answer_sft.train.mixed.lf.json"}
ANSWER_DEV_LF=${ANSWER_DEV_LF:-"$PROJECT_ROOT/training/datasets/answer_sft.dev.mixed.lf.json"}

ROUTER_DATASET_INFO=${ROUTER_DATASET_INFO:-"$PROJECT_ROOT/training/datasets/dataset_info.ecom_cs_router_sft.mixed.json"}
ANSWER_DATASET_INFO=${ANSWER_DATASET_INFO:-"$PROJECT_ROOT/training/datasets/dataset_info.ecom_cs_answer_sft.mixed.json"}

"$CLI_BIN" synthesize-episodes \
  --config "$CONFIG_PATH" \
  --output-train "$TRAIN_SYN_PATH" \
  --output-dev "$DEV_SYN_PATH"

cat "$PROJECT_ROOT/training/datasets/episode_cases.train.seed.jsonl" "$TRAIN_SYN_PATH" > "$TRAIN_MIXED_PATH"
cat "$PROJECT_ROOT/training/datasets/episode_cases.dev.seed.jsonl" "$DEV_SYN_PATH" > "$DEV_MIXED_PATH"

"$CLI_BIN" run --input "$TRAIN_MIXED_PATH" --output "$TRAIN_TRACE_PATH"
"$CLI_BIN" run --input "$DEV_MIXED_PATH" --output "$DEV_TRACE_PATH"

"$CLI_BIN" export-router-sft --input "$TRAIN_TRACE_PATH" --output "$ROUTER_TRAIN_JSONL"
"$CLI_BIN" export-router-sft --input "$DEV_TRACE_PATH" --output "$ROUTER_DEV_JSONL"
"$CLI_BIN" export-answer-sft --input "$TRAIN_TRACE_PATH" --output "$ANSWER_TRAIN_JSONL"
"$CLI_BIN" export-answer-sft --input "$DEV_TRACE_PATH" --output "$ANSWER_DEV_JSONL"

"$CLI_BIN" export-router-lf \
  --input "$TRAIN_TRACE_PATH" \
  --output "$ROUTER_TRAIN_LF" \
  --dataset-name ecom_cs_router_sft_train_mixed \
  --dataset-info "$ROUTER_DATASET_INFO"

"$CLI_BIN" export-router-lf \
  --input "$DEV_TRACE_PATH" \
  --output "$ROUTER_DEV_LF" \
  --dataset-name ecom_cs_router_sft_dev_mixed \
  --dataset-info "$ROUTER_DATASET_INFO"

"$CLI_BIN" export-answer-lf \
  --input "$TRAIN_TRACE_PATH" \
  --output "$ANSWER_TRAIN_LF" \
  --dataset-name ecom_cs_answer_sft_train_mixed \
  --dataset-info "$ANSWER_DATASET_INFO"

"$CLI_BIN" export-answer-lf \
  --input "$DEV_TRACE_PATH" \
  --output "$ANSWER_DEV_LF" \
  --dataset-name ecom_cs_answer_sft_dev_mixed \
  --dataset-info "$ANSWER_DATASET_INFO"

echo "Synthetic seed pipeline completed."
echo "  train synthetic seed: $TRAIN_SYN_PATH"
echo "  dev synthetic seed:   $DEV_SYN_PATH"
echo "  train mixed seed:     $TRAIN_MIXED_PATH"
echo "  dev mixed seed:       $DEV_MIXED_PATH"
echo "  router train lf:      $ROUTER_TRAIN_LF"
echo "  answer train lf:      $ANSWER_TRAIN_LF"
