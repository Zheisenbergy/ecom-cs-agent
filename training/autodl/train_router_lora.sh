#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

LLAMAFACTORY_CLI=${LLAMAFACTORY_CLI:-llamafactory-cli}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-1.7B}
DATA_VARIANT=${DATA_VARIANT:-seed}

if [[ -z "${DATASET_DIR:-}" ]]; then
  if [[ "$DATA_VARIANT" == "mixed" ]]; then
    DATASET_DIR="$SCRIPT_DIR/lf_data/router_mixed"
  else
    DATASET_DIR="$SCRIPT_DIR/lf_data/router"
  fi
fi

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  if [[ "$DATA_VARIANT" == "mixed" ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/outputs/router-qwen3-1.7b-lora-mixed"
  else
    OUTPUT_DIR="$PROJECT_ROOT/outputs/router-qwen3-1.7b-lora"
  fi
fi

DATA_VARIANT="$DATA_VARIANT" ROUTER_DATA_DIR="$DATASET_DIR" "$SCRIPT_DIR/prepare_llamafactory_data.sh"

"$LLAMAFACTORY_CLI" train \
  --stage sft \
  --do_train true \
  --model_name_or_path "$BASE_MODEL" \
  --dataset_dir "$DATASET_DIR" \
  --dataset ecom_cs_router_sft_train \
  --eval_dataset ecom_cs_router_sft_dev \
  --template qwen \
  --finetuning_type lora \
  --lora_target all \
  --cutoff_len "${CUTOFF_LEN:-2048}" \
  --overwrite_cache true \
  --preprocessing_num_workers "${PREPROCESSING_NUM_WORKERS:-8}" \
  --output_dir "$OUTPUT_DIR" \
  --logging_steps "${LOGGING_STEPS:-10}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --plot_loss true \
  --overwrite_output_dir true \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LEARNING_RATE:-1e-4}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3.0}" \
  --lr_scheduler_type cosine \
  --warmup_ratio "${WARMUP_RATIO:-0.1}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS:-50}" \
  --bf16 true \
  --report_to none
