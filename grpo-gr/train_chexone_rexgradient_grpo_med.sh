#!/usr/bin/env bash
set -euo pipefail

# Example usage:
# bash grpo-gr/train_chexone_rexgradient_grpo_med.sh /path/to/ReXGradient-160K /path/to/output_dir

DATA_ROOT="${1:-./data/ReXGradient-160K}"
RUN_OUTPUT_DIR="${2:-./outputs/chexone_rexgradient_grpo_med}"
PREP_DIR="${RUN_OUTPUT_DIR}/prepared_data"

mkdir -p "${PREP_DIR}"

python grpo-gr/prepare_rexgradient_160k_for_grpo.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${PREP_DIR}" \
  --metadata-dir "${DATA_ROOT}/metadata"

accelerate launch \
  --config_file ./accelerate_configs/single_gpu.yaml \
  grpo-gr/GRPO_GR_med.py \
  --dataset_name rr \
  --model_name_or_path StanfordAIMI/CheXOne \
  --setting rr_add_grounded_reasoning_med_think_rethink_matched_repet \
  --train_data_path "${PREP_DIR}/train.jsonl" \
  --train_image_folder_path "${DATA_ROOT}" \
  --eval_data_path "${PREP_DIR}/test.jsonl" \
  --eval_image_folder_path "${DATA_ROOT}" \
  --output_dir "${RUN_OUTPUT_DIR}/checkpoints" \
  --max_turns 1 \
  --max_completion_length 512 \
  --num_generations 4 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-6 \
  --eval_strategy steps \
  --eval_steps 200 \
  --save_steps 200 \
  --logging_steps 10 \
  --report_to wandb
