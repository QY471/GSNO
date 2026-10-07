#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing Cave/}"
: "${CHECKPOINT:?Set CHECKPOINT to a frozen model checkpoint}"
: "${BEST_EPOCH:?Set BEST_EPOCH from the training metadata}"
: "${BEST_PSNR:?Set BEST_PSNR from the training metadata}"
GPU="${GPU:-0}"
OUTPUT="${OUTPUT:-results/cave_cross_scale.json}"
SELECTION_SCALE="${SELECTION_SCALE:-4}"
read -r -a EVAL_SCALES <<< "${SCALES:-4 8 16 32}"

CUDA_VISIBLE_DEVICES="${GPU}" python tools/evaluate_dynamic_model_multiscale.py \
  --module model.gsno \
  --checkpoint "${CHECKPOINT}" \
  --dataset cave \
  --data-path "${DATA_ROOT}/Cave/Test" \
  --scales "${EVAL_SCALES[@]}" --dim 80 \
  --selection-scale "${SELECTION_SCALE}" \
  --selected-best-epoch "${BEST_EPOCH}" \
  --selected-best-psnr "${BEST_PSNR}" \
  --output "${OUTPUT}"
