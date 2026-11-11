#!/usr/bin/env bash
# Evaluate one frozen GSNO checkpoint across the requested fusion ratios.
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing Cave/ and Harvard/.}"
: "${CHECKPOINT:?Set CHECKPOINT to a frozen GSNO checkpoint.}"
: "${BEST_EPOCH:?Set BEST_EPOCH from the training metadata.}"
: "${BEST_PSNR:?Set BEST_PSNR from the training metadata.}"

DATASET="${DATASET:-cave}"
GPU="${GPU:-0}"
OUTPUT="${OUTPUT:-results/${DATASET}_cross_scale.json}"
SELECTION_SCALE="${SELECTION_SCALE:-4}"
read -r -a EVAL_SCALES <<< "${SCALES:-4 8 16 32}"

case "${DATASET}" in
  cave)    DATA_PATH="${DATA_ROOT}/Cave/Test" ;;
  harvard) DATA_PATH="${DATA_ROOT}/Harvard/Test" ;;
  *)
    echo "DATASET must be cave or harvard." >&2
    exit 2
    ;;
esac

CUDA_VISIBLE_DEVICES="${GPU}" python tools/evaluate_gsno_multiscale.py \
  --module model.gsno \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET}" \
  --data-path "${DATA_PATH}" \
  --scales "${EVAL_SCALES[@]}" \
  --dim 80 \
  --selection-scale "${SELECTION_SCALE}" \
  --selected-best-epoch "${BEST_EPOCH}" \
  --selected-best-psnr "${BEST_PSNR}" \
  --output "${OUTPUT}"
