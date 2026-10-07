#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing Harvard/}"
GPU="${GPU:-0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-Checkpoint_Harvard_x8}"

CUDA_VISIBLE_DEVICES="${GPU}" python Train_Harvard.py \
  --data_path "${DATA_ROOT}/Harvard/Train" \
  --test_data_path "${DATA_ROOT}/Harvard/Test" \
  --model gsno \
  --sf 8 --dim 80 --ep_total "${EPOCHS:-1000}" --e_every 5 \
  --checkpoint_root "${CHECKPOINT_ROOT}"
