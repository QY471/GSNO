#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing Cave/}"
GPU="${GPU:-0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-Checkpoint_CAVE}"

CUDA_VISIBLE_DEVICES="${GPU}" python Train_Cave.py \
  --dataset cave \
  --data_path "${DATA_ROOT}/Cave/Train" \
  --test_data_path "${DATA_ROOT}/Cave/Test" \
  --model gsno \
  --sf 4 --dim 80 --ep_total "${EPOCHS:-1000}" --e_every 5 \
  --checkpoint_root "${CHECKPOINT_ROOT}"
