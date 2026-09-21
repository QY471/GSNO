#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing Harvard/}"
GPU="${GPU:-0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-Checkpoint_Harvard}"

CUDA_VISIBLE_DEVICES="${GPU}" python Train_Cave.py \
  --dataset harvard \
  --data_path "${DATA_ROOT}/Harvard/Train" \
  --test_data_path "${DATA_ROOT}/Harvard/Test" \
  --model e3_constrained_elliptical_gaussian \
  --sf 4 --dim 80 --ep_total "${EPOCHS:-1000}" --e_every 5 \
  --checkpoint_root "${CHECKPOINT_ROOT}"
