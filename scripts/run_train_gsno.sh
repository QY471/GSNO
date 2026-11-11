#!/usr/bin/env bash
# Train GSNO on one of the released CAVE or Harvard protocols.
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing Cave/ and Harvard/.}"

DATASET="${DATASET:-cave}"
SCALE="${SCALE:-4}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-1000}"

case "${DATASET}:${SCALE}" in
  cave:4)    TRAIN_ROOT="${DATA_ROOT}/Cave/Train";    TEST_ROOT="${DATA_ROOT}/Cave/Test";    DEFAULT_CHECKPOINT="Checkpoint_CAVE" ;;
  cave:8)    TRAIN_ROOT="${DATA_ROOT}/Cave/Train";    TEST_ROOT="${DATA_ROOT}/Cave/Test";    DEFAULT_CHECKPOINT="Checkpoint_CAVE_x8" ;;
  harvard:4) TRAIN_ROOT="${DATA_ROOT}/Harvard/Train"; TEST_ROOT="${DATA_ROOT}/Harvard/Test"; DEFAULT_CHECKPOINT="Checkpoint_Harvard" ;;
  harvard:8) TRAIN_ROOT="${DATA_ROOT}/Harvard/Train"; TEST_ROOT="${DATA_ROOT}/Harvard/Test"; DEFAULT_CHECKPOINT="Checkpoint_Harvard_x8" ;;
  *)
    echo "DATASET must be cave or harvard, and SCALE must be 4 or 8." >&2
    exit 2
    ;;
esac

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${DEFAULT_CHECKPOINT}}"

CUDA_VISIBLE_DEVICES="${GPU}" python train_gsno.py \
  --dataset "${DATASET}" \
  --data_path "${TRAIN_ROOT}" \
  --test_data_path "${TEST_ROOT}" \
  --model gsno \
  --sf "${SCALE}" \
  --dim 80 \
  --ep_total "${EPOCHS}" \
  --e_every 5 \
  --checkpoint_root "${CHECKPOINT_ROOT}"
