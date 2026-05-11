#!/bin/bash
set -e

# Example:
#   modal=WL bash train.sh
#   modal=NBI epoch_time=150 batch_size=8 learning_rate=1e-4 bash train.sh

MODAL=${modal:-WL}
EPOCH_TIME=${epoch_time:-150}
BATCH_SIZE=${batch_size:-8}
LEARNING_RATE=${learning_rate:-1e-4}
IMG_SIZE=${img_size:-224}
DATA_ROOT=${data_root:-../../dataset}
RUN_NAME=${run_name:-unetpp_${MODAL}}
OUT_DIR=${out_dir:-../outputs/runs/${RUN_NAME}}

python train.py \
  --data_root "$DATA_ROOT" \
  --modal "$MODAL" \
  --run_name "$RUN_NAME" \
  --output_dir "$OUT_DIR" \
  --max_epochs "$EPOCH_TIME" \
  --img_size "$IMG_SIZE" \
  --base_lr "$LEARNING_RATE" \
  --batch_size "$BATCH_SIZE"
