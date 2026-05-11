#!/bin/bash
set -e

# Example:
#   modal=WL bash test.sh
#   modal=NBI bash test.sh

MODAL=${modal:-WL}
IMG_SIZE=${img_size:-224}
DATA_ROOT=${data_root:-../../dataset}
RUN_NAME=${run_name:-attention_unet_${MODAL}}
OUT_DIR=${out_dir:-../outputs/runs/${RUN_NAME}}
CHECKPOINT=${checkpoint:-${OUT_DIR}/best_model.pth}

python test.py \
  --data_root "$DATA_ROOT" \
  --modal "$MODAL" \
  --run_name "$RUN_NAME" \
  --output_dir "$OUT_DIR" \
  --checkpoint "$CHECKPOINT" \
  --img_size "$IMG_SIZE" \
  --batch_size 1
