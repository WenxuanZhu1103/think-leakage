#!/bin/bash
set -euo pipefail
. jobs/env.sh
model_vars ${MODEL:?8B, 4B or 1.7B}
CKPT_ROOT=${CKPT_ROOT:-checkpoints/thinking_leakage/$EXP}
mkdir -p $AD
for s in $STEPS; do
  pt=$CKPT_ROOT/global_step_$s/actor/model_world_size_1_rank_0.pt
  [ -f $AD/$PFX$s/adapter_model.safetensors ] && continue
  CUDA_VISIBLE_DEVICES="" python3 train/extract_lora_adapter.py --pt $pt --base $BASE --out $AD/$PFX$s
done
