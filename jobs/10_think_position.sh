#!/bin/bash
set -euo pipefail
. jobs/env.sh
O=analysis_out/think_leak/position
mkdir -p $O
for MODEL in 8B 4B 1.7B; do
  model_vars $MODEL
  [ -f $O/$MODEL.json ] || python3 scripts/think_position.py --base $BASE --adapter $RL --gen_dir $OUT/gen --out $O/$MODEL.json
done
