#!/bin/bash
set -euo pipefail
. jobs/env.sh
model_vars 8B
O=analysis_out/think_leak/steer32k
mkdir -p $O
python3 scripts/think_vector.py --base $BASE --gen_dir $OUT/gen --traj $V --template_kwarg --think_opener '<think>\n' \
  --steer single:18 --alphas 0 0.25 0.5 0.75 1.0 --out $OUT/think_vector.json
for A in 0 0.25 0.5 0.75 1.0; do
  [ -f $O/L18_a$A.json ] || python3 scripts/steer_continue.py --base $BASE --inp $OUT/think_vector_steer_L18_a$A.jsonl \
    --vectors $OUT/think_vector_vectors.pt --layer 18 --alpha $A --max_total 32768 --bs 8 --out $O/L18_a$A
done
