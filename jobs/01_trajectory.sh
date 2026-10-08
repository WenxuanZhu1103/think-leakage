#!/bin/bash
set -euo pipefail
. jobs/env.sh
model_vars ${MODEL:?8B, 4B or 1.7B}
mkdir -p $OUT
python3 scripts/think_leak_gen.py --base $BASE --traj $V --template_kwarg --n 2 --max_tokens 16384 --think_max_tokens 4096 --out_dir $OUT/gen
python3 scripts/think_vector.py --base $BASE --gen_dir $OUT/gen --traj $V --template_kwarg --think_opener '<think>\n' --out $OUT/think_vector.json
