#!/bin/bash
set -euo pipefail
. jobs/env.sh
model_vars ${MODEL:?8B, 4B or 1.7B}
D=analysis_out/think_leak/ceiling/gen
[ "$MODEL" != 8B ] && D=$OUT/ceiling/gen
CT=""; CTT=""
for c in 15 40 67; do CT="$CT s$c=$AD/$PFX$c"; CTT="$CTT s${c}_think=$AD/$PFX$c"; done
python3 scripts/think_leak_gen.py --base $BASE --template_kwarg --n 8 --max_tokens 16384 --think_max_tokens 32768 \
  --traj $CT --traj_think $CTT --out_dir $D
python3 scripts/think_continue_long.py --base $BASE --inp $D/base_think.jsonl --out $D/base_think_40k.jsonl --max_total 40960
