#!/bin/bash
set -euo pipefail
. jobs/env.sh
model_vars 8B
W=checkpoints/walk_variants/adapters_s2amp
OUT=analysis_out/think_leak/qwen_amp_s2
python3 train/scale_adapter.py --src $AD/${PFX}15 --out_root $W --prefix s2_15 --scales 1.0 1.77 2.5 3.7 5.0 7.0 9.0 12.0
python3 train/scale_adapter.py --src $AD/${PFX}67 --out_root $W --prefix s2_67 --scales 1.0 1.5 2.0 3.0 4.0 5.0
V=""
for c in 1.0 1.77 2.5 3.7 5.0 7.0 9.0 12.0; do V="$V s15c$c=$W/s2_15_c$c"; done
for c in 1.0 1.5 2.0 3.0 4.0 5.0; do V="$V s67c$c=$W/s2_67_c$c"; done
mkdir -p $OUT
python3 scripts/think_leak_gen.py --base $BASE --traj $V --template_kwarg --n 2 --max_tokens 32768 --think_max_tokens 4096 --out_dir $OUT/gen
python3 scripts/llr_dump.py --base $BASE --gen_dir $OUT/gen --template_kwarg --out $OUT/llr_per_text.json
python3 scripts/think_vector.py --base $BASE --gen_dir $OUT/gen --traj $V --template_kwarg --think_opener '<think>\n' \
  --max_resp_tok 32768 --bs 1 --n_boot 500 --out $OUT/think_vector_fulllen.json
