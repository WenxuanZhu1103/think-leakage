#!/bin/bash
set -euo pipefail
. jobs/env.sh
model_vars ${MODEL:?8B, 4B or 1.7B}
DST=${OUT}_32k
mkdir -p $DST
python3 scripts/continue_gen_dir.py --base $BASE --src_dir $OUT/gen --dst_dir $DST/gen --adapters $V --max_total 32768
python3 scripts/llr_dump.py --base $BASE --gen_dir $DST/gen --template_kwarg --out $DST/llr_per_text.json
python3 scripts/think_vector.py --base $BASE --gen_dir $DST/gen --traj $V --template_kwarg --think_opener '<think>\n' \
  --max_resp_tok 32768 --bs 1 --n_boot 500 --out $DST/think_vector_fulllen.json
