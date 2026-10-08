#!/bin/bash
set -euo pipefail
. jobs/env.sh
model_vars ${MODEL:-8B}
O=analysis_out/think_leak/decomp
mkdir -p $O
python3 scripts/think_write_decomp.py --base $BASE --adapter $RL --gen_dir $OUT/gen --vectors $OUT/think_vector_vectors.pt --out $O/${M}_s67.json
