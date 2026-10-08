#!/bin/bash
set -euo pipefail
. jobs/env.sh
model_vars 8B
ARM=q8s2
R=analysis_out/think_dyn/$ARM/rollouts
S=analysis_out/think_dyn/$ARM/scores
T="base=none $V"
python3 scripts/think_dynamics_sample.py --base $BASE --traj $T --n 10 --n_base 20 --n_prompts 256 --max_tokens 8192 --out_dir $R
python3 scripts/think_dynamics_score.py --base $BASE --rollouts_dir $R --vectors $OUT/think_vector_vectors.pt --traj $T --out_dir $S
python3 scripts/think_dynamics_report.py --dir $S > analysis_out/think_dyn/$ARM/report.txt
python3 scripts/think_proj_full.py --base $BASE --rollouts_dir $R --vectors ${OUT}_32k/think_vector_fulllen_vectors.pt --out analysis_out/think_dyn/$ARM/proj_full_base.json
python3 scripts/grad_probe.py --base $BASE --rollouts_dir $R --scores $S/base.jsonl --aime_gen $OUT/gen \
  --vectors ${OUT}_32k/think_vector_fulllen_vectors.pt --grad_dir checkpoints/walk_variants/grad_probe --out analysis_out/think_leak/grad_probe.json
