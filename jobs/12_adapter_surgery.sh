#!/bin/bash
set -euo pipefail
. jobs/env.sh
model_vars 8B
SUB=analysis_out/think_leak/surgery/subspace.pt
A=checkpoints/walk_variants/surgery_adapters
O=analysis_out/think_leak/surgery
O32=analysis_out/think_leak/surgery_32k
CUDA_VISIBLE_DEVICES="" python3 scripts/adapter_surgery.py --adapter $RL --subspace $SUB \
  --specs rm:think:k8 rm:rand:think:k8 rm:topsv:k8 only:writers only:nonwriters --out_root $A --prefix s2x
T="s2intact=$RL"
for t in s2xrm_think_k8 s2xrm_rand_think_k8 s2xrm_topsv_k8 s2xonly_writers s2xonly_nonwriters; do T="$T $t=$A/$t"; done
python3 scripts/think_leak_gen.py --base $BASE --template_kwarg --n 8 --max_tokens 16384 --think_max_tokens 4096 --traj $T --out_dir $O/gen
python3 scripts/think_vector.py --base $BASE --gen_dir $O/gen --traj $T --template_kwarg --think_opener '<think>\n' --n_boot 100 --out $O/think_vector.json
mkdir -p $O32
python3 scripts/continue_gen_dir.py --base $BASE --src_dir $O/gen --dst_dir $O32/gen --adapters $T \
  --tags base_nothink base_think $(for x in $T; do echo -n "${x%%=*} "; done) --max_total 32768
