#!/bin/bash
set -euo pipefail
. jobs/env.sh
model_vars 8B
SUB=analysis_out/think_leak/surgery/subspace.pt
PREV=analysis_out/think_leak/lesion
OUTD=analysis_out/think_leak/lesion32k
CONDS=${CONDS:-"rl_none:rl:none:8 rl_think8:rl:think:8 rl_pca8:rl:pca:8 rl_rand8:rl:rand:8 rl_think1:rl:think:1 base_none:base:none:8 base_think8:base:think:8 base_pca8:base:pca:8 base_rand8:base:rand:8"}
mkdir -p $PREV $OUTD
for c in $CONDS; do
  IFS=':' read -r tag target family k <<< "$c"
  [ -f $PREV/$tag.jsonl ] || python3 scripts/think_lesion.py --base $BASE --adapter $RL --subspace $SUB --target $target --family $family \
    --k $k --layers 1-35 --g 1.0 --out $PREV/$tag
  [ -f $OUTD/$tag.json ] || python3 scripts/think_lesion_continue.py --base $BASE --adapter $RL --subspace $SUB --target $target --family $family \
    --k $k --layers 1-35 --g 1.0 --prev $PREV/$tag.jsonl --max_total 32768 --bs 8 --seed 1 --no_compact --out $OUTD/$tag
done
python3 scripts/think_lesion_report.py --dir $OUTD
