#!/bin/bash
set -euo pipefail
. jobs/env.sh
model_vars ${MODEL:?8B, 4B or 1.7B}
C=analysis_out/think_circuit
mkdir -p $C
[ -f $C/gate_$M.json ] || python3 scripts/circuit_gate.py --base $BASE --gen_dir $OUT/gen --traj $V --out $C/gate_$M.json
[ -f $C/gate_attn_$M.json ] || python3 scripts/circuit_gate_attn.py --base $BASE --gen_dir $OUT/gen --gate_json $C/gate_$M.json --pcts 1,5 --n_rand 5 --out $C/gate_attn_$M.json
[ -f $C/route_${M}_refs.json ] || python3 scripts/circuit_route.py --base $BASE --adapter $RL --gen_dir $OUT/gen --refs_only --out $C/route_${M}_refs.json
NL=$(python3 -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('$BASE').num_hidden_layers)")
LATE=$((NL * 3 / 4))
CH=${CHUNK:-4}
for ((a = 0; a < LATE; a += CH)); do
  b=$((a + CH - 1)); [ $b -ge $LATE ] && b=$((LATE - 1))
  L=$(seq -s, $a $b)
  [ -f $C/route_${M}_scan_L$a-$b.json ] || python3 scripts/circuit_route.py --base $BASE --adapter $RL --gen_dir $OUT/gen \
    --scan_layers $L --head_texts 30 --out $C/route_${M}_scan_L$a-$b.json
done
python3 scripts/circuit_route_summary.py $M
