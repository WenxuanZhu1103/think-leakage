#!/bin/bash
set -euo pipefail
. jobs/env.sh
model_vars ${MODEL:-8B}
SUBDIR=analysis_out/think_leak/surgery
[ "$MODEL" != 8B ] && SUBDIR=$OUT
mkdir -p $SUBDIR
[ -f $SUBDIR/subspace.pt ] || python3 scripts/think_subspace.py --base $BASE --template_kwarg --think_opener '<think>\n' --out $SUBDIR/subspace.pt
