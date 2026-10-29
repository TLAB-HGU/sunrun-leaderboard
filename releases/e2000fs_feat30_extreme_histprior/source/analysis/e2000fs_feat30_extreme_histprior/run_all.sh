#!/bin/bash
# Stages 2-3 of e2000fs_feat30_extreme_histprior (stage 1 = D1, already checkpointed).
set -u
cd /home/t-lab01/sunrun || exit 1
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1
PY=/home/t-lab01/sunrun/.venv/bin/python
RUN=full_20261028_v2
STUDY=analysis/e2000fs_feat30_extreme_histprior/run.py
exec >> analysis/e2000fs_feat30_extreme_histprior/runs/$RUN/console.log 2>&1
echo "=== $(date -u +%FT%TZ) stage 2: D2,D3,C baselines + weighted head ==="
$PY $STUDY --run-id $RUN --windows D2,D3,C --arms parent,reference,hist_weighted
echo "=== $(date -u +%FT%TZ) stage 3: C unweighted histogram ==="
$PY $STUDY --run-id $RUN --windows C --arms hist_unweighted
echo "=== $(date -u +%FT%TZ) finished ==="
