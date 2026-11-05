#!/bin/bash
# e2000fs_calib8_qxgb: QXGB arm (arXiv:2608.15290) on the F1/F2/F3 quarters.
# Three lanes (one per period, blocks run sequentially inside a lane), then merge the block parts.
# Configuration is the compute-budgeted one recorded in the run notes: 5-level dense grid, 300 rounds, eta 0.1
# (capacity-matched to the frozen eta 0.03 x 900 recipe), Gaussian kernel, h = 0.5 in standardised units.
set -u
cd /home/t-lab01/sunrun/research/e2000fs_f123-20261031 || exit 1
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
export QXGB_LEVELS=0.05,0.25,0.5,0.75,0.95
export QXGB_ROUNDS=300
export QXGB_ETA=0.1
export QXGB_KERNEL=gaussian
export QXGB_BANDWIDTH=0.5
PY=/home/t-lab01/sunrun/.venv/bin/python
mkdir -p logs runs/qxgb_v1
for p in F1 F2 F3; do
  (
    for b in 0 1 2; do
      [ -f runs/qxgb_v1/${p}_part${b}.npz ] && continue
      $PY arms.py --arm qxgb --period "$p" --blocks "$b" --threads 2 --out runs/qxgb_v1 \
        > "logs/qxgb_${p}_b${b}.log" 2>&1
    done
  ) &
done
wait
$PY merge_parts.py runs/qxgb_v1
echo "QXGB-RUN-DONE rc=$?"
