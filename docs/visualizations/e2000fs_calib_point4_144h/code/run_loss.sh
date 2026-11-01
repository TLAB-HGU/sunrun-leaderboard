#!/bin/bash
# Nine single-thread refits (3 periods x 3 blocks) in parallel, then the aggregate figure and receipt.
# Each refit must reproduce the archived point4_144h booster exactly; the script exits non-zero otherwise.
set -u
cd /home/t-lab01/sunrun
PY=/home/t-lab01/sunrun/.venv/bin/python
SCRIPT=docs/visualizations/e2000fs_calib_point4_144h/code/loss_history.py
mkdir -p docs/visualizations/e2000fs_calib_point4_144h/loss_history
status=0
for period in F1 F2 F3; do
  for block in 0 1 2; do
    $PY "$SCRIPT" --period "$period" --block "$block" &
  done
done
for job in $(jobs -p); do wait "$job" || status=1; done
if [ "$status" -ne 0 ]; then echo "at least one refit failed"; exit "$status"; fi
$PY "$SCRIPT" --aggregate
