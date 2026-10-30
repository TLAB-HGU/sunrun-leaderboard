#!/bin/sh
# Rebuild the E0472-family inputs up to now into store/v2 (never touches the pinned v1 files).
#   1. ace.parquet           = run.load_ace(sun-db ACE stage)                 (same builder as store/ch-v1/ace.parquet)
#   2. suvi-hourly-v2 + origin-features-v2 = xgboost_suvi_fusion/run.py --mode features
#   3. ch-hourly + all-features + masks   = ch_experiment.py --phase features  (no truth/folds are read in this phase)
set -eu
LB=/home/t-lab01/sunrun/leaderboard
PY=$LB/.venv-fusion/bin/python
STAGE=/home/t-lab01/.local/state/sundb/stage
V2=$LB/store/v2
WORKERS=${WORKERS:-8}
cd "$LB"
mkdir -p "$V2/ch" "$V2/suvi-fusion"
echo "[$(date -u +%FT%TZ)] ace.parquet"
$PY - <<PY
import sys; sys.path.insert(0, "experiments")
from pathlib import Path
from xgboost_suvi_fusion.run import load_ace
from xgboost_suvi_fusion.ch_experiment import write_atomic_parquet
ace = load_ace(["$STAGE/ace_solar"])
write_atomic_parquet(ace, Path("$V2/ch/ace.parquet"))
print("ace rows", len(ace), ace.timestamp_utc.min(), ace.timestamp_utc.max())
PY
echo "[$(date -u +%FT%TZ)] SUVI strips"
$PY experiments/xgboost_suvi_fusion/run.py --mode features \
  --ace "$STAGE/ace_solar" --suvi-meta "$STAGE/suvi_hourly" --images-root /home/t-lab01/sunrun/sun-img/data \
  --feature-cache "$V2/suvi-fusion/suvi-hourly-v2.parquet" --output "$V2/suvi-fusion/origin-features-v2.parquet" \
  --image-workers "$WORKERS"
echo "[$(date -u +%FT%TZ)] CH inputs"
$PY - <<PY
import sys; sys.path.insert(0, "experiments")
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq
from xgboost_suvi_fusion.ch_experiment import write_atomic_parquet
src = "$V2/suvi-fusion/origin-features-v2.parquet"
cols = [c for c in pq.read_schema(src).names if c.startswith("ace_")]
write_atomic_parquet(pd.read_parquet(src, columns=cols), Path("$V2/ch/ace-features.parquet"))
PY
echo "[$(date -u +%FT%TZ)] CH extraction"
$PY experiments/xgboost_suvi_fusion/ch_experiment.py --phase features \
  --ace "$STAGE/ace_solar" --suvi-meta "$STAGE/suvi_hourly" --images-root /home/t-lab01/sunrun/sun-img/data \
  --ace-origin-features "$V2/suvi-fusion/origin-features-v2.parquet" \
  --folds /dev/null --leader-predictions /dev/null --output "$V2/ch" --image-workers "$WORKERS"
echo "[$(date -u +%FT%TZ)] done"
