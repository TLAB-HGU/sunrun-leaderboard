#!/usr/bin/env bash
# Fetch inputs, rebuild and verify both experiments. Run from the repository root.
#   bash reproduce/run_all.sh            # full run (pointweight4 trains 216 LightGBM models, ~30-40 min on 2 threads)
#   bash reproduce/run_all.sh --smoke    # 3-tree pointweight4 + full e1604 (about a minute)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
DATA="${SUNRUN_REPRO_DATA:-reproduce/reproduce_data}"
OUT="${SUNRUN_REPRO_OUT:-reproduce/outputs}"
export SUNRUN_REPRO_DATA="$DATA" SUNRUN_REPRO_OUT="$OUT"
SMOKE=""; [ "${1:-}" = "--smoke" ] && SMOKE="--smoke"

"$PY" reproduce/fetch_data.py --reference

# e1604_hsspeak25: raw ACE -> 29 features (checked against the frozen matrices) -> heads -> bump
"$PY" reproduce/e1604_hsspeak25/run.py --reference "$DATA/reference/e1604_predictions.parquet"

# e2000fs_feat30_pointweight4: raw ACE + frozen grids -> lag-0 cache -> 72 horizon models x 3 stages -> score
"$PY" reproduce/feat30_pointweight4/build_features.py --out "$OUT/pointweight4_cache"
"$PY" reproduce/feat30_pointweight4/compare_cache.py "$OUT/pointweight4_cache" "$DATA/reference/pointweight4_cache"
"$PY" reproduce/feat30_pointweight4/train_predict.py --cache "$OUT/pointweight4_cache" $SMOKE \
    --reference "$DATA/reference/pointweight4_forecast.npz"
