"""Official-window forecasts for the E0141 recipe (D71: chronological OOF stack of e0003_line + d33_hsplit + d45_lvlmean).

Same code path as experiments/d71_stack/runner.py (E0141, 3h train grid). The global convex weights are the fixed pre-rule product
recorded by E0141 (inverse dev-period OOF MSE, clipped [0.05,0.90], renormalised); selection/audit/official statistics never
enter them. One 7-fit set on targets < 2026-05-29, predicting every official origin x 72h. Writes $EXPOS_OUT/predictions_official.parquet.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/expos_runners"))
sys.path.insert(0, str(LB / "experiments/d71_stack"))
import runner as R  # noqa: E402  (E0141 code, unchanged)
import official_members as OM  # noqa: E402

OUT = Path(os.environ.get("EXPOS_OUT", "."))
# E0141 selection.json weights (dev-period OOF only): dev_mses 7888.60 / 7705.80 / 7712.87
WEIGHTS = {"e0003_line": 0.3282447610385238, "d33_hsplit": 0.33603162776340306, "d45_lvlmean": 0.33572361119807315}


def blend(mem):
    return R.STK.stack_predictions(mem, WEIGHTS)


def main():
    params, df, cols, meta, block = OM.prepare(R)
    params = R.STK.fit_params()
    models = OM.fit_set(R, df, cols, params)
    te, _ = R.add_strips(R.base_grid(OM.ORIGINS), meta, block)
    pred = blend(OM.members(R, models, te, cols))
    if not np.isfinite(pred).all() or len(pred) != len(OM.ORIGINS) * 72:
        raise SystemExit("non-finite or incomplete official predictions; refusing to write")
    R.write(te, pred, "official")
    sample = te[te["origin"].isin(OM.ORIGINS[::14][:200])]
    t0 = time.perf_counter()
    for _, g in sample.groupby("origin"):
        blend(OM.members(R, models, g, cols))
    per_fold = (time.perf_counter() - t0) / sample["origin"].nunique()
    (OUT / "timing.json").write_text(json.dumps({"inference_seconds_per_fold": per_fold, "train_rows": models[2],
                                                 "origins": len(OM.ORIGINS), "img_stale_frac": float(te["img_stale"].mean())}))
    print(f"train={models[2]} origins={len(OM.ORIGINS)} rows={len(te)} per_fold={per_fold:.6f}s", flush=True)


if __name__ == "__main__":
    main()
