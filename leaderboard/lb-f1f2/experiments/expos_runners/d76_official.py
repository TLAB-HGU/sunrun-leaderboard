"""Official-window forecasts for the E0145 recipe (D76: per-lead mixture of e0003_line + d33_hsplit + d45_lvlmean).

Same code path as experiments/d76_moe/runner.py (E0145, 3h train grid). The per-lead convex weights (leads 1-6, 7-24, 25-48, 49-72) are the fixed pre-rule product
recorded by E0145 (inverse dev-period OOF MSE, clipped [0.05,0.90], renormalised); selection/audit/official statistics never
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
sys.path.insert(0, str(LB / "experiments/d76_moe"))
import runner as R  # noqa: E402  (E0145 code, unchanged)
import official_members as OM  # noqa: E402

OUT = Path(os.environ.get("EXPOS_OUT", "."))
# E0145 selection.json per-lead weights (dev-period OOF only)
WEIGHTS = {"1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272, "d45_lvlmean": 0.3946530834632206}, "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209, "d45_lvlmean": 0.3411415011482895}, "25-48": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333, "d45_lvlmean": 0.3333333333333333}, "49-72": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333, "d45_lvlmean": 0.3333333333333333}}


def blend(mem, te_h):
    return R.MOE.blend_per_lead(mem, WEIGHTS, te_h)


def main():
    params, df, cols, meta, block = OM.prepare(R)
    params = R.MOE.fit_params()
    models = OM.fit_set(R, df, cols, params)
    te, _ = R.add_strips(R.base_grid(OM.ORIGINS), meta, block)
    pred = blend(OM.members(R, models, te, cols), te["h"].to_numpy())
    if not np.isfinite(pred).all() or len(pred) != len(OM.ORIGINS) * 72:
        raise SystemExit("non-finite or incomplete official predictions; refusing to write")
    R.write(te, pred, "official")
    sample = te[te["origin"].isin(OM.ORIGINS[::14][:200])]
    t0 = time.perf_counter()
    for _, g in sample.groupby("origin"):
        blend(OM.members(R, models, g, cols), g["h"].to_numpy())
    per_fold = (time.perf_counter() - t0) / sample["origin"].nunique()
    (OUT / "timing.json").write_text(json.dumps({"inference_seconds_per_fold": per_fold, "train_rows": models[2],
                                                 "origins": len(OM.ORIGINS), "img_stale_frac": float(te["img_stale"].mean())}))
    print(f"train={models[2]} origins={len(OM.ORIGINS)} rows={len(te)} per_fold={per_fold:.6f}s", flush=True)


if __name__ == "__main__":
    main()
