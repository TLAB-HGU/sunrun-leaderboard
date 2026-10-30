"""D148 step 2: the E0466 recipe (D147 lean arm s3_std) plus the five DONKI WSA-Enlil features and enlil_dt_target.

Only change vs E0466: the base XGBoost models also see the existing+enlil ENLIL columns (info_ablation.enlil_features,
runs with modelCompletionTime <= origin), read from the local DONKI copy up to 2026-05 plus store/donki-live
(2026-06..2026-10, live API; June identical to the local copy). Every grid built by run_official_a2b.base_grid_for
(OOF paths, candidate pool, official grid) gets the same ENLIL join, so OOF windows, residual pool and official predictions
stay consistent. lean.py and the a2b helpers are imported read-only and patched in-process.
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/d147_e0421_lean"))
import lean as L  # noqa: E402 (read-only)

ia, A = L.ia, L.A
LIVE = LB / "store/donki-live"
ENLIL = ["enlil_next_arr_h", "enlil_since_arr_h", "enlil_n_pending", "enlil_kp", "enlil_glancing"]


def enlil_runs():
    files = [f for f in sorted(glob.glob(str(ia.OUT / "donki/WSAEnlilSimulations_*.json"))) if f[-12:-5] < "2026-06"]
    files += sorted(glob.glob(str(LIVE / "WSAEnlilSimulations_*.json")))
    rows = []
    for f in files:
        for s in json.load(open(f)):
            if s.get("estimatedShockArrivalTime"):
                rows.append((pd.Timestamp(s["modelCompletionTime"]), pd.Timestamp(s["estimatedShockArrivalTime"]),
                             max(s.get("kp_90") or 0, s.get("kp_135") or 0, s.get("kp_180") or 0),
                             bool(s.get("isEarthGB"))))
    return pd.DataFrame(rows, columns=["done", "arrival", "kp", "glancing"]).drop_duplicates().sort_values("done")


RUNS = enlil_runs()
_TABLE = {}


def enlil_table(origins):
    o = pd.DatetimeIndex(origins, tz="UTC") if pd.DatetimeIndex(origins).tz is None else pd.DatetimeIndex(origins)
    miss = o[~o.isin(pd.DatetimeIndex(list(_TABLE)) if _TABLE else pd.DatetimeIndex([], tz="UTC"))].unique()
    if len(miss):
        f = ia.enlil_features(miss, RUNS)
        _TABLE.update({t: r for t, r in zip(f.index, f[ENLIL].to_numpy())})
    return pd.DataFrame([_TABLE[t] for t in o], index=o, columns=ENLIL)


_base_grid_for = A.base_grid_for


def base_grid_for(origins, ace, HR, meta, block, cols):
    g = _base_grid_for(origins, ace, HR, meta, block, [c for c in cols if not c.startswith("enlil_")])
    e = enlil_table(pd.to_datetime(g["origin"], utc=True))
    for c in ENLIL:
        g[c] = e[c].to_numpy()
    g["enlil_dt_target"] = g["h"] - g["enlil_next_arr_h"]
    return g


_feature_columns = L.feature_columns


def feature_columns(df, cfg):
    cols = _feature_columns(df, cfg)
    add = ENLIL + ["enlil_dt_target"]
    missing = [c for c in add if c not in df.columns]
    if missing:
        raise SystemExit(f"train frame lacks ENLIL columns {missing}")
    return cols + add


if __name__ == "__main__":
    ia.enlil_runs = enlil_runs
    A.base_grid_for = base_grid_for
    L.feature_columns = feature_columns
    print(f"d148 lean+enlil runs={len(RUNS)} last_done={RUNS['done'].max()}", flush=True)
    sys.argv = [sys.argv[0], "--arm", "s3_std"]
    L.main()
