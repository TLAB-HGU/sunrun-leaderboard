"""Protected scorer for HSSPEAK72: pooled rel, 7-day block CI, per-fold, event F1.

Reads the candidate and the two control experiments from the expos store. Never writes
to the store and never re-fits anything.

Usage:
    python -m experiments.solar_hsspeak72_20261019.score <EXPOS_OUT dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB))
from expos import core, evaluate, roll3  # noqa: E402

CONTROL = {1: ("E1508", "roll3"), 2: ("E1508", "roll3"), 3: ("E1508", "roll3"),
           4: ("E1529", "roll3x"), 5: ("E1529", "roll3x"), 6: ("E1529", "roll3x")}
SEEDS = (0, 1, 2)


def _frame(x: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"origin": pd.to_datetime(x.origin_last_input_utc, utc=True),
                         "h": x.horizon_hours.astype(int),
                         "pred": x.pred_kms.values}).sort_values(["origin", "h"]).reset_index(drop=True)


def _fold(cand: Path, ctl_dir: Path, k: int, sub: str):
    c = [_frame(pd.read_parquet(cand / sub / f"f{k}_s{s}.parquet")) for s in SEEDS]
    r = [_frame(pd.read_parquet(ctl_dir / f"f{k}_s{s}.parquet")) for s in SEEDS]
    for a in c + r:
        if not (a.origin.to_numpy() == c[0].origin.to_numpy()).all():
            raise SystemExit(f"fold {k}: grid mismatch")
    return c, r


def _f1(parts: list[pd.DataFrame]) -> float:
    out = []
    for s in SEEDS:
        merged = []
        for k, f in enumerate(parts, start=1):
            d = f.copy()
            d["fold"] = k
            d[f"s{s}"] = d["pred"].values
            merged.append(d[["origin", "h", "fold", f"s{s}"]])
        out.append(roll3._f1(pd.concat(merged, ignore_index=True), [s])[s])
    return float(np.mean(out))


def main(cand: Path) -> dict:
    ace = evaluate._ace()[0]
    S, R, G = [], [], []
    per = {}
    cf, rf = [], []
    for k in range(1, 7):
        eid, sub = CONTROL[k]
        c, r = _fold(cand, core.RUNS / eid / sub, k, "roll3" if k <= 3 else "roll3x")
        t = pd.DatetimeIndex(c[0].origin) + pd.to_timedelta(c[0].h, unit="h")
        y = ace.reindex(t).values
        m = np.isfinite(y)
        qc = np.mean([a.pred.values for a in c], axis=0)
        qr = np.mean([a.pred.values for a in r], axis=0)
        sc = ((qc - y) ** 2)[m]
        sr = ((qr - y) ** 2)[m]
        S.append(sc)
        R.append(sr)
        G.append(np.asarray(t)[m])
        per[k] = float(sc.sum() / sr.sum() - 1)
        cf += c
        rf += r
    S = np.concatenate(S)
    R = np.concatenate(R)
    G = np.concatenate(G)
    rel = float(S.sum() / R.sum() - 1)
    o = np.argsort(G)
    S, R, G = S[o], R[o], G[o]
    Gv = pd.DatetimeIndex(G).tz_convert("UTC") if pd.DatetimeIndex(G).tz else pd.DatetimeIndex(G)
    off = ((Gv - Gv[0]).total_seconds() // 3600).astype("int64")
    blk = (off // (7 * 24)).astype("int64")
    ss = np.bincount(blk, weights=S)
    rr = np.bincount(blk, weights=R)
    nb = len(ss)
    rng = np.random.default_rng(0)
    draws = np.empty(4000)
    for i in range(4000):
        j = rng.integers(0, nb, nb)
        draws[i] = ss[j].sum() / rr[j].sum() - 1
    lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    f1c, f1r = _f1(cf), _f1(rf)
    improved = sum(v <= 1e-9 for v in per.values())
    gates = {"rel_le_minus1pct": rel <= -0.01,
             "ci_upper_lt_0": hi < 0,
             "folds_not_worse_ge_4": improved >= 4,
             "event_f1_nonregression": f1c >= f1r - 1e-9}
    res = {"pooled_rel": rel, "ci95": [lo, hi], "per_fold": per,
           "folds_not_worse": improved, "folds_total": 6,
           "f1": f1c, "f1_control": f1r, "gates": gates,
           "passed": all(gates.values())}
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(Path(sys.argv[1]))
