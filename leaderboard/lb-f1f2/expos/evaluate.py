"""Fixed evaluator: preofficial three-period scoring and the single official scoring of the frozen candidate."""
import json
import sys

import numpy as np
import pandas as pd

from . import core

PERIODS = {"dev": ("2025-08-04T00:00:00Z", "2025-11-27T23:00:00Z"),
           "selection": ("2025-12-04T00:00:00Z", "2026-02-25T23:00:00Z"),
           "audit": ("2026-03-04T00:00:00Z", "2026-05-28T23:00:00Z")}
BANDS = ((1, 6), (7, 24), (25, 48), (49, 72))
GATE_MIN_PERIODS, GATE_MEAN_MAX = 2, -0.02
OFFICIAL = {"snapshot": "/home/t-lab01/.cache/huggingface/hub/datasets--tlabtlab--sunrun-lb-store/snapshots/"
                        "bcf5d5417d99eaa331ddca40581d48953e09a58e",
            "champion": "store/mse20-four-regimes/run-20261003/baseline-replay.parquet",
            "total_mse_max": 3289.0890348547073,
            "cells": ("low_low_high", "low_high_high", "high_low_high", "high_high_high"), "cell_max": 3800.0}


def _ace():
    a = pd.read_parquet(core.LB / "store/ch-v1/ace.parquet")
    a.index = pd.to_datetime(a.pop("timestamp_utc"), utc=True).dt.floor("h")
    a = a[~a.index.duplicated(keep="last")].sort_index()
    a = a.reindex(pd.date_range(a.index.min(), a.index.max(), freq="h", tz="UTC"))
    filled = a["filled_speed_kms"].ffill()
    observed = (~a["was_missing"].fillna(True).astype(bool)) & a["filled_speed_kms"].notna()
    return filled, observed


def _load(path):
    p = pd.read_parquet(path)
    need = {"origin_last_input_utc", "horizon_hours", "pred_kms"}
    if not need <= set(p.columns):
        raise core.ExposError(f"{path.name} needs columns {sorted(need)}")
    p = pd.DataFrame({"origin": pd.to_datetime(p["origin_last_input_utc"], utc=True),
                      "h": p["horizon_hours"].astype(int), "y_pred": p["pred_kms"].astype(float)})
    if p.duplicated(["origin", "h"]).any() or not np.isfinite(p["y_pred"]).all() or not p["h"].between(1, 72).all():
        raise core.ExposError(f"{path.name}: duplicate keys, non-finite predictions or horizons outside 1..72")
    if (p.groupby("origin")["h"].size() != 72).any():
        raise core.ExposError(f"{path.name}: every origin needs all 72 horizons (predict the full grid, not only rows with truth)")
    return p


def _events(p, filled, observed):
    sys.path.insert(0, str(core.LB))
    from experiments.mse20_scope_expansion.diagnostics import unique_event_report
    r = unique_event_report(p, filled, observed, p["origin"].unique())
    m = r.get("metrics") or {}
    return {"f1": r.get("f1"), "tp": m.get("tp"), "fp": m.get("fp"), "fn": m.get("fn"),
            "indeterminate": r.get("indeterminate")}


def score_period(p, name, filled, observed):
    a, b = (pd.Timestamp(t) for t in PERIODS[name])
    p = p[p["origin"].between(a, b)].copy()
    if p.empty:
        raise core.ExposError(f"no {name} origins in predictions")
    t = p["origin"] + pd.to_timedelta(p["h"], unit="h")
    p["y"] = filled.reindex(pd.DatetimeIndex(t)).to_numpy()
    p["obs"] = observed.reindex(pd.DatetimeIndex(t)).fillna(False).to_numpy()
    q = p[p["obs"]]
    se = (q["y_pred"] - q["y"]) ** 2
    out = {"mse": float(se.mean()), "rows": int(len(q)), "origins": int(p["origin"].nunique()),
           "bands": {f"{lo}-{hi}": float(se[q["h"].between(lo, hi)].mean()) for lo, hi in BANDS},
           "events": _events(p[["origin", "h", "y_pred"]], filled, observed)}
    return out, q[["origin", "h", "y_pred", "y"]]


def _pooled_f1(periods):
    tp = sum(periods[k]["events"]["tp"] or 0 for k in periods)
    fp = sum(periods[k]["events"]["fp"] or 0 for k in periods)
    fn = sum(periods[k]["events"]["fn"] or 0 for k in periods)
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None


def evaluate(eid, reference=None):
    s = core.state()
    x = s["experiments"].get(eid)
    if not x:
        raise core.ExposError(f"unknown experiment {eid}")
    changed = core.check_protected(s)
    if changed:
        core.append("flag", id=eid, flag=f"protected_modified:{','.join(changed)}")
        raise core.ExposError(f"protected files changed: {changed}")
    filled, observed = _ace()
    out = core.RUNS / eid
    periods, rows = {}, {}
    for name in PERIODS:
        path = out / f"predictions_{name}.parquet"
        if not path.exists():
            raise core.ExposError(f"missing {path}")
        periods[name], rows[name] = score_period(_load(path), name, filled, observed)
    poison = out / "poison.json"
    poison_ok = poison.exists() and json.loads(poison.read_text()).get("passed") is True
    ref = reference or x.get("reference")
    gate = {"pass": False, "reasons": ["reference experiment (no gate)"]} if x["kind"] == "reference" else None
    rel = {}
    if gate is None:
        if not ref or not s["experiments"].get(ref, {}).get("evals"):
            raise core.ExposError("needs an evaluated reference experiment (--reference)")
        reasons = []
        for name in PERIODS:
            r = _load(core.RUNS / ref / f"predictions_{name}.parquet")
            m = rows[name].merge(r.rename(columns={"y_pred": "ref"}), on=["origin", "h"], how="inner")
            if len(m) != len(rows[name]):
                reasons.append(f"{name}: support differs from reference ({len(m)} vs {len(rows[name])} rows)")
            rel[name] = float(((m.y_pred - m.y) ** 2).mean() / ((m.ref - m.y) ** 2).mean() - 1)
        ref_f1 = _pooled_f1(s["experiments"][ref]["evals"][-1]["periods"])
        f1 = _pooled_f1(periods)
        n_imp, mean_rel = sum(v < 0 for v in rel.values()), float(np.mean(list(rel.values())))
        if n_imp < GATE_MIN_PERIODS:
            reasons.append(f"improves in {n_imp}/3 periods")
        if mean_rel > GATE_MEAN_MAX:
            reasons.append(f"mean relative {mean_rel:+.4f} > {GATE_MEAN_MAX}")
        if f1 is None or ref_f1 is None or f1 < ref_f1:
            reasons.append(f"pooled unique-event F1 {f1} vs reference {ref_f1}")
        gate = {"pass": not reasons, "reasons": reasons, "relative": rel, "mean_relative": mean_rel,
                "n_improved": n_imp, "pooled_f1": f1, "reference_pooled_f1": ref_f1, "reference": ref}
    summary = {k: round(v["mse"], 2) for k, v in periods.items()}
    summary["pooled_f1"] = _pooled_f1(periods)
    return core.append("evaluate", id=eid, periods=periods, gate=gate, poison_ok=poison_ok, summary=summary)


def official_score():
    """Score the frozen candidate once on the official window with the leaderboard scorer."""
    s = core.state()
    if not s["frozen"]:
        raise core.ExposError("nothing frozen")
    if s["official"]:
        raise core.ExposError("official scoring already done; it is allowed exactly once")
    fz = s["frozen"]
    if core.sha256(fz["predictions"]) != fz["predictions_sha256"]:
        raise core.ExposError("frozen predictions changed after freeze")
    sys.path.insert(0, str(core.LB))
    from scorer import scoring
    snap = core.Path(OFFICIAL["snapshot"])
    folds, truth = pd.read_parquet(snap / "folds.parquet"), pd.read_parquet(snap / "truth.parquet")
    naive = pd.read_parquet(snap / "naive.parquet")
    filled, observed = _ace()
    res = {}
    for name, path in (("candidate", fz["predictions"]), ("champion", core.LB / OFFICIAL["champion"])):
        pred = pd.read_parquet(path)
        err = scoring.validate(pred, folds)
        if err:
            raise core.ExposError(f"official validator rejected {name}: {err[:3]}")
        sc = scoring.score(pred, truth, folds, naive)
        res[name] = {"mse": sc["mse"], "mse_observed": sc["mse_observed"],
                     "cells": {r: v["mse"] for r, v in sc["by_regime"].items()},
                     "events": _events(_load(core.Path(path)), filled, observed)}
    c, ch = res["candidate"], res["champion"]
    gates = {"total_mse": c["mse"] <= OFFICIAL["total_mse_max"],
             **{f"cell_{k}": (c["cells"].get(k) or np.inf) < OFFICIAL["cell_max"] for k in OFFICIAL["cells"]},
             "f1_nonregression": (c["events"]["f1"] or 0) >= (ch["events"]["f1"] or 0)}
    return core.append("official", id=fz["id"], results=res, gates=gates, achieved=all(gates.values()))
