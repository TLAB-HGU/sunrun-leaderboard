"""Shared pieces of the e2000fs_calib2_detroute figures.

The detector figures read the frozen study inputs (`analysis/A0014/outputs/cache`) and the archived run
`research/e2000fs_calib2-20261031/runs/detroute_v1`; nothing here fits a new model except `refit_detector`,
which re-runs the archived recipe only to reproduce it. Every script verifies the reproduction against the
archived booster and the leaderboard MSE before it writes a figure.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402

ROOT = Path("/home/t-lab01/sunrun")
RUN = ROOT / "research/e2000fs_calib2-20261031/runs/detroute_v1"
BODY = ROOT / "research/e2000fs_calib-20261030/runs/tweedie15_v1/forecast.npz"
TAIL = ROOT / "research/e2000fs_calib2-20261031/runs/mixture_v1/forecast.npz"
OUT = Path(__file__).resolve().parents[1]
LANE = ROOT / "research/e2000fs_calib-20261030/tweedie_lane.py"

TAU = 0.234
BANDS = (0, 24, 48)
STAGES = ("f1", "final")
CUTOFF = {"f1": pd.Timestamp("2026-02-26", tz="UTC"), "final": pd.Timestamp("2026-05-29", tz="UTC")}
ROUNDS = 400
HIGH = 550.0
# Frozen recipe of research/e2000fs_calib2-20261031/detector_lane.py (verbatim).
PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cpu",
              max_depth=4, eta=0.05, subsample=0.8, colsample_bytree=0.7, min_child_weight=20,
              reg_lambda=5, nthread=3, seed=42, verbosity=0)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def imported(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tweedie():
    return imported("calib_tweedie", LANE)


def load_features():
    """Frozen design inputs, official-window bundles and truth, exactly as the detector lane loaded them."""
    tw = tweedie()
    names, selected, origins, series, y, bundle = tw.load_data()
    f1_bundle = dict(np.load(tw.CACHE / "f1.npz"))
    off_bundle = dict(np.load(tw.CACHE / "official.npz"))
    o_f1 = pd.DatetimeIndex(f1_bundle["origins"], tz="UTC")
    o_f2 = pd.DatetimeIndex(off_bundle["origins"], tz="UTC")
    return dict(tw=tw, names=names, selected=selected, origins=origins, series=series, y=y, bundle=bundle,
                f1_bundle=f1_bundle, off_bundle=off_bundle, o_f1=o_f1, o_f2=o_f2,
                y_f1=tw.official_truth(o_f1), y_f2=tw.official_truth(o_f2))


def long_design(tw, data, bundle, start, rows, target):
    """Row-level design of one 24 h band: 237 block features plus the horizon index (the lane's long_design)."""
    x = tw.design(bundle, data["names"], data["selected"], start)
    if rows is not None:
        x = x[rows]
        target = target[rows]
    n = len(x)
    feats = np.repeat(x, 24, axis=0)
    hz = np.tile(np.arange(start + 1, start + 25), n).astype(np.float32)[:, None]
    labels = (target[:, start:start + 24] >= HIGH).astype(np.float32).reshape(-1)
    return np.hstack([feats, hz]), labels


def column_names(data):
    """Design column -> source feature name. Columns are the selected constants, then 24 leads x the selected
    varying features, then the horizon index."""
    constant = [n for n in data["names"]["constant"] if n in data["selected"]]
    varying = [n for n in data["names"]["varying"] if n in data["selected"]]
    columns = list(constant)
    for _lead in range(24):
        columns.extend(varying)
    columns.append("horizon_hours")
    assert len(columns) == 21 + 24 * 9 + 1 == 238, len(columns)
    return columns, constant, varying


def fit_rows(data, stage):
    origins = data["origins"]
    return np.flatnonzero(origins + pd.Timedelta(hours=144) <= CUTOFF[stage])


def official_arrays(data):
    """Archived arm forecasts, truth and folds in the submission order (F1 origins, then F2 origins)."""
    zb, zt = np.load(BODY), np.load(TAIL)
    body, tail = zb["predictions"].astype(np.float64), zt["predictions"].astype(np.float64)
    origins = pd.DatetimeIndex(zb["origins_ns"], tz="UTC")
    assert origins.equals(pd.DatetimeIndex(zt["origins_ns"], tz="UTC"))
    assert origins.equals(data["o_f1"].append(data["o_f2"]))
    target = np.vstack([data["y_f1"], data["y_f2"]])
    n_f1 = len(data["y_f1"])
    assert len(origins) == n_f1 + len(data["y_f2"]) == 4970
    return dict(origins=origins, body=body, tail=tail, target=target, n_f1=n_f1)


def routed_score(pred, target, n_f1):
    error = pred - target
    high = target >= HIGH
    f1, f2 = slice(0, n_f1), slice(n_f1, len(target))
    return dict(total=float(np.mean(error ** 2)),
                ge550=float(np.mean(error[high] ** 2)),
                f1=float(np.mean(error[f1] ** 2)), f2=float(np.mean(error[f2] ** 2)),
                f1_ge550=float(np.mean(error[f1][high[f1]] ** 2)),
                f2_ge550=float(np.mean(error[f2][high[f2]] ** 2)))


def archived_probabilities():
    z = np.load(RUN / "exceedance_prob.npz")
    return z["prob"].astype(np.float64), pd.DatetimeIndex(z["origins_ns"], tz="UTC")


def refit_detector(data, stage, start, evals=True):
    """Re-run the archived detector recipe for one (stage, band). With evals, record the per-round logloss."""
    rows = fit_rows(data, stage)
    bundle_ap, target_ap = ((data["f1_bundle"], data["y_f1"]) if stage == "f1"
                            else (data["off_bundle"], data["y_f2"]))
    x_tr, lab_tr = long_design(data["tw"], data, data["bundle"], start, rows, data["y"])
    x_ap, lab_ap = long_design(data["tw"], data, bundle_ap, start, None, target_ap)
    dtrain = xgb.DMatrix(x_tr, label=lab_tr, nthread=PARAMS["nthread"])
    history = None
    if evals:
        dvalid = xgb.DMatrix(x_ap, label=lab_ap, nthread=PARAMS["nthread"])
        result = {}
        model = xgb.train(PARAMS, dtrain, num_boost_round=ROUNDS,
                          evals=[(dtrain, "train"), (dvalid, "validation")], evals_result=result,
                          verbose_eval=False)
        history = pd.DataFrame({"round": np.arange(1, ROUNDS + 1),
                                "train_logloss": result["train"]["logloss"],
                                "validation_logloss": result["validation"]["logloss"]})
    else:
        model = xgb.train(PARAMS, dtrain, num_boost_round=ROUNDS, verbose_eval=False)
    return dict(model=model, history=history, x_apply=x_ap, labels_apply=lab_ap, rows=rows,
                train_rows=len(x_tr), apply_rows=len(x_ap))


def archived_booster(stage, start):
    booster = xgb.Booster()
    booster.load_model(RUN / f"detector_{stage}_b{BANDS.index(start)}.ubj")
    return booster


def receipt():
    return json.loads((OUT / "leaderboard-result.json").read_text())
