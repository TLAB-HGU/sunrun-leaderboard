"""D133 T1b-redo rocv: E0105 head structure with genuine per-cell seed propagation.

E0105-identical pipeline except seed handling: ia.build rows at full 3h
density, existing_all cols + 17 proxy-rule SUVI longitude strips (ST import
only), per-fold expanding train with 72h purge (target_t < fold_start - 72h),
causal forecast-time-aligned strips, real poison test. The head keeps the
E0105 structure: two short band experts (1-6, 7-24, each fit only on its
band's rows of the same causal train frame) plus one E0003-line single head
(fit on all rows of the same train frame); test predictions are the hard
switch over horizons at h=24 (hsplit rules). No weight, no blending, no
averaging. Fixed preregistered split at h=24 (hsplit rules); no selection
statistic is consumed anywhere. Fresh refits from inputs; prior prediction
files are never read. GPU build (XGBoost device=cuda, sequential fits x
n_jobs 4 = 4 threads total, shared budget <= 20).

Redo deviation from E0105 (preregistered): every (fold, seed) cell fits its
own heads -- level (short) experts with random_state=s, event (long) head
with random_state=100+s -- and writes its own f{k}_s{s}.parquet. No
intra-cell seed mean, no identical copies across seed cells; seed SE is
estimated from genuinely different fits, never 0 by construction.

Loop copies experiments/expos_runners/e0003_rocv.py: for each of the 8
expos.rocv folds and each seed in SEEDS, train as above and predict every
fold origin x 72h, writing $EXPOS_OUT/rocv/f{k}_s{s}.parquet (columns
origin_last_input_utc, horizon_hours, pred_kms). Seed changes random_state
only (level s, event 100+s). Also writes $EXPOS_OUT/poison.json (real
ST.poison_test, exact key "passed") and $EXPOS_OUT/selection.json
(fixed-rule record, no fitted choice).

EXPOS_SMOKE=1 runs fold 0 / seed 0 only on a 24h train grid with 20 trees on
CPU (~1 min pipeline gate); it is not an evaluation.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB))
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
sys.path.insert(0, str(LB / "experiments/expos_runners"))
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments/d33_hsplit"))
sys.path.insert(0, str(LB / "experiments/d45_lvlmean"))
import info_ablation as ia  # noqa: E402  (read-only import)
import strips as ST  # noqa: E402  (read-only import)
import hsplit as HS  # noqa: E402  (read-only switch-rule reference)
import lvlmean as LM  # noqa: E402  (read-only params reference: E0003 + cuda/threads)
from existing_all import full_grid as base_grid  # noqa: E402  (read-only row admin)
from expos import rocv  # noqa: E402

OUT = Path(os.environ.get("EXPOS_OUT", ".")) / "rocv"
SEEDS = (0, 1, 2)
EVENT_SEED_OFFSET = 100
SMOKE = os.environ.get("EXPOS_SMOKE") == "1"
SCRATCH = LB / "store/scratch/worker-t1br"

FCACHE_VERSION = 1
FCACHE_CODE = ("experiments/d133_t1rocv/e0105r_rocv.py",
               "experiments/d45_lvlmean/lvlmean.py",
               "experiments/d33_hsplit/hsplit.py",
               "experiments/d32_bandexpert/bands.py",
               "experiments/d1_longitude_strips/strips.py",
               "experiments/headroom_audit/info_ablation.py",
               "experiments/headroom_audit/headroom.py",
               "experiments/expos_runners/existing_all.py")
FCACHE_DATA = ("store/suvi-fusion/suvi-hourly-v2.parquet", "store/ch-v1/ch-hourly.parquet",
               "store/ch-v1/ace.parquet")
_FCACHE_STATS = {"hits": 0, "misses": 0}


def _file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _origins_sha(df) -> str:
    o = pd.DatetimeIndex(pd.to_datetime(df["origin"], utc=True)).asi8.tobytes()
    return hashlib.sha256(o + np.ascontiguousarray(df["h"].to_numpy()).tobytes()).hexdigest()


def fcache_key(kind: str, payload: dict) -> str:
    blob = json.dumps({"v": FCACHE_VERSION, "kind": kind, "payload": payload,
                       "code": {f: _file_sha(LB / f) for f in FCACHE_CODE},
                       "data": {f: [(LB / f).stat().st_size, (LB / f).stat().st_mtime_ns]
                                for f in FCACHE_DATA}}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def fcache_get(key):
    d = SCRATCH / "featcache"
    pq, js = d / (key + ".parquet"), d / (key + ".json")
    if not (pq.is_file() and js.is_file()):
        return None
    try:
        meta = json.loads(js.read_text())
        if meta.get("key") != key or meta.get("version") != FCACHE_VERSION:
            return None
        df = pd.read_parquet(pq)
        if len(df) != meta.get("rows") or list(df.columns) != meta.get("columns"):
            return None
        if _origins_sha(df) != meta.get("origins_sha"):
            return None
        return df
    except Exception:
        return None


def fcache_put(key, df):
    d = SCRATCH / "featcache"
    d.mkdir(parents=True, exist_ok=True)
    meta = {"key": key, "version": FCACHE_VERSION, "rows": int(len(df)),
            "columns": [str(c) for c in df.columns], "origins_sha": _origins_sha(df)}
    tmp = d / (key + ".tmp.parquet")
    df.to_parquet(tmp, index=False)
    (d / (key + ".json")).write_text(json.dumps(meta, sort_keys=True))
    tmp.rename(d / (key + ".parquet"))
    return df


def add_strips(df: pd.DataFrame, meta, block) -> pd.DataFrame:
    """Join per-(origin,h) strip features onto a long (origin,h) table."""
    origins = pd.DatetimeIndex(df["origin"].unique())
    sel = ST.select_slots(origins, meta, block)
    by_origin = {o: i for i, o in enumerate(sel["origin"].to_numpy())}
    order = np.array([by_origin[o] for o in df["origin"].to_numpy()])
    feats, _ = ST.strip_features(origins, meta, block, selection=sel)
    nrow = len(df)
    for c in ST.strip_columns():
        v = feats[c].to_numpy().reshape(len(origins), 72)
        df[c] = v[order, (df["h"].to_numpy() - 1)].astype(np.float32)
    assert len(df) == nrow
    return df, sel


def build_train(freq: str, cols_extra: list) -> pd.DataFrame:
    """Strip-augmented train frame, reused via the hash-keyed feat cache."""
    key = fcache_key("train", {"freq": freq, "extra": cols_extra})
    hit = fcache_get(key)
    if hit is not None:
        _FCACHE_STATS["hits"] += 1
        return hit
    _FCACHE_STATS["misses"] += 1
    df = ia.build(freq=freq)
    meta, block = ST.load_inputs()
    df, _ = add_strips(df, meta, block)
    return fcache_put(key, df)


def fit_params() -> dict:
    """E0105 recipe params (E0003 params + cuda/threads); seeds set per head."""
    p = LM.fit_params()
    if SMOKE:
        p = dict(p, device="cpu", n_estimators=20)
    return p


def level_seed(s: int) -> int:
    """Preregistered level-branch seed for cell s: random_state=s."""
    if int(s) not in SEEDS:
        raise ValueError(f"redo allows only seeds {SEEDS}, got {s}")
    return int(s)


def event_seed(s: int) -> int:
    """Preregistered event-branch seed for cell s: random_state=100+s."""
    if int(s) not in SEEDS:
        raise ValueError(f"redo allows only seeds {SEEDS}, got {s}")
    return EVENT_SEED_OFFSET + int(s)


def run_fold_seed(k: int, s: int, df, cols: list, meta, block, params: dict) -> dict:
    """Fit this cell's own E0105-structure head, hard-switch predict, write f{k}_s{s}.

    Level (short) experts use random_state=s, the event (long) head uses
    random_state=100+s; every cell fits fresh, so seed cells always differ.
    """
    ls, es = level_seed(s), event_seed(s)
    lo, hi = rocv.fold_bounds(k)
    tr = df[df["target_t"] < lo - pd.Timedelta("72h")]
    tr_s1 = tr[tr["h"] <= 6]
    tr_s2 = tr[(tr["h"] >= 7) & (tr["h"] <= HS.SPLIT_H)]
    if min(len(tr_s1), len(tr_s2), len(tr)) == 0:
        raise SystemExit(f"fold {k}: empty side train; refusing to fit.")
    p_lv = dict(params)
    p_lv["random_state"] = ls
    m_s1 = xgb.XGBRegressor(**p_lv).fit(tr_s1[cols], tr_s1["y"])
    m_s2 = xgb.XGBRegressor(**p_lv).fit(tr_s2[cols], tr_s2["y"])
    p_ev = dict(params)
    p_ev["random_state"] = es
    m_long = xgb.XGBRegressor(**p_ev).fit(tr[cols], tr["y"])
    te_origins = df.loc[df["origin"].between(lo, hi), "origin"].unique()
    if SMOKE:
        te_origins = te_origins[:4]
    if len(te_origins) == 0:
        raise SystemExit(f"fold {k}: no test origins; refusing to write.")
    te = base_grid(te_origins)
    te, sel = add_strips(te, meta, block)
    h = te["h"].to_numpy()
    m1 = h <= 6
    m2 = (h >= 7) & (h <= HS.SPLIT_H)
    m3 = h > HS.SPLIT_H
    # hard switch by horizon: no weight, no blending, no averaging
    pred = np.empty(len(te), dtype=np.float64)
    for mask, model, side in ((m1, m_s1, "short/1-6"), (m2, m_s2, "short/7-24"), (m3, m_long, "long/E0003-line")):
        p = model.predict(te.loc[mask, cols])
        if not np.isfinite(np.asarray(p, dtype=float)).all():
            raise SystemExit(f"fold {k} seed {s}/{side}: non-finite predictions; refusing to write.")
        pred[mask] = np.asarray(p, dtype=np.float64)
    if not np.isfinite(pred).all():
        raise SystemExit(f"fold {k} seed {s}: non-finite stitched predictions; refusing to write.")
    pd.DataFrame({"origin_last_input_utc": te["origin"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "horizon_hours": te["h"].astype(int), "pred_kms": pred.astype(float)}
                 ).to_parquet(OUT / f"f{k}_s{s}.parquet", index=False)
    print(f"fold {k} seed {s} level_seed={ls} event_seed={es} "
          f"train={len(tr)} test={len(te)} origins={len(te_origins)}", flush=True)
    return {"fold": k, "seed": s, "level_seed": ls, "event_seed": es,
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_origins": int(len(te_origins)), "selection": sel}


def main():
    folds = [0] if SMOKE else list(range(len(rocv.FOLDS)))
    seeds = [0] if SMOKE else list(SEEDS)
    freq = "24h" if SMOKE else "3h"
    print(f"d133-t1br-rocv smoke={SMOKE} grid={freq} folds={folds} seeds={seeds} "
          f"split_h={HS.SPLIT_H} recipe=E0105-redo", flush=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    probe = fit_params()
    if SMOKE:
        assert probe["n_estimators"] == 20
    else:
        assert probe.get("device") == "cuda", "XGBoost device=cuda required"
        assert probe.get("tree_method") == "hist", "E0003 hist recipe"
        assert int(probe.get("n_jobs", 0)) <= 4, "n_jobs<=4 keeps single-GPU budget"
        assert "cuda" in json.dumps(probe).lower()
    assert HS.check_split()["split_h"] == 24
    assert LM.check_split()["split_h"] == 24
    assert [level_seed(s) for s in seeds] == ([0] if SMOKE else [0, 1, 2])
    assert [event_seed(s) for s in seeds] == ([100] if SMOKE else [100, 101, 102])
    for k, v in ia.PARAMS.items():
        if k in ("n_jobs", "random_state"):
            continue
        if SMOKE and k == "n_estimators":
            continue
        assert probe[k] == v, f"param {k} differs from E0003 recipe"

    df = build_train(freq, ST.strip_columns())
    cols = ia.arms(df.columns)["existing_all"] + ST.strip_columns()
    assert len(ST.strip_columns()) == 17
    print(f"train rows={len(df)} strip_cols={len(ST.strip_columns())} "
          f"train_img_stale_frac={float(df['img_stale'].mean()):.4f} "
          f"fcache={_FCACHE_STATS}", flush=True)

    meta, block = ST.load_inputs()
    params = fit_params()
    infos = []
    for k in folds:
        for s in seeds:
            infos.append(run_fold_seed(k, s, df, cols, meta, block, params))

    all_origins = pd.DatetimeIndex(
        np.concatenate([pd.DatetimeIndex(df["origin"].unique())] +
                       [pd.DatetimeIndex(i["selection"]["origin"].to_numpy()) for i in infos])).unique()
    cut = pd.to_datetime(infos[-1]["selection"]["slot"]).dropna().median().isoformat()
    poison = ST.poison_test(all_origins, cut, meta=meta, block=block)
    (OUT.parent / "poison.json").write_text(json.dumps(poison, indent=1, default=ST.manifest_json_default))
    print("poison:", json.dumps({kk: poison[kk] for kk in ("passed", "cutoff_utc", "pre_invariant",
          "n_pre_rows", "n_post_rows", "moved_columns")}, default=str), flush=True)
    if not poison["passed"]:
        raise SystemExit("poison test failed Refusing to present predictions as causal.")

    (OUT.parent / "selection.json").write_text(json.dumps(
        {"rule": "E0105 head structure (short band experts 1-6/7-24 + E0003-line long head, "
                 "hard switch h=24, no weight/blend) with genuine per-cell seed propagation "
                 "(level random_state=s, event random_state=100+s, fresh fits per cell, no "
                 "intra-cell mean) under rocv (8 folds x 3 seeds, full 3h grid, fold-72h purge); "
                 + HS.SELECT_RULE,
         "recipe": "E0105-redo", "split_h": HS.SPLIT_H, "short": ["1-6", "7-24"], "long": "25-72/E0003-line",
         "level_seed_rule": "random_state=s", "event_seed_rule": "random_state=100+s",
         "event_seed_offset": EVENT_SEED_OFFSET,
         "seed_cells": "per-cell fits: every f{k}_s{s} from its own level/event heads; "
                        "seed SE estimated from different fits, never 0 by construction",
         "folds": folds, "seeds": seeds, "freq": freq, "smoke": bool(SMOKE),
         "fit_params": params,
         "fitted_choice": None,
         "cells": [{"fold": i["fold"], "seed": i["seed"], "level_seed": i["level_seed"],
                     "event_seed": i["event_seed"], "n_train": i["n_train"],
                     "n_test": i["n_test"], "n_origins": i["n_origins"]} for i in infos],
         "fcache": {"version": FCACHE_VERSION, **_FCACHE_STATS},
         "scratch": str(SCRATCH)},
        indent=1, default=ST.manifest_json_default))
    print(f"done folds={folds} seeds={seeds} smoke={SMOKE}", flush=True)


if __name__ == "__main__":
    main()
