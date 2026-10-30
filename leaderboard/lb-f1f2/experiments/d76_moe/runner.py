"""ExpOS runner D76: per-lead mixture of E0003-line + D33 + D45.

One shared fit set per period (7 XGBoost fits on cuda): short 1-6h experts at
level seeds (0,1,2), short 7-24h experts at (0,1,2), long E0003-line head at
fixed event seed 1. Members are stitched from those fits (runner imports
only; no member prediction file is read). Per-lead convex weights come from
the fixed pre-rule in moe.py on dev-period OOF only; selection/audit truth
never feeds any weight. Writes predictions_{dev,selection,audit}.parquet plus
a real future-poison test, selection/weights record, importance, and proxy
manifest. Scratch is isolated to store/scratch/worker-d76 with a hash-keyed
feat cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
sys.path.insert(0, str(LB / "experiments/expos_runners"))
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments/d33_hsplit"))
sys.path.insert(0, str(LB / "experiments/d45_lvlmean"))
sys.path.insert(0, str(Path(__file__).parent))
import info_ablation as ia  # noqa: E402 (read-only import)
import strips as ST  # noqa: E402 (read-only import)
import hsplit as HS  # noqa: E402 (read-only switch-rule reference)
import lvlmean as LM  # noqa: E402 (read-only seed-rule reference)
import moe as MOE  # noqa: E402
from existing_all import full_grid as base_grid  # noqa: E402 (read-only import)
from existing_all import write  # noqa: E402

OUT = Path(os.environ.get("EXPOS_OUT", "."))
SMOKE = os.environ.get("EXPOS_SMOKE") == "1"
PERIODS = dict(ia.PERIODS)
SCRATCH = Path(os.environ.get("EXPOS_SCRATCH", str(LB / "store/scratch/worker-d76")))

FCACHE_VERSION = 1
FCACHE_CODE = ("experiments/d76_moe/runner.py", "experiments/d76_moe/moe.py",
               "experiments/d33_hsplit/hsplit.py",
               "experiments/d45_lvlmean/lvlmean.py",
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


def importance_of(model, cols: list) -> dict:
    imp = pd.Series(np.asarray(model.feature_importances_, dtype=float), index=cols)
    return {str(k): float(v) for k, v in imp.items()}


def _dev_truth_for(te: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """Dev-period labels aligned to te order via (origin,h) join on the train frame.

    Dev test is OOF relative to the dev train; only the dev period may call
    this. Selection/audit labels are never fetched for weights. Rows without
    a label stay NaN; weight MSEs use the finite-label subset only.
    """
    key = te[["origin", "h"]].copy()
    key["origin"] = pd.to_datetime(key["origin"], utc=True)
    ref = df[["origin", "h", "y"]].copy()
    ref["origin"] = pd.to_datetime(ref["origin"], utc=True)
    m = key.merge(ref, on=["origin", "h"], how="left")
    y = m["y"].to_numpy(dtype=np.float64)
    if len(y) != len(te):
        raise SystemExit("dev-OOF label join failed; refusing to fit weights.")
    if int(np.isfinite(y).sum()) < 100:
        raise SystemExit("dev-OOF finite labels too few; refusing to fit weights.")
    return y


def run_period(name: str, df, cols: list, meta, block, params: dict,
               train_cap=None, origin_cap=None) -> dict:
    """Fit the shared 7-model set and stitch the three members (no weight here)."""
    lo, hi = (pd.Timestamp(t, tz="UTC") for t in PERIODS[name])
    tr = df[df["target_t"] < lo - pd.Timedelta("72h")]
    if train_cap:
        tr = tr.head(int(train_cap))
    tr_s1 = tr[tr["h"] <= 6]
    tr_s2 = tr[(tr["h"] >= 7) & (tr["h"] <= HS.SPLIT_H)]
    side_rows = {"1-6": int(len(tr_s1)), "7-24": int(len(tr_s2)), "25-72": int(len(tr))}
    if min(side_rows.values()) == 0:
        raise SystemExit(f"{name}: empty side train {side_rows}; refusing to fit.")
    LM.check_seeds(LM.LEVEL_SEEDS)
    LM.check_event_seed(LM.EVENT_SEED)
    short = {}
    for s in LM.LEVEL_SEEDS:
        p = dict(params)
        p["random_state"] = int(s)
        short[s] = (xgb.XGBRegressor(**p).fit(tr_s1[cols], tr_s1["y"]),
                    xgb.XGBRegressor(**p).fit(tr_s2[cols], tr_s2["y"]))
    p_ev = dict(params)
    p_ev["random_state"] = LM.check_event_seed(LM.EVENT_SEED)
    m_long = xgb.XGBRegressor(**p_ev).fit(tr[cols], tr["y"])
    te_origins = df.loc[df["origin"].between(lo, hi), "origin"].unique()
    if origin_cap:
        te_origins = te_origins[:int(origin_cap)]
    if len(te_origins) == 0:
        raise SystemExit(f"{name}: no test origins; refusing to write.")
    te = base_grid(te_origins)
    te, sel = add_strips(te, meta, block)
    h = te["h"].to_numpy()
    m1 = h <= 6
    m2 = (h >= 7) & (h <= HS.SPLIT_H)
    m3 = h > HS.SPLIT_H
    pred_long_all = np.asarray(m_long.predict(te[cols]), dtype=np.float64)
    pred_s1_by_seed = {s: np.asarray(short[s][0].predict(te.loc[m1, cols]), dtype=np.float64)
                       for s in LM.LEVEL_SEEDS}
    pred_s2_by_seed = {s: np.asarray(short[s][1].predict(te.loc[m2, cols]), dtype=np.float64)
                       for s in LM.LEVEL_SEEDS}
    pred_long_m3 = np.asarray(m_long.predict(te.loc[m3, cols]), dtype=np.float64)
    # Member M0: pure E0003 line everywhere.
    mem0 = pred_long_all
    # Member M1: D33 hard switch (seed-1 shorts + long).
    mem1 = np.empty(len(te), dtype=np.float64)
    mem1[m1] = pred_s1_by_seed[LM.EVENT_SEED]
    mem1[m2] = pred_s2_by_seed[LM.EVENT_SEED]
    mem1[m3] = pred_long_m3
    # Member M2: D45 (short 3-seed mean + fixed long).
    mean_s1 = LM.mean_predictions(pred_s1_by_seed)
    mean_s2 = LM.mean_predictions(pred_s2_by_seed)
    mem2 = np.empty(len(te), dtype=np.float64)
    mem2[m1] = mean_s1
    mem2[m2] = mean_s2
    mem2[m3] = pred_long_m3
    for arr, tag in ((mem0, "e0003_line"), (mem1, "d33_hsplit"), (mem2, "d45_lvlmean")):
        if not np.isfinite(np.asarray(arr, dtype=float)).all():
            raise SystemExit(f"{name}/{tag}: non-finite member predictions; refusing to write.")
    dev_lead_mses = None
    if name == "dev":
        y = _dev_truth_for(te, df)
        dev_lead_mses = {}
        for (lo_h, hi_h), lead in zip(MOE.LEAD_BOUNDS, MOE.LEADS):
            band = (h >= lo_h) & (h <= hi_h) & np.isfinite(y)
            if int(band.sum()) < 10:
                raise SystemExit(f"dev-OOF finite labels too few on lead {lead}; refusing to fit weights.")
            dev_lead_mses[lead] = {m: float(np.mean((a[band] - y[band]) ** 2)) for m, a in
                                   (("e0003_line", mem0), ("d33_hsplit", mem1), ("d45_lvlmean", mem2))}
        print(f"{name} dev_oof_n={int(np.isfinite(y).sum())}/{len(y)} "
              f"dev_lead_mses={ {k: {m: round(v, 2) for m, v in d.items()} for k, d in dev_lead_mses.items()} }",
              flush=True)
    importance = {"1-6/seed0": importance_of(short[LM.LEVEL_SEEDS[0]][0], cols),
                  "7-24/seed0": importance_of(short[LM.LEVEL_SEEDS[0]][1], cols),
                  "25-72-line": importance_of(m_long, cols)}
    strip_share = {}
    for side, imp in importance.items():
        tot = sum(imp.values()) or 1.0
        strip_share[side] = float(sum(v for k, v in imp.items() if k.startswith(("strip_", "img_"))) / tot)
    print(f"{name} train={len(tr)} test={len(te)} origins={len(te_origins)} "
          f"side_rows={side_rows} strip_share={ {k: round(v, 4) for k, v in strip_share.items()} }",
          flush=True)
    return {"name": name, "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_origins": int(len(te_origins)), "side_rows": side_rows,
            "stale_frac": float(te["img_stale"].mean()), "selection": sel,
            "importance": importance, "strip_share": strip_share,
            "te": te, "mem0": mem0, "mem1": mem1, "mem2": mem2,
            "dev_lead_mses": dev_lead_mses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--manifest-n", type=int, default=200)
    a = ap.parse_args()
    freq = "6h" if (a.pilot or SMOKE) else "3h"
    train_cap = 4000 if SMOKE else None
    origin_cap = 6 if SMOKE else None
    print(f"d76-moe smoke={SMOKE} pilot={a.pilot} grid={freq} "
          f"members={list(MOE.MEMBERS)} leads={list(MOE.LEADS)} level_seeds={list(LM.LEVEL_SEEDS)} "
          f"event_seed={LM.EVENT_SEED} split_h={HS.SPLIT_H} device=cuda", flush=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    assert Path(str(SCRATCH)).name in ("worker-d76", "worker-d76f"), "scratch isolation to worker-d76/worker-d76f only"
    params = MOE.fit_params()
    assert params.get("device") == "cuda", "XGBoost device=cuda required"
    assert params.get("tree_method") == "hist", "E0003 hist recipe"
    assert int(params.get("n_jobs", 0)) <= 4, "n_jobs<=4 keeps 3-period total within budget"
    assert "cuda" in json.dumps(params).lower()
    assert MOE.check_split()["split_h"] == 24
    assert MOE.check_partition()["n_horizons"] == 72
    assert len(ST.strip_columns()) == 17

    df = build_train(freq, ST.strip_columns())
    cols = ia.arms(df.columns)["existing_all"] + ST.strip_columns()
    print(f"train rows={len(df)} strip_cols={len(ST.strip_columns())} "
          f"train_img_stale_frac={float(df['img_stale'].mean()):.4f} "
          f"fcache={_FCACHE_STATS}", flush=True)

    meta, block = ST.load_inputs()

    def _run(item):
        name, (lo, hi) = item
        return run_period(name, df, cols, meta, block, params, train_cap, origin_cap)

    with ThreadPoolExecutor(max_workers=MOE.FIT_JOBS) as ex:
        futs = {n: ex.submit(_run, (n, PERIODS[n])) for n in PERIODS}
        infos = {n: f.result() for n, f in futs.items()}

    dev_lead_mses = infos["dev"]["dev_lead_mses"]
    if dev_lead_mses is None:
        raise SystemExit("dev-OOF lead MSEs missing; refusing to fit weights.")
    weights = MOE.inverse_mse_weights_per_lead(dev_lead_mses)
    print(f"dev_lead_mses={ {k: {m: round(v, 2) for m, v in d.items()} for k, d in dev_lead_mses.items()} } "
          f"weights={ {k: {m: round(v, 4) for m, v in d.items()} for k, d in weights.items()} }", flush=True)

    for n in PERIODS:
        blended = MOE.blend_per_lead(
            {"e0003_line": infos[n]["mem0"], "d33_hsplit": infos[n]["mem1"],
             "d45_lvlmean": infos[n]["mem2"]}, weights, infos[n]["te"]["h"].to_numpy())
        write(infos[n]["te"], blended, n)
        print(f"wrote {n} blended rows={len(blended)}", flush=True)

    (OUT / "selection.json").write_text(json.dumps(
        {"rule": MOE.SELECT_RULE, "full_rule": MOE.FULL_RULE,
         "members": list(MOE.MEMBERS), "leads": list(MOE.LEADS),
         "split_h": HS.SPLIT_H,
         "level_seeds": list(LM.LEVEL_SEEDS), "event_seed": LM.EVENT_SEED,
         "weight_rule": "per-lead inverse dev-period OOF MSE, clipped [0.05, 0.90], renormalized per lead; "
                        "dev test is OOF vs dev train; no selection/audit statistic used",
         "dev_lead_mses": dev_lead_mses, "weights": weights,
         "weight_source": "dev period only (chronological OOF)",
         "fitted_choice": {"weights": weights},
         "fit_params": params,
         "periods": {n: {"n_train": infos[n]["n_train"], "side_rows": infos[n]["side_rows"],
                         "n_test": infos[n]["n_test"],
                         "n_origins": infos[n]["n_origins"]} for n in PERIODS},
         "fcache": {"version": FCACHE_VERSION, **_FCACHE_STATS},
         "pilot": bool(a.pilot), "freq": freq, "scratch": str(SCRATCH)},
        indent=1, default=ST.manifest_json_default))
    (OUT / "importance.json").write_text(json.dumps(
        {n: infos[n]["importance"] for n in PERIODS}, indent=1, default=str))

    all_origins = pd.DatetimeIndex(
        np.concatenate([pd.DatetimeIndex(df["origin"].unique())] +
                       [pd.DatetimeIndex(infos[n]["selection"]["origin"].to_numpy()) for n in PERIODS])).unique()
    cut = pd.to_datetime(infos["audit"]["selection"]["slot"]).dropna().median().isoformat()
    poison = ST.poison_test(all_origins, cut, meta=meta, block=block)
    (OUT / "poison.json").write_text(json.dumps(poison, indent=1, default=ST.manifest_json_default))
    print("poison:", json.dumps({k: poison[k] for k in ("passed", "cutoff_utc", "pre_invariant",
          "n_pre_rows", "n_post_rows", "moved_columns")}, default=str), flush=True)
    if not poison["passed"]:
        raise SystemExit("poison test failed Refusing to present predictions as causal.")

    if SMOKE:
        (OUT / "availability_manifest.json").write_text(json.dumps(
            {"smoke": True, "summary": "smoke-lite, S3 manifest skipped"}, indent=1))
    else:
        manifest = ST.build_manifest({k: PERIODS[k] for k in PERIODS}, n_per=a.manifest_n)
        manifest["created_utc"] = ST.utcnow()
        manifest["proxy_rule"] = ("newest slot S<=O-1h with status ok, s3_modified<=O, "
                                  "local bytes present, finite cells; else newest earlier "
                                  "qualifying + img_age_h; none -> NaN + img_stale=1")
        manifest["head"] = ("per-lead mixture e0003_line+d33_hsplit+d45_lvlmean, "
                            "per-lead inverse dev-OOF MSE weights, stitched at h=24")
        manifest["split_h"] = HS.SPLIT_H
        manifest["weights"] = weights
        manifest["scratch"] = str(SCRATCH)
        (OUT / "availability_manifest.json").write_text(
            json.dumps(manifest, indent=1, default=ST.manifest_json_default))
        print("manifest:", json.dumps(manifest["summary"]), flush=True)


if __name__ == "__main__":
    main()
