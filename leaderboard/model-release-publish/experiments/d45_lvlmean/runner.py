"""ExpOS runner D45: level-branch 3-seed mean + fixed event branch (hard switch).

E0003-identical pipeline except the head: ia.build rows, existing_all cols +
17 proxy-rule SUVI longitude strips (d1 ST import only), per-period expanding
train with 72h purge, causal forecast-time-aligned strips, real poison test,
200/period availability manifest. The ONLY change vs the E0003 recipe is the
head: two short band experts (1-6, 7-24), each refit once per level seed
(0, 1, 2) on that band's rows of the same causal train frame, whose test
predictions on h<=24 are the fixed uniform 1/3 mean; plus one E0003-line
passer seed 1 for h>=25. The event side is a single fixed head, never averaged
(X0075 guard). Test predictions are the hard switch over horizons at h=24
(hsplit rules); no fitted choice consumes any selection statistic. Fresh
refits from inputs; prior prediction files are never read. GPU build
(XGBoost device=cuda, 3 period jobs x n_jobs 4 = 12 threads total).

`--pilot` runs the identical pipeline on a 6h train/test grid (evaluate still
needs all three period files, so all periods are predicted at pilot density).
`EXPOS_SMOKE=1` runs the same code on capped fits/origins (~1min) with a
smoke-lite manifest (S3 manifest skipped, poison kept) outside expos before
any registered run.

Writes $EXPOS_OUT/predictions_{dev,selection,audit}.parquet (full origin x
72h), poison.json (real future-poison test, exact key "passed"),
availability_manifest.json (200 samples/period), selection.json (fixed-rule
record with level-seed spread stats, full/repro conditions, E0102 fragility
note) and importance.json (per-side gain).
"""

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
sys.path.insert(0, str(Path(__file__).parent))
import info_ablation as ia  # noqa: E402  (read-only import)
import strips as ST  # noqa: E402  (read-only import)
import hsplit as HS  # noqa: E402  (read-only switch-rule reference)
import lvlmean as LM  # noqa: E402
from existing_all import full_grid as base_grid  # noqa: E402  (read-only import)
from existing_all import write  # noqa: E402

OUT = Path(os.environ.get("EXPOS_OUT", "."))
SMOKE = os.environ.get("EXPOS_SMOKE") == "1"
PERIODS = dict(ia.PERIODS)
SCRATCH = LB / "store/scratch/worker-d45"

FCACHE_VERSION = 1
FCACHE_CODE = ("experiments/d45_lvlmean/runner.py", "experiments/d45_lvlmean/lvlmean.py",
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


def importance_of(model, cols: list) -> dict:
    imp = pd.Series(np.asarray(model.feature_importances_, dtype=float), index=cols)
    return {str(k): float(v) for k, v in imp.items()}


def run_period(name: str, df, cols: list, meta, block, params: dict, train_cap=None, origin_cap=None) -> dict:
    """Fit level band experts x3 seeds + fixed event head; hard-switch predict."""
    lo, hi = (pd.Timestamp(t, tz="UTC") for t in PERIODS[name])
    tr = df[df["target_t"] < lo - pd.Timedelta("72h")]
    if train_cap:
        tr = tr.head(int(train_cap))
    tr_s1 = tr[tr["h"] <= 6]
    tr_s2 = tr[(tr["h"] >= 7) & (tr["h"] <= HS.SPLIT_H)]
    side_rows = {"1-6": int(len(tr_s1)), "7-24": int(len(tr_s2)), "25-72": int(len(tr))}
    if min(side_rows.values()) == 0:
        raise SystemExit(f"{name}: empty side train {side_rows}; refusing to fit.")
    short = {}
    for s in LM.check_seeds(LM.LEVEL_SEEDS):
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
    # hard switch by horizon: level side is the 3-seed mean, event side is the
    # single fixed head; no weight, no blending, no averaging on the event side
    pred_s1 = LM.mean_predictions({s: short[s][0].predict(te.loc[m1, cols]) for s in LM.LEVEL_SEEDS})
    pred_s2 = LM.mean_predictions({s: short[s][1].predict(te.loc[m2, cols]) for s in LM.LEVEL_SEEDS})
    pred_long = np.asarray(m_long.predict(te.loc[m3, cols]), dtype=np.float64)
    for arr, side in ((pred_s1, "short/1-6"), (pred_s2, "short/7-24"), (pred_long, "long/E0003-line")):
        if not np.isfinite(np.asarray(arr, dtype=float)).all():
            raise SystemExit(f"{name}/{side}: non-finite predictions; refusing to write.")
    pred = np.empty(len(te), dtype=np.float64)
    pred[m1], pred[m2], pred[m3] = pred_s1, pred_s2, pred_long
    if not np.isfinite(pred).all():
        raise SystemExit(f"{name}: non-finite stitched predictions; refusing to write.")
    # prediction-only level-seed spread (no truth): smoothing evidence
    spread = {}
    for tag, mask, heads in (("1-6", m1, 0), ("7-24", m2, 1)):
        stack = np.stack([np.asarray(short[s][heads].predict(te.loc[mask, cols]), dtype=np.float64)
                          for s in LM.LEVEL_SEEDS], axis=0)
        spread[tag] = {"mean_std": float(np.std(stack, axis=0).mean()),
                       "mean_range": float((stack.max(axis=0) - stack.min(axis=0)).mean()),
                       "n": int(mask.sum())}
    write(te, pred, name)
    importance = {"1-6": importance_of(short[LM.LEVEL_SEEDS[0]][0], cols),
                  "7-24": importance_of(short[LM.LEVEL_SEEDS[0]][1], cols),
                  "25-72-line": importance_of(m_long, cols)}
    strip_share = {}
    for side, imp in importance.items():
        tot = sum(imp.values()) or 1.0
        strip_share[side] = float(sum(v for k, v in imp.items() if k.startswith(("strip_", "img_"))) / tot)
    print(f"{name} train={len(tr)} test={len(te)} origins={len(te_origins)} "
          f"side_rows={side_rows} strip_share={ {k: round(v, 4) for k, v in strip_share.items()} } "
          f"level_spread={ {k: round(v['mean_std'], 3) for k, v in spread.items()} }",
          flush=True)
    return {"name": name, "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_origins": int(len(te_origins)), "side_rows": side_rows,
            "stale_frac": float(te["img_stale"].mean()), "selection": sel,
            "importance": importance, "strip_share": strip_share, "level_spread": spread}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--manifest-n", type=int, default=200)
    a = ap.parse_args()
    freq = "6h" if (a.pilot or SMOKE) else "3h"
    train_cap = 4000 if SMOKE else None
    origin_cap = 6 if SMOKE else None
    print(f"d45-lvlmean smoke={SMOKE} pilot={a.pilot} grid={freq} "
          f"level_seeds={list(LM.check_seeds(LM.LEVEL_SEEDS))} event_seed={LM.check_event_seed(LM.EVENT_SEED)} "
          f"split_h={HS.SPLIT_H} device=cuda", flush=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    params = LM.fit_params()
    assert params.get("device") == "cuda", "XGBoost device=cuda required"
    assert params.get("tree_method") == "hist", "E0003 hist recipe"
    assert int(params.get("n_jobs", 0)) <= 4, "n_jobs<=4 keeps 3-period total within budget"
    assert "cuda" in json.dumps(params).lower()
    assert LM.check_split()["split_h"] == 24
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

    with ThreadPoolExecutor(max_workers=LM.FIT_JOBS) as ex:
        futs = {n: ex.submit(_run, (n, PERIODS[n])) for n in PERIODS}
        infos = {n: f.result() for n, f in futs.items()}

    if SMOKE:
        # smoke-lite: predictions + poison only, manifest skipped
        (OUT / "selection.json").write_text(json.dumps(
            {"rule": LM.SELECT_RULE, "full_rule": LM.FULL_RULE, "smoke": True}, indent=1))
        (OUT / "importance.json").write_text(json.dumps(
            {n: infos[n]["importance"] for n in PERIODS}, indent=1, default=str))
        (OUT / "availability_manifest.json").write_text(json.dumps(
            {"smoke": True, "summary": "smoke-lite, S3 manifest skipped"}, indent=1))
    else:
        (OUT / "selection.json").write_text(json.dumps(
            {"rule": LM.SELECT_RULE, "full_rule": LM.FULL_RULE,
             "split_h": HS.SPLIT_H, "short": ["1-6", "7-24"], "long": "25-72/E0003-line",
             "level_seeds": list(LM.LEVEL_SEEDS), "event_seed": LM.EVENT_SEED,
             "level_mean": "fixed uniform 1/3 over level seeds, event side never averaged",
             "fit_params": params, "fitted_choice": None,
             "fragility_note": ("E0102 is the seed-fragile sole submit candidate "
                                "(1/3 seeds pass, X0097: means +0.52/-1.08/-1.74, F1 "
                                ".418/.414/.389); a D45 pass is report-only, never "
                                "sole-pass robustness; no freeze/official-score."),
             "level_spread": {n: infos[n]["level_spread"] for n in PERIODS},
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

    if not SMOKE:
        manifest = ST.build_manifest({k: PERIODS[k] for k in PERIODS}, n_per=a.manifest_n)
        manifest["created_utc"] = ST.utcnow()
        manifest["proxy_rule"] = ("newest slot S<=O-1h with status ok, s3_modified<=O, "
                                  "local bytes present, finite cells; else newest earlier "
                                  "qualifying + img_age_h; none -> NaN + img_stale=1")
        manifest["head"] = ("level branch 3-seed uniform mean (h<=24) + fixed event head "
                            "seed 1 (h>=25), hard switch h=24, no weight/blend")
        manifest["split_h"] = HS.SPLIT_H
        manifest["scratch"] = str(SCRATCH)
        (OUT / "availability_manifest.json").write_text(
            json.dumps(manifest, indent=1, default=ST.manifest_json_default))
        print("manifest:", json.dumps(manifest["summary"]), flush=True)


if __name__ == "__main__":
    main()
