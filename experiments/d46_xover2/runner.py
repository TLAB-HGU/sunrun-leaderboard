"""ExpOS runner D46: D33-switch level + E0060-pinned-joint event branch (hard common-verdict).

Pipeline (E0003-identical rows/purge/proxy-manifest; the ONLY changes vs the
E0003 recipe are the two parent mechanisms combined by hard branching):
- Level (MSE parent E0102, D33): per seed in (0,1,2), two short band experts
  (1-6, 7-24 on band rows) + one E0003-line head (all rows) on the same
  causal train frame; test cells h<=24 from the short experts, h>=25 from
  the long head (xover2.LEVEL_RULE, source hsplit.py).
- Event (F1 parent E0060, D21): per seed in (0,1,2), one joint XGBoost-cuda
  head on existing_all + 17 strips + 4 analog cols for the PINNED key
  CH6V2/l1 (E0060 winner; no re-selection per X0074) on the same train rows
  (xover2.EVENT_RULE, key grid source joint.py/f1first.py).
- Branch (this direction): per-cell final = joint seed-mean where all 3
  joint seeds vote surge (>=600) else D33 level seed-mean (xover2.BRANCH_RULE).
  No blending anywhere. NOT the D44 vote+replay gate (5 fresh seeds,
  dev-selected raw replay) and NOT the D45 level-mean smoothing.

E0102 note (X0096/X0097): seed-fragile submit candidate (means
+0.52/-1.08/-1.74, F1 0.418/0.414/0.389); this pilot reports only, claims no
single-pass adoption, no freeze/official-score.

Writes $EXPOS_OUT/predictions_{dev,selection,audit}.parquet (full origin x
72h at the run grid), poison.json (real future-poison tests with exact key
"passed"), availability_manifest.json (200 samples/period, proxy rule) and
selection.json (parent ids, pinned key, 3-seed rule, branch fire fractions,
preregistered FULL conditions).

`--pilot` runs the identical pipeline on a 6h grid. `EXPOS_SMOKE=1` runs a
~1-minute end-to-end smoke path (capped origins/train rows, manifest S3
skipped, poison kept) outside expos before any registered run.
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
sys.path.insert(0, str(LB))
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
sys.path.insert(0, str(LB / "experiments/expos_runners"))
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments/d33_hsplit"))
sys.path.insert(0, str(Path(__file__).parent))
import info_ablation as ia  # noqa: E402
import strips as ST  # noqa: E402  (import-only: never vendored)
import hsplit as HS  # noqa: E402  (read-only split contract)
import xover2 as XO2  # noqa: E402
from existing_all import full_grid as base_grid  # noqa: E402
from existing_all import write  # noqa: E402

OUT = Path(os.environ.get("EXPOS_OUT", "."))
SMOKE = os.environ.get("EXPOS_SMOKE") == "1"
PERIODS = dict(ia.PERIODS)
SCRATCH = LB / "store/scratch/worker-d46"

PROBES = {"dev": "2025-09-15T00:00:00Z", "selection": "2026-01-05T00:00:00Z",
          "audit": "2026-04-06T00:00:00Z"}
FIT_JOBS = 3  # 3 period finals fit in parallel workers
FIT_NJOBS = 6  # 3 parallel fits x 6 threads = 18 <= 20 total CPU threads
SMOKE_ORIGINS = 12
SMOKE_TRAIN_ROWS = 20000


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
    blob = json.dumps({"v": XO2.FCACHE_VERSION, "kind": kind, "payload": payload,
                       "code": {f: _file_sha(LB / f) for f in
                                ("experiments/d46_xover2/runner.py",
                                 "experiments/d46_xover2/xover2.py")},
                       "data": {f: [(LB / f).stat().st_size, (LB / f).stat().st_mtime_ns]
                                for f in ("store/suvi-fusion/suvi-hourly-v2.parquet",
                                          "store/ch-v1/ch-hourly.parquet",
                                          "store/ch-v1/ace.parquet")}},
                      sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def add_strips(df: pd.DataFrame, meta, block) -> tuple:
    """Join per-(origin,h) strip features onto a long table (via ST import)."""
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
    d = SCRATCH / "featcache"
    pq, js = d / (key + ".parquet"), d / (key + ".json")
    if pq.is_file() and js.is_file():
        try:
            meta = json.loads(js.read_text())
            df = REAL_READ_PARQUET(pq)  # cache-hit path (poison-safe: rebuilt on miss)
            if (meta.get("key") == key and len(df) == meta.get("rows")
                    and _origins_sha(df) == meta.get("origins_sha")):
                return df
        except Exception:
            pass
    df = ia.build(freq=freq)
    meta, block = ST.load_inputs()
    df, _ = add_strips(df, meta, block)
    d.mkdir(parents=True, exist_ok=True)
    meta_rec = {"key": key, "rows": int(len(df)), "origins_sha": _origins_sha(df)}
    tmp = d / (key + ".tmp.parquet")
    df.to_parquet(tmp, index=False)
    js.write_text(json.dumps(meta_rec, sort_keys=True))
    tmp.rename(pq)
    return df


def _ace():
    a = pd.read_parquet(LB / "store/ch-v1/ace.parquet")
    a.index = pd.to_datetime(a.pop("timestamp_utc"), utc=True).dt.floor("h")
    a = a[~a.index.duplicated(keep="last")].sort_index()
    a = a.reindex(pd.date_range(a.index.min(), a.index.max(), freq="h", tz="UTC"))
    filled = a["filled_speed_kms"].ffill()
    observed = (~a["was_missing"].fillna(True).astype(bool)) & a["filled_speed_kms"].notna()
    return filled, observed


def analog_table(scratch, freq, window, origins, state, filled_s, scaler):
    keys, metric = XO2.split_config(XO2.PINNED_KEY)
    payload = {"freq": freq, "window": window, "config": XO2.PINNED_KEY,
               "origins": hashlib.sha256(
                   pd.DatetimeIndex(pd.to_datetime(origins, utc=True)).asi8.tobytes()
               ).hexdigest()}
    return XO2.fcache_table(
        scratch, "analog", payload,
        lambda: XO2.analog_long(origins, state, filled_s, keys, metric, scaler))


REAL_READ_PARQUET = pd.read_parquet


def _poisoned_read(cutoff, seed, mode):
    rng = np.random.default_rng(seed)

    def read(path, *a, **k):
        df = REAL_READ_PARQUET(path, *a, **k)
        if "timestamp_utc" in df.columns:
            t = pd.to_datetime(df["timestamp_utc"], utc=True)
        elif isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            t = idx.tz_convert("UTC") if idx.tz is not None else pd.to_datetime(idx, utc=True)
            t = pd.to_datetime(t, utc=True)
        else:
            return df
        t = pd.to_datetime(t, utc=True)
        if mode == "future":
            late = np.asarray(t > cutoff)
        else:  # pool: the pinned key's own replay pool window (positive control)
            late = np.asarray((t >= cutoff - pd.Timedelta(hours=int(XO2.WIN_HI_H))) &
                              (t <= cutoff - pd.Timedelta(hours=int(XO2.WIN_LO_H))))
        num = df.select_dtypes("number").columns
        df = df.copy()
        df[num] = df[num].astype("float64")
        if int(late.sum()) and len(num):
            df.loc[late, num] = rng.normal(1000, 500, (int(late.sum()), len(num)))
        return df

    return read


def analog_poison_check(joint_models, cols_joint, meta, block, state,
                        filled_s, scaler, probes=None, seed=0):
    """Future-poison invariance + pool-poison sensitivity for the joint path."""
    keys, metric = XO2.split_config(XO2.PINNED_KEY)
    report = {"scope": "D33 level (strips-only) + pinned-joint event path (frozen heads)",
              "pinned_key": XO2.PINNED_KEY, "probes": []}
    ok = True
    for name, iso in (probes or PROBES).items():
        o = pd.Timestamp(iso, tz="UTC")
        te = add_strips(base_grid([o]), meta, block)[0]
        rep = XO2.analog_long(pd.DatetimeIndex([o], tz="UTC"), state, filled_s,
                              keys, metric, scaler)
        te_full = te.merge(rep, on=["origin", "h"], how="left")
        preds = {s: np.asarray(joint_models[name][s].predict(te_full[cols_joint]),
                               dtype=np.float64) for s in XO2.SEEDS}
        entry = {"period": name, "origin": iso, "n_rows": int(len(rep)),
                 "unanimous": bool(XO2.common_fire_mask(preds).any())}
        try:
            pd.read_parquet = _poisoned_read(o, seed, "future")
            try:
                pstate, pfilled = XO2.build_state()
                pte = add_strips(base_grid([o]), meta, block)[0]
            finally:
                pd.read_parquet = REAL_READ_PARQUET
            prep = XO2.analog_long(pd.DatetimeIndex([o], tz="UTC"), pstate, pfilled,
                                   keys, metric, scaler)
            pte_full = pte.merge(prep, on=["origin", "h"], how="left")
            ppreds = {s: np.asarray(joint_models[name][s].predict(pte_full[cols_joint]),
                                    dtype=np.float64) for s in XO2.SEEDS}
            entry["future_invariant"] = bool(
                np.allclose(rep[["analog_best", "analog_mean3"]].to_numpy(float),
                            prep[["analog_best", "analog_mean3"]].to_numpy(float),
                            equal_nan=True)
                and all(np.allclose(preds[s], ppreds[s], equal_nan=True) for s in XO2.SEEDS))
            pd.read_parquet = _poisoned_read(o, seed + 1, "pool")
            try:
                cstate, cfilled = XO2.build_state()
                crep = XO2.analog_long(pd.DatetimeIndex([o], tz="UTC"), cstate, cfilled,
                                       keys, metric, scaler)
            finally:
                pd.read_parquet = REAL_READ_PARQUET
            entry["positive_control_changed"] = bool(
                not np.allclose(rep["analog_best"].to_numpy(float),
                                crep["analog_best"].to_numpy(float), equal_nan=True))
            entry["positive_control_sensitive"] = entry["positive_control_changed"]
        except Exception as e:  # never claim a pass on an aborted check
            entry["error"] = f"{type(e).__name__}: {e}"
        entry_ok = (entry.get("future_invariant") is True
                    and entry.get("positive_control_sensitive") is True
                    and "error" not in entry)
        entry["ok"] = entry_ok
        ok = ok and entry_ok
        print(f"poison-analog {name} {iso}: ok={entry_ok} "
              f"unanim={entry.get('unanimous')} future_inv={entry.get('future_invariant')} "
              f"ctrl={entry.get('positive_control_changed')}", flush=True)
        report["probes"].append(entry)
    report["passed"] = bool(ok)
    return report


def importance_of(model, cols: list) -> dict:
    imp = pd.Series(np.asarray(model.feature_importances_, dtype=float), index=cols)
    return {str(k): float(v) for k, v in imp.items()}


def run_period(name: str, df, cols_level: list, cols_joint: list,
               meta, block, state, filled_s, scaler) -> dict:
    """Fit D33-switch level heads + pinned-joint heads per seed, branch hard."""
    lo, hi = pd.Timestamp(PERIODS[name][0], tz="UTC"), pd.Timestamp(PERIODS[name][1], tz="UTC")
    tr = df[df["target_t"] < lo - pd.Timedelta("72h")]
    if SMOKE:
        tr = tr.head(SMOKE_TRAIN_ROWS)
    tr_s1 = tr[tr["h"].between(1, 6)]
    tr_s2 = tr[tr["h"].between(7, 24)]
    if len(tr_s1) == 0 or len(tr_s2) == 0 or len(tr) == 0:
        raise SystemExit(f"{name}: empty side train; refusing to fit.")
    keys, metric = XO2.split_config(XO2.PINNED_KEY)
    tr_rep = analog_table(SCRATCH, "tr", name, pd.DatetimeIndex(tr["origin"].unique()),
                          state, filled_s, scaler)
    tr_full = tr.merge(tr_rep, on=["origin", "h"], how="left")
    if tr_full[XO2.analog_columns()].isna().all().all():
        raise SystemExit(f"{name}: no finite analog train columns; refusing to fit.")
    te_origins = df.loc[df["origin"].between(lo, hi), "origin"].unique()
    if SMOKE:
        te_origins = pd.DatetimeIndex(te_origins)[:SMOKE_ORIGINS].to_numpy()
    if len(te_origins) == 0:
        raise SystemExit(f"{name}: no test origins; refusing to write.")
    te = base_grid(te_origins)
    te, sel = add_strips(te, meta, block)
    te_rep = analog_table(SCRATCH, "te", name, pd.DatetimeIndex(te["origin"].unique()),
                          state, filled_s, scaler)
    te_full = te.merge(te_rep, on=["origin", "h"], how="left")

    level_by_seed, joint_by_seed = {}, {}
    level_models, joint_models = {}, {}
    for s in XO2.check_seeds(XO2.SEEDS):
        lp = XO2.level_params(s)
        m_s1 = xgb.XGBRegressor(**lp).fit(tr_s1[cols_level], tr_s1["y"])
        m_s2 = xgb.XGBRegressor(**lp).fit(tr_s2[cols_level], tr_s2["y"])
        m_long = xgb.XGBRegressor(**lp).fit(tr[cols_level], tr["y"])
        h = te["h"].to_numpy()
        pred = np.empty(len(te), dtype=np.float64)
        for mask, model in ((h <= 6, m_s1), ((h >= 7) & (h <= XO2.SPLIT_H), m_s2),
                            (h > XO2.SPLIT_H, m_long)):
            p = model.predict(te.loc[mask, cols_level])
            if not np.isfinite(np.asarray(p, dtype=float)).all():
                raise SystemExit(f"{name}/level-s{s}: non-finite; refusing to write.")
            pred[mask] = np.asarray(p, dtype=np.float64)
        level_by_seed[s] = pred
        level_models[s] = (m_s1, m_s2, m_long)
        jp = XO2.joint_params(s)
        mj = xgb.XGBRegressor(**jp).fit(tr_full[cols_joint], tr_full["y"])
        pj = np.asarray(mj.predict(te_full[cols_joint]), dtype=np.float64)
        if not np.isfinite(pj).all():
            raise SystemExit(f"{name}/joint-s{s}: non-finite; refusing to write.")
        joint_by_seed[s] = pj
        joint_models[s] = mj
    final, fired = XO2.apply_branch(level_by_seed, joint_by_seed)
    votes = XO2.surge_votes(joint_by_seed)
    if not np.isfinite(final).all():
        raise SystemExit(f"{name}: non-finite branched predictions; refusing to write.")
    write(te, final, name)
    stats = XO2.branch_stats(fired, votes)
    print(f"{name} train={len(tr)} test={len(te)} origins={len(te_origins)} "
          f"fire_frac={stats['fire_frac']:.4f} n_fired={stats['n_fired']} "
          f"mean_votes={stats['mean_votes']:.2f}", flush=True)
    importance = {}
    for s in XO2.SEEDS:
        m_s1, m_s2, m_long = level_models[s]
        importance[f"level-s{s}/1-6"] = importance_of(m_s1, cols_level)
        importance[f"level-s{s}/7-24"] = importance_of(m_s2, cols_level)
        importance[f"level-s{s}/25-72-line"] = importance_of(m_long, cols_level)
        importance[f"joint-s{s}/{XO2.PINNED_KEY}"] = importance_of(joint_models[s], cols_joint)
    return {"name": name, "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_origins": int(len(te_origins)), "selection": sel,
            "stale_frac": float(te["img_stale"].mean()),
            "branch": stats, "importance": importance,
            "joint_models": joint_models}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--manifest-n", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=3, choices=(1, 3))
    a = ap.parse_args()
    freq = "6h" if a.pilot else "3h"
    print(f"d46-xover2 pilot={a.pilot} grid={freq} split_h={XO2.SPLIT_H} "
          f"pinned={XO2.PINNED_KEY} seeds={list(XO2.SEEDS)} device=cuda smoke={SMOKE}",
          flush=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    assert XO2.check_split()["split_h"] == HS.check_split()["split_h"] == 24
    assert HS.SPLIT_H == XO2.SPLIT_H == 24
    assert FIT_JOBS * FIT_NJOBS <= 20
    assert ST.strip_columns() and len(ST.strip_columns()) == 17
    assert XO2.PINNED_KEY in XO2.GRID

    df = build_train(freq, ST.strip_columns())
    cols_level = ia.arms(df.columns)["existing_all"] + ST.strip_columns()
    cols_joint = cols_level + XO2.analog_columns()
    assert not set(XO2.analog_columns()) & set(df.columns), "analog columns must be new"
    print(f"train rows={len(df)} level_cols={len(cols_level)} joint_cols={len(cols_joint)} "
          f"train_img_stale_frac={float(df['img_stale'].mean()):.4f}", flush=True)

    meta, block = ST.load_inputs()
    state, filled_s = XO2.build_state()
    dev_lo = pd.Timestamp(PERIODS["dev"][0], tz="UTC")
    scaler = XO2.scaler_from_predev(state, dev_lo)

    if a.jobs == 3:
        with ThreadPoolExecutor(max_workers=FIT_JOBS) as ex:
            futs = {name: ex.submit(run_period, name, df, cols_level, cols_joint,
                                 meta, block, state, filled_s, scaler) for name in PERIODS}
            infos = {name: f.result() for name, f in futs.items()}
    else:
        infos = {name: run_period(name, df, cols_level, cols_joint,
                              meta, block, state, filled_s, scaler) for name in PERIODS}

    (OUT / "selection.json").write_text(json.dumps(
        {"direction": "D46",
         "mse_parent": {"id": "E0102", "source": "experiments/d33_hsplit/hsplit.py",
                        "mechanism": XO2.LEVEL_RULE, "seeds": list(XO2.SEEDS),
                        "note": "E0102 seed-fragile submit candidate only (X0096/X0097 "
                                "means +0.52/-1.08/-1.74 F1 0.418/0.414/0.389); "
                                "no single-pass adoption claim, no freeze/official-score"},
         "f1_parent": {"id": "E0060", "source": "experiments/d21_f1joint/joint.py "
                                                "grid + f1first.py floor via d10_analog replay",
                       "mechanism": XO2.EVENT_RULE, "pinned_key": XO2.PINNED_KEY,
                       "grid": list(XO2.GRID), "floor": XO2.F1_FLOOR,
                       "note": "pinned (no re-selection) per X0074/X0064 instability; "
                               "dev-only provenance, no fitted choice"},
         "branch": XO2.BRANCH_RULE,
         "seeds": list(XO2.check_seeds(XO2.SEEDS)),
         "branch_fire": {name: {"fire_frac": infos[name]["branch"]["fire_frac"],
                             "n_fired": infos[name]["branch"]["n_fired"],
                             "mean_votes": infos[name]["branch"]["mean_votes"]} for name in PERIODS},
         "full_iff": XO2.FULL_IFF, "jobs": int(a.jobs), "fit_njobs": FIT_NJOBS,
         "pilot": bool(a.pilot), "smoke": SMOKE, "freq": freq,
         "scratch": str(SCRATCH)}, indent=1, default=ST.manifest_json_default))
    (OUT / "importance.json").write_text(json.dumps(
        {name: infos[name]["importance"] for name in PERIODS}, indent=1, default=str))

    all_origins = pd.DatetimeIndex(
        np.concatenate([pd.DatetimeIndex(df["origin"].unique())] +
                       [pd.DatetimeIndex(infos[name]["selection"]["origin"].to_numpy())
                        for name in PERIODS])).unique()
    cut = pd.to_datetime(infos["audit"]["selection"]["slot"]).dropna().median().isoformat()
    strip_rep = ST.poison_test(all_origins, cut, meta=meta, block=block)
    print("poison-strip:", json.dumps({k: strip_rep[k] for k in (
        "passed", "pre_invariant", "n_pre_rows", "n_post_rows")}, default=str), flush=True)
    joint_by_period = {name: infos[name]["joint_models"] for name in PERIODS}
    probes = {"dev": PROBES["dev"]} if SMOKE else dict(PROBES)
    analog_rep = analog_poison_check(joint_by_period, cols_joint, meta, block, state,
                                    filled_s, scaler, probes=probes)
    poison = {"passed": bool(strip_rep["passed"] and analog_rep["passed"]),
              "cutoff_utc": pd.to_datetime(cut, utc=True).isoformat(),
              "strip": strip_rep, "analog": analog_rep,
              "note": ("strip perturb on post-cutoff cache + pinned-key joint "
                       "future-poison probes with pool-window positive controls; "
                       "pre-cutoff features and branched predictions invariant")}
    (OUT / "poison.json").write_text(json.dumps(poison, indent=1, default=ST.manifest_json_default))
    print("poison:", json.dumps({"passed": poison["passed"]}, default=str), flush=True)
    if not poison["passed"]:
        raise SystemExit("poison test failed Refusing to present predictions as causal.")

    if SMOKE:
        manifest = {"smoke": True, "direction": "D46",
                    "note": "smoke path: S3 HEAD manifest skipped, pipeline kept end-to-end",
                    "parents": ["E0102", "E0060"], "pinned_key": XO2.PINNED_KEY,
                    "scratch": str(SCRATCH)}
    else:
        manifest = ST.build_manifest({k: PERIODS[k] for k in PERIODS}, n_per=a.manifest_n,
                                     workers=8)
        manifest["created_utc"] = ST.utcnow()
        manifest["direction"] = "D46"
        manifest["parents"] = ["E0102", "E0060"]
        manifest["pinned_key"] = XO2.PINNED_KEY
        manifest["head"] = "hard event branching: unanimous joint surge else D33 level mean"
        manifest["scratch"] = str(SCRATCH)
    (OUT / "availability_manifest.json").write_text(
        json.dumps(manifest, indent=1, default=ST.manifest_json_default))
    print("manifest:", json.dumps(manifest.get("summary", {"smoke": True})), flush=True)


if __name__ == "__main__":
    main()
