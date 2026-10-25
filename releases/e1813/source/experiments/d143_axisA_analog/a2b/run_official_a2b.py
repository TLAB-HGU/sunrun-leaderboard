"""D143 A2b official-window run: fixed strong base plus residual dv analog.

Per seed: fresh CPU base mixture fits on targets < cutoff and fresh OOF
fits on targets < OOF_TRAIN_END for scenario paths and the blend weight,
causal residual replay on the 96-dim past plus error plus path scenario
(pre-cutoff pool for official origins), additive blend with the OOF-only
weight. Writes $EXPOS_OUT/predictions_official_s{seed}.parquet plus
poison.json plus meta.json. CPU only. Every array is fit inside this run.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

LB = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
sys.path.insert(0, str(Path(__file__).parent))
import info_ablation as ia  # noqa: E402 (shared recipe inputs)
import strips as ST  # noqa: E402 (shared recipe inputs)
import spec_a2b as SP  # noqa: E402 (D143 A2b own constants)
import resid_lib_a2b as RL  # noqa: E402 (D143 A2b own helpers)

OUT = Path(os.environ.get("EXPOS_OUT", "."))
SMOKE = os.environ.get("EXPOS_SMOKE") == "1"
SCRATCH = LB / SP.SCRATCH
CUTOFF = pd.Timestamp(SP.CUTOFF_ISO, tz="UTC")
OOF_TRAIN_END = pd.Timestamp(SP.OOF_TRAIN_END_ISO, tz="UTC")
OOF_START = pd.Timestamp(SP.OOF_START_ISO, tz="UTC")
OOF_END = pd.Timestamp(SP.OOF_END_ISO, tz="UTC")

CODE_FILES = ["experiments/d143_axisA_analog/a2b/spec_a2b.py",
              "experiments/d143_axisA_analog/a2b/resid_lib_a2b.py",
              "experiments/d143_axisA_analog/a2b/run_official_a2b.py",
              "experiments/headroom_audit/info_ablation.py",
              "experiments/headroom_audit/headroom.py",
              "experiments/d1_longitude_strips/strips.py"]
DATA_FILES = ["store/ch-v1/ace.parquet", "store/ch-v1/ch-hourly.parquet",
              "store/suvi-fusion/suvi-hourly-v2.parquet"]
CHUNK_ORIGINS = 1500


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _cache_key():
    payload = {"freq": SP.TRAIN_FREQ, "smoke": False, "axis": "A2b",
               "code": {f: _file_sha(LB / f) for f in CODE_FILES},
               "data": {f: [(LB / f).stat().st_size, (LB / f).stat().st_mtime_ns]
                        for f in DATA_FILES}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _cache_get(key):
    d = SCRATCH / "featcache"
    pq, js = d / (key + ".parquet"), d / (key + ".json")
    if not (pq.is_file() and js.is_file()):
        return None
    try:
        meta = json.loads(js.read_text())
        if meta.get("key") != key:
            return None
        df = pd.read_parquet(pq)
        if len(df) != meta.get("rows") or list(df.columns) != meta.get("columns"):
            return None
        return df
    except Exception:
        return None


def _cache_put(key, df):
    d = SCRATCH / "featcache"
    d.mkdir(parents=True, exist_ok=True)
    meta = {"key": key, "rows": int(len(df)), "columns": [str(c) for c in df.columns]}
    tmp = d / (key + ".tmp.parquet")
    df.to_parquet(tmp, index=False)
    (d / (key + ".json")).write_text(json.dumps(meta, sort_keys=True))
    tmp.rename(d / (key + ".parquet"))
    return df


def attach_strips_train(df, meta, block):
    """Join per-(origin,h) strip columns onto a long (origin,h) table."""
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
    return df


def build_train():
    if not SMOKE:
        key = _cache_key()
        hit = _cache_get(key)
        if hit is not None:
            print(f"train featcache hit rows={len(hit)}", flush=True)
            return hit
    df = ia.build(freq=SP.TRAIN_FREQ)
    meta, block = ST.load_inputs()
    df = attach_strips_train(df, meta, block)
    if not SMOKE:
        _cache_put(key, df)
    return df


def filled_hourly():
    ace = pd.read_parquet(LB / "store/ch-v1/ace.parquet")
    ace["timestamp_utc"] = pd.to_datetime(ace["timestamp_utc"], utc=True)
    s = ace.set_index("timestamp_utc")["filled_speed_kms"].sort_index()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
    return s.reindex(full)


def base_grid_for(origins, ace, HR, meta, block, cols):
    o = pd.DatetimeIndex(origins, tz="UTC")
    feats = (ia.speed_features(o, ace).join(HR.ace_features(o)).join(HR.ch_features(o)))
    grid = pd.DataFrame({"origin": np.repeat(o.to_numpy(), 72), "h": np.tile(ia.H, len(o))})
    grid["target_t"] = pd.to_datetime(grid["origin"], utc=True) + pd.to_timedelta(grid["h"], unit="h")
    for d in (26, 27, 28):
        grid[f"rec{d}"] = ace.reindex(pd.DatetimeIndex(grid["target_t"] - pd.Timedelta(days=d))).to_numpy()
    grid["rec_mean"] = grid[["rec26", "rec27", "rec28"]].mean(1)
    grid = grid.join(feats, on="origin")
    sdf, _ = ST.strip_features(o, meta, block)
    sdf["origin"] = pd.to_datetime(sdf["origin"], utc=True)
    grid["origin"] = pd.to_datetime(grid["origin"], utc=True)
    grid = grid.merge(sdf, on=["origin", "h"], how="left", suffixes=("", "_dup"))
    missing = [c for c in cols if c not in grid.columns]
    assert not missing, f"missing base columns {missing[:5]}"
    return grid


def fit_seed_models(depth, seed_idx, tr_s1, tr_s2, tr_long, cols):
    level = RL.level_seeds(seed_idx)
    models = {}
    for lv in level:
        p = RL.fit_params(dict(SP.BASE_PARAMS), depth, lv)
        models[lv] = (xgb.XGBRegressor(**p).fit(tr_s1[cols], tr_s1["y"]),
                      xgb.XGBRegressor(**p).fit(tr_s2[cols], tr_s2["y"]))
    p_ev = RL.fit_params(dict(SP.BASE_PARAMS), depth, RL.event_seed(seed_idx))
    return models, xgb.XGBRegressor(**p_ev).fit(tr_long[cols], tr_long["y"]), level


def predict_mixture(models, m_long, level, event, te, cols):
    h = te["h"].to_numpy()
    m1 = h <= 6
    m2 = (h >= 7) & (h <= SP.SPLIT_H)
    m3 = h > SP.SPLIT_H
    long_all = np.asarray(m_long.predict(te[cols]), dtype=np.float64)
    s1 = {s: np.asarray(models[s][0].predict(te.loc[m1, cols]), dtype=np.float64) for s in level}
    s2 = {s: np.asarray(models[s][1].predict(te.loc[m2, cols]), dtype=np.float64) for s in level}
    tail = np.asarray(m_long.predict(te.loc[m3, cols]), dtype=np.float64)
    members = RL.stitch_members(s1, s2, long_all, tail, m1, m2, m3, level, event)
    return RL.blend_per_lead(members, SP.WEIGHTS, h)


def predict_paths_mixture(models, m_long, level, event, origins, ace, HR, meta, block, cols):
    o = pd.DatetimeIndex(origins, tz="UTC")
    outs = []
    for a in range(0, len(o), CHUNK_ORIGINS):
        fr = base_grid_for(o[a:a + CHUNK_ORIGINS], ace, HR, meta, block, cols)
        outs.append(np.asarray(predict_mixture(models, m_long, level, event, fr, cols), dtype=float))
    return RL.as_path_matrix(np.concatenate(outs))


def predict_paths_flat(models, m_long, level, event, origins, ace, HR, meta, block, cols):
    o = pd.DatetimeIndex(origins, tz="UTC")
    outs = []
    for a in range(0, len(o), CHUNK_ORIGINS):
        fr = base_grid_for(o[a:a + CHUNK_ORIGINS], ace, HR, meta, block, cols)
        outs.append(np.asarray(predict_mixture(models, m_long, level, event, fr, cols), dtype=float))
    return np.concatenate(outs)


def cand_grid(all_queries, lookback_h):
    lo = (pd.DatetimeIndex(all_queries, tz="UTC").min() - pd.Timedelta(f"{int(lookback_h)}h")).ceil("h")
    hi = (pd.DatetimeIndex(all_queries, tz="UTC").max() - pd.Timedelta("655h")).floor("h")
    return pd.date_range(lo, hi, freq="h", tz="UTC")


def main():
    t0 = time.time()
    assert Path(str(SCRATCH)).name == "worker-t1a2", "scratch isolation to worker-t1a2 only"
    SCRATCH.mkdir(parents=True, exist_ok=True)
    RL.check_blend_grid(SP.BLEND_GRID)
    assert SP.K == 15 and SP.SCENARIO_H == 12 and SP.ERR_H == 12 and SP.PATH_H == 72, "preregistered A2b setting"
    assert SP.SCENARIO_DIM == 96, "12 past plus 12 error plus 72 path"
    assert SP.PATH_SOURCE == "oof-fit", "scenario paths come from the seed OOF fit only"
    assert SP.BASE_DEPTH == 3 and SP.BASE_PARAMS.get("n_estimators") == 900, "E0408 base setting"
    assert abs(float(SP.BASE_PARAMS.get("learning_rate", 0)) - 0.03) < 1e-12, "E0408 base setting"
    assert SP.RESIDUAL_DEF == "dv = y - yhat_base", "residual definition pinned"
    seeds = (SP.SMOKE_SEED,) if SMOKE else tuple(SP.SEEDS)
    lookback_h = SP.SMOKE_LOOKBACK_H if SMOKE else SP.POOL_MAX_LOOKBACK_H

    sys.path.insert(0, str(LB / "experiments/headroom_audit"))
    import headroom as HR  # noqa: E402 (shared recipe inputs)

    df = build_train()
    meta, block = ST.load_inputs()
    cols = ia.arms(df.columns)["existing_all"] + ST.strip_columns()
    assert len(cols) > 10 and len(ST.strip_columns()) == 17, "existing_all plus 17 strips present"
    RL.check_weights(SP.WEIGHTS)
    tr_full = df[df["target_t"] < CUTOFF]
    tr_oof = df[df["target_t"] < OOF_TRAIN_END]
    if SMOKE:
        tr_full_s1 = tr_full[tr_full["h"] <= 6].head(2000)
        tr_full_s2 = tr_full[(tr_full["h"] >= 7) & (tr_full["h"] <= SP.SPLIT_H)].head(2000)
        tr_full_long = tr_full.head(4000)
        tr_oof_s1 = tr_oof[tr_oof["h"] <= 6].head(2000)
        tr_oof_s2 = tr_oof[(tr_oof["h"] >= 7) & (tr_oof["h"] <= SP.SPLIT_H)].head(2000)
        tr_oof_long = tr_oof.head(4000)
    else:
        tr_full_s1 = tr_full[tr_full["h"] <= 6]
        tr_full_s2 = tr_full[(tr_full["h"] >= 7) & (tr_full["h"] <= SP.SPLIT_H)]
        tr_full_long = tr_full
        tr_oof_s1 = tr_oof[tr_oof["h"] <= 6]
        tr_oof_s2 = tr_oof[(tr_oof["h"] >= 7) & (tr_oof["h"] <= SP.SPLIT_H)]
        tr_oof_long = tr_oof
    assert min(len(tr_full_long), len(tr_oof_long)) > 0, "empty train; refusing to fit"
    print(f"train rows full={len(tr_full_long)} oof={len(tr_oof_long)} cols={len(cols)} seeds={list(seeds)}", flush=True)

    ace = filled_hourly()
    oof_origins = pd.date_range(SP.OOF_START_ISO, SP.OOF_END_ISO, freq=SP.OOF_FREQ, tz="UTC")
    origins = pd.date_range(SP.ORIGIN_START, SP.ORIGIN_END, freq="h", tz="UTC")
    if SMOKE:
        oof_origins = oof_origins[:8]
        origins = origins[:8]
    else:
        assert len(origins) == SP.N_OFFICIAL_ORIGINS, "official grid must be 2833"
    cand_times = cand_grid(list(oof_origins) + list(origins), lookback_h)
    print(f"cand grid hours={len(cand_times)} lookback_h={lookback_h}", flush=True)

    ref = df[["origin", "h", "y"]].copy()
    ref["origin"] = pd.to_datetime(ref["origin"], utc=True)
    key = pd.DataFrame({"origin": np.repeat(oof_origins.to_numpy(), 72),
                        "h": np.tile(np.arange(1, 73), len(oof_origins))})
    key["origin"] = pd.to_datetime(key["origin"], utc=True)
    y_oof = key.merge(ref, on=["origin", "h"], how="left")["y"].to_numpy(dtype=float)
    assert int(np.isfinite(y_oof).sum()) > 100, "OOF labels too few; refusing to fit weights"

    oof_hind_origins = oof_origins - pd.Timedelta("12h")
    off_hind_origins = origins - pd.Timedelta("12h")

    for s in seeds:
        mo, mm_long, level = fit_seed_models(SP.BASE_DEPTH, s, tr_oof_s1, tr_oof_s2, tr_oof_long, cols)
        ev = RL.event_seed(s)
        t1 = time.time()
        te_oof_flat = predict_paths_flat(mo, mm_long, level, ev, oof_origins, ace, HR, meta, block, cols)
        q_oof = RL.as_path_matrix(te_oof_flat)
        q_off = predict_paths_mixture(mo, mm_long, level, ev, origins, ace, HR, meta, block, cols)
        c_paths = predict_paths_mixture(mo, mm_long, level, ev, cand_times, ace, HR, meta, block, cols)
        q_hind_oof = predict_paths_mixture(mo, mm_long, level, ev, oof_hind_origins, ace, HR, meta, block, cols)
        q_hind_off = predict_paths_mixture(mo, mm_long, level, ev, off_hind_origins, ace, HR, meta, block, cols)
        print(f"seed {s}: paths oof={q_oof.shape} off={q_off.shape} cand={c_paths.shape} "
              f"seconds={round(time.time() - t1, 1)}", flush=True)
        t2 = time.time()
        dv_oof, cov_oof = RL.residual_for_origins(
            oof_origins, ace, q_oof, q_hind_oof, cand_times, c_paths, SP.K, lookback_h=lookback_h)
        dv_off, cov_off = RL.residual_for_origins(
            origins, ace, q_off, q_hind_off, cand_times, c_paths, SP.K, cutoff=CUTOFF, lookback_h=lookback_h)
        print(f"seed {s}: residual oof_cov={float(np.isfinite(dv_oof).mean()):.3f} "
              f"off_cov={float(np.isfinite(dv_off).mean()):.3f} seconds={round(time.time() - t2, 1)}",
              flush=True)
        w, oof_base_mse, oof_blend_mse, _ = RL.fit_blend_weight_dv(te_oof_flat, dv_oof, y_oof, SP.BLEND_GRID)
        mf, mf_long, levelf = fit_seed_models(SP.BASE_DEPTH, s, tr_full_s1, tr_full_s2, tr_full_long, cols)
        evf = RL.event_seed(s)
        te = base_grid_for(origins, ace, HR, meta, block, cols)
        b_full = np.asarray(predict_mixture(mf, mf_long, levelf, evf, te, cols), dtype=float)
        final = RL.blend_dv(b_full, np.asarray(dv_off, dtype=float).ravel(), w)
        assert np.isfinite(final).all() and len(final) == len(origins) * 72, "non-finite final"
        frame = RL.frame_predictions(te[["origin", "h"]], final)
        first = frame[frame["origin_last_input_utc"] == frame["origin_last_input_utc"].iloc[0]]
        assert (first["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
        assert (frame.groupby("origin_last_input_utc")["horizon_hours"].count() == 72).all()
        frame.to_parquet(OUT / f"predictions_official_s{s}.parquet", index=False)
        print(f"seed {s}: w={w:.2f} oof_base={oof_base_mse:.1f} oof_blend={oof_blend_mse:.1f} "
              f"rows={len(frame)}", flush=True)

    probe = pd.DatetimeIndex(list(oof_origins[:4]) + list(origins[:4]), tz="UTC") \
        if not SMOKE else pd.DatetimeIndex(list(oof_origins) + list(origins), tz="UTC")

    def _frame_for(filled_series):
        return base_grid_for(probe, filled_series, HR, meta, block, cols)[cols]

    ps = RL.scenario_poison(probe, ace, CUTOFF)
    pf = RL.frame_poison(probe, ace, CUTOFF, _frame_for)
    pst = ST.poison_test(probe, CUTOFF, meta=meta, block=block)
    passed = bool(ps["passed"] and pf["passed"] and pst["passed"])
    record = {"k": SP.K, "scenario_h": SP.SCENARIO_H, "err_h": SP.ERR_H, "path_h": SP.PATH_H,
              "path_source": SP.PATH_SOURCE, "residual": SP.RESIDUAL_DEF, "scenario_dim": SP.SCENARIO_DIM,
              "seeds": list(seeds), "smoke": bool(SMOKE), "blend": "oof-only",
              "passed": passed, "cutoff_utc": ps["cutoff_utc"],
              "perturbation": float(ps["perturbation"]),
              "n_pre_rows": int(ps["n_pre_rows"]), "n_post_rows": int(ps["n_post_rows"]),
              "pre_invariant": bool(ps["pre_invariant"] and pf["pre_invariant"] and pst.get("pre_invariant", True)),
              "moved": bool(ps["moved"] and pf["moved"] and len(pst.get("moved_columns", [])) > 0),
              "moved_columns": sorted(set(ps["moved_columns"]) | set(pf["moved_columns"]) | set(pst.get("moved_columns", []))),
              "note": "perturbed post-cutoff filled speeds and cache inputs; pre-cutoff past scenarios and "
                      "model-input frame blocks bitwise identical",
              "scenario": ps,
              "frame": {kk: pf[kk] for kk in ("passed", "pre_invariant", "moved", "n_pre_rows", "n_post_rows")},
              "strips": {kk: pst[kk] for kk in ("passed", "pre_invariant", "n_pre_rows", "n_post_rows") if kk in pst}}
    (OUT / "poison.json").write_text(json.dumps(record, indent=1, default=str))
    print("poison:", json.dumps({k: record[k] for k in
          ("passed", "cutoff_utc", "pre_invariant", "n_pre_rows", "n_post_rows")}, default=str), flush=True)
    if not passed:
        raise SystemExit("poison test failed; refusing to present predictions as causal")
    (OUT / "meta.json").write_text(json.dumps({
        "direction": SP.DIRECTION, "axis": "A2b", "k": SP.K, "seeds": list(seeds),
        "cutoff": SP.CUTOFF_ISO, "oof": {"train_end": SP.OOF_TRAIN_END_ISO,
        "start": SP.OOF_START_ISO, "end": SP.OOF_END_ISO, "freq": SP.OOF_FREQ},
        "path_source": SP.PATH_SOURCE, "residual": SP.RESIDUAL_DEF, "scenario_dim": SP.SCENARIO_DIM,
        "base": {"depth": SP.BASE_DEPTH, "n_estimators": SP.BASE_PARAMS["n_estimators"],
                 "lr": SP.BASE_PARAMS["learning_rate"], "weights": "pinned-devOOF"},
        "train_rows": int(len(tr_full_long)), "n_origins": int(len(origins)),
        "smoke": bool(SMOKE), "seconds": round(time.time() - t0, 1)}, indent=1, default=str))
    print(f"done k={SP.K} seconds={round(time.time() - t0, 1)}", flush=True)


if __name__ == "__main__":
    main()
