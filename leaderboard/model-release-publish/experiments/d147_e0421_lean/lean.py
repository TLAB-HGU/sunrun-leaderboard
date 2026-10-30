"""D147 E0421 lean alignment: replace the heuristic parts of E0421 one at a time and test non-inferiority.

E0421 (D143 A2b) = an E0408-structure base (per-lead mixture of e0003_line + d33_hsplit + d45_lvlmean, 7 XGBoost fits,
pinned E0145 weights, depth 3 / 900 trees / lr 0.03) plus a residual corrector: dv = y - base is replayed as the uniform
mean of the K=15 nearest past scenarios (96-dim: past 12 h speed, 12 h base error, 72 h base path; raw km/s, equal
subspace weight; Carrington exclusion; pre-cutoff pool) and added as base + w * dv, w on a 0..1 grid on one OOF window.

This runner re-uses the a2b code read-only and changes exactly the factors named by --arm (or --cfg):
  members   mixture (E0421) | d33 (single member: one short 1-6 and 7-24 head + the long head, 3 fits)
  blend     grid_main (E0421) | ols_pooled (closed-form least squares w over all OOF windows, clipped to [0, 1])
  dist      raw_equal (E0421) | std_equal (each subspace divided by its pre-2026 pool variance)
  k / kcv   15 (E0421) | kcv: K chosen from (7, 15, 30, 60) by pooled OOF MSE
  drop      () | groups v_lag, v_roll, img_meta removed from the base features
  strip     {} | strip constants (COVER_MIN, STALE_H, ...) overridden before any feature is built
  hyper     None (E0421) | nested: depth and tree count chosen by early stopping on 2025-09..12 (before every OOF window)
Every arm also scores three expanding OOF windows (W0 Jan, W1 Feb, W2 Mar-May 2026, each with its own fit on earlier
targets) and writes oof_<window>_s<seed>.parquet for the non-inferiority statistic; official predictions use the W2 fit for
paths and the full fit (targets < 2026-05-29) for the base, exactly as E0421. No official truth is read; no stored
prediction of another experiment is read. CPU only.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

LB = Path(__file__).resolve().parents[2]
A2B = LB / "experiments/d143_axisA_analog/a2b"
for p in (LB / "experiments/headroom_audit", LB / "experiments/d1_longitude_strips", A2B):
    sys.path.insert(0, str(p))
import info_ablation as ia  # noqa: E402 (read-only)
import strips as ST  # noqa: E402 (read-only; constants may be overridden in-process by the strip arm)
import spec_a2b as SP  # noqa: E402 (read-only)
import resid_lib_a2b as RL  # noqa: E402 (read-only)
import run_official_a2b as A  # noqa: E402 (read-only helpers)
import headroom as HR  # noqa: E402 (read-only)

OUT = Path(os.environ.get("EXPOS_OUT", "."))
SMOKE = os.environ.get("EXPOS_SMOKE") == "1"
SCRATCH = LB / "store/scratch/worker-d147"
CUTOFF = pd.Timestamp(SP.CUTOFF_ISO, tz="UTC")
WINDOWS = (("W0", "2026-01-01T00:00:00Z", "2026-01-04T00:00:00Z", "2026-01-31T23:00:00Z"),
           ("W1", "2026-02-01T00:00:00Z", "2026-02-04T00:00:00Z", "2026-02-28T23:00:00Z"),
           ("W2", SP.OOF_TRAIN_END_ISO, SP.OOF_START_ISO, SP.OOF_END_ISO))
SCALE_END = pd.Timestamp("2026-01-01T00:00:00Z")  # pool statistics only from data before every OOF window
K_GRID = (7, 15, 30, 60)
HYPER_TRAIN_END = pd.Timestamp("2025-09-01T00:00:00Z")
HYPER_VAL = ("2025-09-04T00:00:00Z", "2025-12-28T23:00:00Z")
DEFAULT = {"members": "mixture", "blend": "grid_main", "dist": "raw_equal", "k": 15, "kcv": False,
           "drop": [], "strip": {}, "hyper": None}
DROP = {"v_lag": lambda c: c.startswith("v_lag"),
        "v_roll": lambda c: c.startswith(("v_m", "v_sd")),
        "img_meta": lambda c: c in ("img_age_h", "img_stale")}
ARMS = {
    "control": {},
    "s1_d33": {"members": "d33"},
    "s2_olspool": {"blend": "ols_pooled"},
    "s5_drop": {"drop": ["v_lag", "v_roll", "img_meta"]},
    "s3_std": {"dist": "std_equal"},
    "s4_kcv": {"kcv": True},
    "s6_nothresh": {"strip": {"COVER_MIN": 0.0, "STALE_H": 1.0e9}},
    "s7_nested": {"hyper": "nested"},
}


# ---- features -------------------------------------------------------------------------------------------------------
def build_train(strip_cfg):
    """E0421's cached train frame when the strip constants are untouched, else a fresh build under worker-d147."""
    if not strip_cfg:
        hit = A._cache_get(A._cache_key())  # read-only use of worker-t1a2's feature cache (same frame as E0421)
        if hit is not None:
            print(f"train featcache (E0421) hit rows={len(hit)}", flush=True)
            return hit
    h = hashlib.sha256()
    for f in ("experiments/d147_e0421_lean/lean.py", "experiments/d1_longitude_strips/strips.py",
              "experiments/headroom_audit/info_ablation.py", "experiments/headroom_audit/headroom.py"):
        h.update((LB / f).read_bytes())
    h.update(json.dumps({"strip": strip_cfg, "freq": SP.TRAIN_FREQ, "smoke": SMOKE}, sort_keys=True).encode())
    pq = SCRATCH / "featcache" / (h.hexdigest() + ".parquet")
    if pq.is_file():
        return pd.read_parquet(pq)
    df = ia.build(freq=SP.TRAIN_FREQ)
    meta, block = ST.load_inputs()
    df = A.attach_strips_train(df, meta, block)
    pq.parent.mkdir(parents=True, exist_ok=True)
    tmp = pq.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(pq)
    return df


def feature_columns(df, cfg):
    cols = ia.arms(df.columns)["existing_all"] + ST.strip_columns()
    for g in cfg["drop"]:
        keep = [c for c in cols if not DROP[g](c)]
        if len(keep) == len(cols):
            raise SystemExit(f"drop group {g} removed nothing")
        cols = keep
    return cols


# ---- base model -----------------------------------------------------------------------------------------------------
def select_hyper(df, cols):
    """Nested choice on data before every OOF window: depth by validation MSE, trees by early stopping (long head proxy)."""
    tr = df[df["target_t"] < HYPER_TRAIN_END]
    o = pd.to_datetime(df["origin"], utc=True)
    va = df[(o >= pd.Timestamp(HYPER_VAL[0])) & (o <= pd.Timestamp(HYPER_VAL[1])) & np.isfinite(df["y"])]
    if SMOKE:
        tr, va = tr.head(4000), va.head(2000)
    rows = []
    for d in (3, 4, 5, 6):
        p = {**SP.BASE_PARAMS, "max_depth": d, "n_estimators": 60 if SMOKE else 3000, "random_state": 0,
             "early_stopping_rounds": 10 if SMOKE else 100, "eval_metric": "rmse"}
        m = xgb.XGBRegressor(**p).fit(tr[cols], tr["y"], eval_set=[(va[cols], va["y"])], verbose=False)
        rows.append({"depth": d, "n_estimators": int(m.best_iteration) + 1, "val_rmse": float(m.best_score)})
        print(f"hyper depth={d} best_iter={m.best_iteration} val_rmse={m.best_score:.3f}", flush=True)
    best = min(rows, key=lambda r: r["val_rmse"])
    return best["depth"], best["n_estimators"], rows


def fit_set(cfg, depth, params, s, tr, cols):
    level = RL.level_seeds(s)
    ev = RL.event_seed(s)
    tr1 = tr[tr["h"] <= 6]
    tr2 = tr[(tr["h"] >= 7) & (tr["h"] <= SP.SPLIT_H)]
    trl = tr
    if SMOKE:
        tr1, tr2, trl = tr1.head(2000), tr2.head(2000), tr.head(4000)
    shorts = level if cfg["members"] == "mixture" else (ev,)
    models = {}
    for lv in shorts:
        p = RL.fit_params(dict(params), depth, lv)
        models[lv] = (xgb.XGBRegressor(**p).fit(tr1[cols], tr1["y"]),
                      xgb.XGBRegressor(**p).fit(tr2[cols], tr2["y"]))
    p_ev = RL.fit_params(dict(params), depth, ev)
    return models, xgb.XGBRegressor(**p_ev).fit(trl[cols], trl["y"]), level, ev


def predict(cfg, fs, te, cols):
    models, m_long, level, ev = fs
    if cfg["members"] == "mixture":
        return np.asarray(A.predict_mixture(models, m_long, level, ev, te, cols), dtype=float)
    h = te["h"].to_numpy()
    m1, m2, m3 = h <= 6, (h >= 7) & (h <= SP.SPLIT_H), h > SP.SPLIT_H
    out = np.empty(len(te), dtype=float)
    out[m1] = models[ev][0].predict(te.loc[m1, cols])
    out[m2] = models[ev][1].predict(te.loc[m2, cols])
    out[m3] = m_long.predict(te.loc[m3, cols])
    return out


def paths(cfg, fs, origins, ace, meta, block, cols):
    o = pd.DatetimeIndex(origins, tz="UTC")
    outs = []
    for a in range(0, len(o), A.CHUNK_ORIGINS):
        fr = A.base_grid_for(o[a:a + A.CHUNK_ORIGINS], ace, HR, meta, block, cols)
        outs.append(predict(cfg, fs, fr, cols))
    return RL.as_path_matrix(np.concatenate(outs))


# ---- residual corrector ---------------------------------------------------------------------------------------------
def pool_scales(filled, cand_times, cand_paths):
    """Variances of the three scenario subspaces over candidate rows before SCALE_END (pre-window data only)."""
    ct = pd.DatetimeIndex(cand_times, tz="UTC")
    past = RL.scenario_past_matrix(ct, filled)
    cp = np.asarray(cand_paths, dtype=float)
    err = np.full_like(past, np.nan)
    err[12:] = past[12:] - cp[:-12, 0:12]  # cand grid is hourly and contiguous: row i-12 is the hindcast 12 h earlier
    early = np.asarray(ct < SCALE_END)
    v = []
    for m in (past, err, cp):
        x = m[early]
        x = x[np.isfinite(x).all(axis=1)]
        v.append(float(np.var(x)) if len(x) > 100 else float(np.nanvar(m)))
    return tuple(max(x, 1e-6) for x in v)


def residual_multi(origins, filled, q_paths, q_hind, cand_times, cand_paths, ks, dist, scales,
                   cutoff=None, lookback_h=8760):
    """RL.residual_for_origins generalised to several K and a scaled distance; identical to it for raw_equal + one K."""
    origins = pd.DatetimeIndex(origins, tz="UTC")
    cand_times = pd.DatetimeIndex(cand_times, tz="UTC")
    hours = filled.index.sort_values()
    fvals = filled.reindex(hours).to_numpy(dtype=float)
    past_q = RL.scenario_past_matrix(origins, filled)
    qh = np.asarray(q_hind, dtype=float)
    qp = np.asarray(q_paths, dtype=float)
    cp = np.asarray(cand_paths, dtype=float)
    err_q_all = RL.scenario_error_matrix(past_q, qh[:, 0:12])
    kmax = max(ks)
    out = {k: np.full((len(origins), 72), np.nan) for k in ks}
    cut = pd.to_datetime(cutoff, utc=True) if cutoff is not None else None
    sp, se, sh = scales if dist == "std_equal" else (1.0, 1.0, 1.0)
    for i, o in enumerate(origins):
        q0, qe, q1 = past_q[i], err_q_all[i], qp[i]
        if not (np.isfinite(q0).all() and np.isfinite(qe).all() and np.isfinite(q1).all()):
            continue
        cands = RL.candidate_pool(o, hours, len(hours), lookback_h=lookback_h)
        if len(cands) == 0:
            continue
        cand_h = hours[cands]
        if cut is not None:
            cands = cands[np.asarray((cand_h + pd.Timedelta("72h")) < cut)]
            cand_h = hours[cands]
        if len(cands) == 0:
            continue
        loc = cand_times.searchsorted(cand_h)
        valid = (loc < len(cand_times)) & (np.asarray(cand_times[np.minimum(loc, len(cand_times) - 1)]) == np.asarray(cand_h))
        cands, loc = cands[valid], loc[valid]
        if len(cands) == 0:
            continue
        pc = np.full((len(cands), 12), np.nan)
        for j, p in enumerate(cands):
            if p - 11 >= 0:
                pc[j] = fvals[p - 11: p + 1]
        hc = np.full((len(cands), 12), np.nan)
        ok_h = loc >= 12
        hc[ok_h] = cp[loc[ok_h] - 12, 0:12]
        ec = pc - hc
        ok = np.isfinite(pc).all(axis=1) & np.isfinite(ec).all(axis=1) & np.isfinite(cp[loc]).all(axis=1)
        cands_f, loc_f = cands[ok], loc[ok]
        if len(cands_f) == 0:
            continue
        dp = np.mean((pc[ok] - q0) ** 2, axis=1) / sp
        de = np.mean((ec[ok] - qe) ** 2, axis=1) / se
        dh = np.mean((cp[loc_f] - q1) ** 2, axis=1) / sh
        order = np.argsort(dp + de + dh, kind="stable")[:kmax]
        curves = []
        for p, li in zip(cands_f[order], loc_f[order]):
            seg = fvals[p + 1: p + 73]
            if len(seg) < 72:
                seg = np.pad(seg, (0, 72 - len(seg)), constant_values=np.nan)
            curves.append(seg - cp[int(li)])
        curves = np.asarray(curves, dtype=float)
        for k in ks:
            with np.errstate(all="ignore"):
                out[k][i] = np.nanmean(curves[:k], axis=0)
    return out


def blend_weight(cfg, win):
    """grid_main: E0421's grid argmin on W2. ols_pooled: closed-form least squares over all windows, clipped to [0, 1]."""
    if cfg["blend"] == "grid_main":
        w2 = win["W2"]
        return float(RL.fit_blend_weight_dv(w2["base"], w2["dv"], w2["y"], SP.BLEND_GRID)[0])
    num = den = 0.0
    for r in win.values():
        a, b, t = (np.asarray(r[x], dtype=float).ravel() for x in ("dv", "base", "y"))
        m = np.isfinite(a) & np.isfinite(b) & np.isfinite(t)
        num += float(np.sum(a[m] * (t[m] - b[m])))
        den += float(np.sum(a[m] ** 2))
    return float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0


def oof_mse(win, w):
    se, n = 0.0, 0
    for r in win.values():
        b, a, t = (np.asarray(r[x], dtype=float).ravel() for x in ("base", "dv", "y"))
        f = np.where(np.isfinite(a), b + w * a, b)
        m = np.isfinite(f) & np.isfinite(t)
        se += float(np.sum((f[m] - t[m]) ** 2))
        n += int(m.sum())
    return se / max(n, 1)


# ---- main -----------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS) + ["custom"])
    ap.add_argument("--cfg", default="{}", help="JSON overrides for --arm custom")
    a = ap.parse_args()
    cfg = {**DEFAULT, **(json.loads(a.cfg) if a.arm == "custom" else ARMS[a.arm])}
    for k, v in cfg["strip"].items():
        setattr(ST, k, v)
    t0 = time.time()
    seeds = (SP.SMOKE_SEED,) if SMOKE else tuple(SP.SEEDS)
    lookback_h = SP.SMOKE_LOOKBACK_H if SMOKE else SP.POOL_MAX_LOOKBACK_H
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    print(f"d147 arm={a.arm} cfg={json.dumps(cfg, sort_keys=True)} smoke={SMOKE} seeds={list(seeds)}", flush=True)

    df = build_train(cfg["strip"])
    meta, block = ST.load_inputs()
    cols = feature_columns(df, cfg)
    params, depth, hyper_rows = dict(SP.BASE_PARAMS), SP.BASE_DEPTH, None
    if cfg["hyper"] == "nested":
        depth, n_est, hyper_rows = select_hyper(df, cols)
        params["n_estimators"] = n_est
    print(f"cols={len(cols)} depth={depth} n_estimators={params['n_estimators']}", flush=True)
    ace = A.filled_hourly()
    origins = pd.date_range(SP.ORIGIN_START, SP.ORIGIN_END, freq="h", tz="UTC")
    wins = []
    for name, tr_end, o0, o1 in WINDOWS:
        oo = pd.date_range(o0, o1, freq=SP.OOF_FREQ, tz="UTC")
        wins.append((name, pd.Timestamp(tr_end), oo[:8] if SMOKE else oo))
    if SMOKE:
        origins = origins[:8]
    else:
        assert len(origins) == SP.N_OFFICIAL_ORIGINS
    cand_times = A.cand_grid([o for _, _, oo in wins for o in oo] + list(origins), lookback_h)
    ref = df[["origin", "h", "y"]].copy()
    ref["origin"] = pd.to_datetime(ref["origin"], utc=True)
    ks = K_GRID if cfg["kcv"] else (int(cfg["k"]),)
    tr_full = df[df["target_t"] < CUTOFF]
    record = {"arm": a.arm, "cfg": cfg, "cols": len(cols), "depth": depth, "n_estimators": params["n_estimators"],
              "hyper": hyper_rows, "seeds": {}}

    for s in seeds:
        win, fs_w2, cand_w2, scales_w2 = {}, None, None, None
        for name, tr_end, oo in wins:
            fs = fit_set(cfg, depth, params, s, df[df["target_t"] < tr_end], cols)
            base = paths(cfg, fs, oo, ace, meta, block, cols)
            hind = paths(cfg, fs, oo - pd.Timedelta("12h"), ace, meta, block, cols)
            cand = paths(cfg, fs, cand_times, ace, meta, block, cols)
            scales = pool_scales(ace, cand_times, cand)
            if cfg["dist"] == "raw_equal" and not cfg["kcv"]:
                dv, _ = RL.residual_for_origins(oo, ace, base, hind, cand_times, cand, int(cfg["k"]), lookback_h=lookback_h)
                dvs = {int(cfg["k"]): dv}
            else:
                dvs = residual_multi(oo, ace, base, hind, cand_times, cand, ks, cfg["dist"], scales, lookback_h=lookback_h)
            key = pd.DataFrame({"origin": np.repeat(oo.to_numpy(), 72), "h": np.tile(np.arange(1, 73), len(oo))})
            key["origin"] = pd.to_datetime(key["origin"], utc=True)
            y = key.merge(ref, on=["origin", "h"], how="left")["y"].to_numpy(dtype=float)
            win[name] = {"base": base.ravel(), "dvs": {k: v.ravel() for k, v in dvs.items()}, "y": y, "key": key}
            if name == "W2":
                fs_w2, cand_w2, scales_w2 = fs, cand, scales
            print(f"seed {s} {name}: oof origins={len(oo)} train_end={tr_end.date()} "
                  f"cov={float(np.isfinite(dvs[ks[0]]).mean()):.3f}", flush=True)
        choice = {}
        for k in ks:
            view = {n: {"base": r["base"], "dv": r["dvs"][k], "y": r["y"]} for n, r in win.items()}
            w = blend_weight(cfg, view)
            choice[k] = (oof_mse(view, w), w)
        k_best = min(choice, key=lambda k: choice[k][0])
        mse_best, w = choice[k_best]
        q_off = paths(cfg, fs_w2, origins, ace, meta, block, cols)
        q_hind_off = paths(cfg, fs_w2, origins - pd.Timedelta("12h"), ace, meta, block, cols)
        if cfg["dist"] == "raw_equal" and not cfg["kcv"]:
            dv_off, _ = RL.residual_for_origins(origins, ace, q_off, q_hind_off, cand_times, cand_w2, int(cfg["k"]),
                                                cutoff=CUTOFF, lookback_h=lookback_h)
        else:
            dv_off = residual_multi(origins, ace, q_off, q_hind_off, cand_times, cand_w2, (k_best,), cfg["dist"],
                                    scales_w2, cutoff=CUTOFF, lookback_h=lookback_h)[k_best]
        fs_full = fit_set(cfg, depth, params, s, tr_full, cols)
        te = A.base_grid_for(origins, ace, HR, meta, block, cols)
        final = RL.blend_dv(predict(cfg, fs_full, te, cols), np.asarray(dv_off, dtype=float).ravel(), w)
        assert np.isfinite(final).all() and len(final) == len(origins) * 72, "non-finite final"
        RL.frame_predictions(te[["origin", "h"]], final).to_parquet(OUT / f"predictions_official_s{s}.parquet", index=False)
        for name, r in win.items():
            dv = r["dvs"][k_best]
            pred = np.where(np.isfinite(dv), r["base"] + w * dv, r["base"])
            o = r["key"].assign(y=r["y"], base=r["base"], dv=dv, pred=pred)
            o.to_parquet(OUT / f"oof_{name}_s{s}.parquet", index=False)
        per_win = {n: oof_mse({n: {"base": r["base"], "dv": r["dvs"][k_best], "y": r["y"]}}, w) for n, r in win.items()}
        record["seeds"][str(s)] = {"k": int(k_best), "w": w, "oof_mse_pooled": mse_best, "oof_mse_by_window": per_win,
                                   "k_scores": {str(k): v[0] for k, v in choice.items()}, "scales": scales_w2}
        print(f"seed {s}: k={k_best} w={w:.3f} oof_pooled={mse_best:.1f} by_window={ {n: round(v, 1) for n, v in per_win.items()} }",
              flush=True)

    probe = pd.DatetimeIndex(list(wins[-1][2][:4]) + list(origins[:4]), tz="UTC")
    # The frame poison perturbs the filled speed series; when the drop arm removes every column built from it, the
    # positive control has nothing to move, so the check runs on a superset (model columns + v_lag0).
    pcols = cols if any(c.startswith("v_lag") for c in cols) else cols + ["v_lag0"]

    def _frame_for(filled_series):
        return A.base_grid_for(probe, filled_series, HR, meta, block, pcols)[pcols]

    ps = RL.scenario_poison(probe, ace, CUTOFF)
    pf = RL.frame_poison(probe, ace, CUTOFF, _frame_for)
    pst = ST.poison_test(probe, CUTOFF, meta=meta, block=block)
    passed = bool(ps["passed"] and pf["passed"] and pst["passed"])
    (OUT / "poison.json").write_text(json.dumps({"passed": passed, "scenario": ps, "frame": {k: pf[k] for k in (
        "passed", "pre_invariant", "moved", "n_pre_rows", "n_post_rows")}, "strips": {k: pst[k] for k in (
        "passed", "pre_invariant", "n_pre_rows", "n_post_rows") if k in pst}}, indent=1, default=str))
    print("poison:", passed, flush=True)
    if not passed:
        raise SystemExit("poison test failed; refusing to present predictions as causal")
    record["seconds"] = round(time.time() - t0, 1)
    (OUT / "selection.json").write_text(json.dumps(record, indent=1, default=str))
    print(f"done arm={a.arm} seconds={record['seconds']}", flush=True)


if __name__ == "__main__":
    main()
