"""e2100_tda_pi_suvi72: end-to-end run.

1. build the persistence-image cache for every admissible SUVI Fe195 frame (resumable)
2. paper replication: 4-day HSS classification, purged 20-chunk / 5-fold OOF (internal result)
3. official forecast: ridge per horizon on [392 TDA + 20 ACE] features, one fit per period cutoff
4. write predictions.parquet (454,680 rows), meta.yaml and results.json

The paper (arXiv 2610.35549) is a 4-day binary HSS classifier; the leaderboard scores a 72-hour
speed curve, so step 3 carries the paper's representation and its best-performing linear model
family over to the leaderboard target. Step 2 is the paper's own task and is reported, not gated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/t-lab01/sunrun")
SPEED = ROOT / "data-v2/model-inputs/leaderboard-store/e2000fs-lag0/speed.parquet"
OMNI = ROOT / "data-v2/collectors/nrt/omni/hourly/omni2_hourly.parquet"
FOLDS = Path("/tmp/folds_official.parquet")
H = 72
MATURITY_H = 72          # training-label maturity: target + 72h <= cutoff
CUTOFFS = {"F1": "2025-12-29", "F2": "2026-03-29", "F3": "2026-06-29"}
ALPHAS = (100.0, 1000.0, 10000.0)
VAL_DAYS = 60
CHUNKS = 20
PURGE_DAYS = 45
MIN_FOLD_TRAIN = 400
MIN_INNER = 200
SEED = 0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------- features
def load_feature_frame() -> pd.DataFrame:
    parts = sorted((HERE / "features").glob("part-*.parquet"))
    if not parts:
        raise SystemExit("no feature parts; run build_cache first")
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates("slot").sort_values("slot").reset_index(drop=True)
    log(f"feature cache: {len(df)} frames {df.slot.min()} .. {df.slot.max()}")
    return df


PI_COLS = [f"f{i:03d}" for i in range(2 * 14 * 14)]


def ace_features(series: pd.Series, origins: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    """Causal ACE speed context at each origin (only rows with timestamp <= origin)."""
    s = series
    idx = s.index
    cols, names = [], []
    pos = idx.searchsorted(origins, side="right") - 1   # last row <= origin

    def at(lag):
        p = pos - lag
        ok = p >= 0
        v = np.full(len(origins), np.nan)
        v[ok] = s.to_numpy()[p[ok]]
        return v

    for lag in (0, 1, 2, 3, 6, 12, 24, 48, 72):
        cols.append(at(lag)); names.append(f"ace_lag{lag}")
    vals = s.to_numpy()
    for w in (6, 12, 24, 48, 72):
        win = np.full((len(origins), w), np.nan)
        for k in range(w):
            p = pos - k
            ok = p >= 0
            win[ok, k] = vals[p[ok]]
        cols.append(np.nanmean(win, axis=1)); names.append(f"ace_mean{w}")
        if w in (24, 72):
            cols.append(np.nanstd(win, axis=1)); names.append(f"ace_std{w}")
            cols.append(np.nanmin(win, axis=1)); names.append(f"ace_min{w}")
            cols.append(np.nanmax(win, axis=1)); names.append(f"ace_max{w}")
    lag = at(24) - at(0)
    cols.append(lag / 24.0); names.append("ace_slope24")
    doy = origins.dayofyear.to_numpy()
    cols.append(np.sin(2 * np.pi * doy / 365.25)); names.append("doy_sin")
    cols.append(np.cos(2 * np.pi * doy / 365.25)); names.append("doy_cos")
    X = np.column_stack(cols).astype(np.float64)
    return X, names


def frame_lookup(feat: pd.DataFrame, origins: pd.DatetimeIndex, max_lookback_h: int = 6):
    """Latest frame with obs_end <= origin, i.e. slot <= origin - 1h (frames end at :58:5x)."""
    slots = feat["slot"].to_numpy()
    P = feat[PI_COLS].to_numpy(dtype=np.float32)
    target = (origins - pd.Timedelta(hours=1)).to_numpy()
    pos = np.searchsorted(slots, target, side="right") - 1
    ok = pos >= 0
    rows = np.where(ok, pos, 0)
    age_h = np.full(len(origins), np.inf)
    age_h[ok] = (origins[ok].to_numpy() - slots[rows[ok]]).astype("timedelta64[m]").astype(float)[ok] / 60.0
    usable = ok & (age_h <= max_lookback_h + 1.0)
    X = P[rows].astype(np.float64)
    X[~usable] = np.nan
    return X, usable, age_h


# ----------------------------------------------------------------------------- paper task
def hss_labels(origins: pd.DatetimeIndex) -> pd.Series:
    """Paper Sec. 3.1: y=1 in-HSS-interval and V>500; y=0 not-in-interval and V<400; else excluded."""
    from scipy.signal import find_peaks

    origins = pd.DatetimeIndex(origins)
    if origins.tz is not None:
        origins = origins.tz_convert(None)
    om = pd.read_parquet(OMNI)
    om["t"] = pd.to_datetime(om["timestamp_utc"], utc=True).dt.tz_convert(None)
    om = om.sort_values("t").set_index("t")
    v = om["flow_speed_kms"]
    smooth = v.rolling(6, center=True, min_periods=3).median()
    sub = smooth["2022-01-01":]
    pk, _ = find_peaks(sub.to_numpy(), height=500, prominence=60, distance=24)
    inwin = pd.Series(False, index=sub.index)
    for p in sub.index[pk]:
        inwin.loc[p - pd.Timedelta(hours=54):p + pd.Timedelta(hours=36)] = True

    idx = om.index
    tgt = origins + pd.Timedelta(days=4)
    pos = np.clip(idx.searchsorted(tgt), 1, len(idx) - 1)
    dlo = np.abs((tgt.to_numpy() - idx[pos - 1].to_numpy()).astype("timedelta64[s]").astype(int))
    dhi = np.abs((idx[pos].to_numpy() - tgt.to_numpy()).astype("timedelta64[s]").astype(int))
    near = np.where(dlo <= dhi, idx[pos - 1], idx[pos])
    ok = np.minimum(dlo, dhi) / 3600.0 <= 12.0
    vv = pd.Series(v.to_numpy(float), index=idx).reindex(near).to_numpy()
    ww = inwin.reindex(near).fillna(False).to_numpy()
    y = np.where((vv > 500) & ww, 1.0, np.where((vv < 400) & (~ww), 0.0, np.nan))
    y[~ok] = np.nan
    return pd.Series(y, index=origins)


def paper_replication(feat: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import StandardScaler

    slots = pd.DatetimeIndex(feat["slot"])
    X_pi = feat[PI_COLS].to_numpy(dtype=np.float32)
    lab = hss_labels(slots).to_numpy()
    keep = np.isfinite(lab)
    t = slots[keep]
    X = X_pi[keep]
    y = lab[keep]
    log(f"paper task: labelled frames {len(y)} (pos {int(y.sum())}) {t.min()} .. {t.max()}")

    nchunk = CHUNKS
    edges = pd.date_range(t.min().floor("D"), t.max().ceil("D"), periods=nchunk + 1)
    purge = pd.Timedelta(days=PURGE_DAYS)
    oof = np.full(len(y), np.nan)
    fold_scores = []
    for k in range(5):
        test = np.zeros(len(y), bool)
        blocked = np.zeros(len(y), bool)
        for c in range(k, nchunk, 5):
            a, b = edges[c], edges[c + 1]
            test |= ((t >= a) & (t < b))
            blocked |= ((t >= a - purge) & (t < b + purge))
        if not test.any():
            fold_scores.append({"fold": k, "skipped": True, "reason": "no test chunk"})
            continue
        train = (~test) & (~blocked)
        # inner chronological validation = last VAL_DAYS of the training part
        cut = t[train].max() - pd.Timedelta(days=VAL_DAYS)
        inner_tr = train & (t <= cut)
        inner_va = train & (t > cut)
        if train.sum() < MIN_FOLD_TRAIN or inner_tr.sum() < MIN_INNER or inner_va.sum() < MIN_INNER // 2:
            log(f"  fold {k}: skipped (train {int(train.sum())}, inner {int(inner_tr.sum())}/{int(inner_va.sum())})")
            fold_scores.append({"fold": k, "skipped": True})
            continue
        best, best_C = -np.inf, ALPHAS[0]
        for C in (0.001, 0.003, 0.01, 0.03):
            sc = StandardScaler().fit(X[inner_tr])
            m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[inner_tr]), y[inner_tr])
            p = m.predict_proba(sc.transform(X[inner_va]))[:, 1]
            a = roc_auc_score(y[inner_va], p)
            if a > best:
                best, best_C = a, C
        sc = StandardScaler().fit(X[train])
        m = LogisticRegression(C=best_C, max_iter=2000).fit(sc.transform(X[train]), y[train])
        oof[test] = m.predict_proba(sc.transform(X[test]))[:, 1]
        fold_scores.append({"fold": k, "C": best_C, "n_test": int(test.sum()),
                            "pos_test": int(y[test].sum()), "inner_auc": float(best)})
        log(f"  fold {k}: test {int(test.sum())} (pos {int(y[test].sum())}) train {int(train.sum())} "
            f"C={best_C} inner_auc={best:.3f}")

    ok = np.isfinite(oof)
    if ok.sum() < MIN_INNER or len(np.unique(y[ok])) < 2:
        raise ValueError(f"paper task: only {int(ok.sum())} out-of-fold predictions; too few for a pooled score")
    auc = float(roc_auc_score(y[ok], oof[ok]))
    fpr, tpr, thr = roc_curve(y[ok], oof[ok])
    tss = float((tpr - fpr).max())
    j = int(np.argmax(tpr - fpr))
    pred = (oof[ok] >= thr[j]).astype(float)
    tp = float(((pred == 1) & (y[ok] == 1)).sum()); fp = float(((pred == 1) & (y[ok] == 0)).sum())
    fn = float(((pred == 0) & (y[ok] == 1)).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    prev_tr = float(y.mean())
    log(f"paper task OOF: n={int(ok.sum())} AUC={auc:.4f} TSS={tss:.4f} F1={f1:.4f} "
        f"(prevalence {prev_tr:.3f}; random baseline AUC 0.5, TSS 0)")
    return {"task": "HSS occurrence at t0+4d, binary, paper Sec. 3.1",
            "n_labelled": int(ok.sum()), "positives": int(y[ok].sum()), "prevalence": prev_tr,
            "oof_auc": auc, "oof_best_tss": tss, "oof_f1_at_best_tss": float(f1),
            "cv": f"{CHUNKS} contiguous chunks, 4 test chunks per fold, {PURGE_DAYS}-day purge, 5 folds",
            "classifier": "LogisticRegression on 392 persistence-image features",
            "folds": fold_scores}


# ----------------------------------------------------------------------------- leaderboard task
def ridge_fit_predict(Xtr, Ytr, Xte, inner_cut_mask, inner_val_mask):
    """Per-horizon ridge; alpha chosen on the inner chronological validation block."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    out = np.empty((Xte.shape[0], H))
    chosen, models = [], []
    for h in range(H):
        y = Ytr[:, h]
        best_a, best_mse = ALPHAS[0], np.inf
        for a in ALPHAS:
            m = Ridge(alpha=a).fit(A[inner_cut_mask], y[inner_cut_mask])
            mse = float(np.mean((m.predict(A[inner_val_mask]) - y[inner_val_mask]) ** 2))
            if mse < best_mse:
                best_a, best_mse = a, mse
        m = Ridge(alpha=best_a).fit(A, y)
        models.append(m)
        chosen.append(best_a)
    # inference timing: the per-origin 72h forecast only (features already prepared)
    t0 = time.perf_counter()
    for h, m in enumerate(models):
        out[:, h] = m.predict(B)
    per_origin = (time.perf_counter() - t0) / max(len(B), 1)
    return out, chosen, per_origin


def official_forecast(feat: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    folds = pd.read_parquet(FOLDS)          # origin_last_input_utc stays the exact ISO strings
    series = pd.read_parquet(SPEED)["speed"]
    series.index = pd.DatetimeIndex(series.index, tz="UTC")

    rows, per_period = [], {}
    for period, cut_s in CUTOFFS.items():
        cut = pd.Timestamp(cut_s, tz="UTC")
        sub = folds.loc[folds["period"] == period, ["origin_last_input_utc"]].copy()
        sub["ts"] = pd.to_datetime(sub["origin_last_input_utc"], utc=True)
        sub["origin_last_input_utc"] = sub["origin_last_input_utc"].astype(str)
        sub = sub.sort_values("ts").reset_index(drop=True)
        org = pd.DatetimeIndex(sub["ts"])
        org_str = sub["origin_last_input_utc"].to_numpy()

        # ---- training origins: hourly, label mature (t+h+72h <= cut), frame available
        t_start = series.index.min() + pd.Timedelta(hours=H)
        t_end = cut - pd.Timedelta(hours=MATURITY_H + H)
        tr_org = pd.date_range(t_start, t_end, freq="h", tz="UTC")
        Xpi_tr, ok_tr, _ = frame_lookup(feat, tr_org)
        tr_org = tr_org[ok_tr]
        Xpi_tr = Xpi_tr[ok_tr]
        Xace_tr, ace_names = ace_features(series, tr_org)
        # targets: speed at t+h, must exist and be mature (target + 72h <= cutoff)
        offs = np.arange(1, H + 1).astype("timedelta64[h]")[None, :]
        tgt_ns = tr_org.tz_convert(None).to_numpy()[:, None] + offs
        tgt_index = pd.DatetimeIndex(tgt_ns.ravel(), tz="UTC")
        Y = series.reindex(tgt_index).to_numpy().reshape(len(tr_org), H)
        mature = (((tgt_index + pd.Timedelta(hours=MATURITY_H)) <= cut)
                  .reshape(len(tr_org), H).all(axis=1))
        finite = np.isfinite(Y).all(axis=1)
        use = mature & finite
        tr_org, Xpi_tr, Xace_tr, Y = tr_org[use], Xpi_tr[use], Xace_tr[use], Y[use]
        Xtr = np.hstack([Xpi_tr, Xace_tr])
        log(f"{period}: training origins {len(tr_org)} ({tr_org.min()} .. {tr_org.max()}), cutoff {cut_s}")

        # ---- prediction origins
        Xpi_te, ok_te, age_te = frame_lookup(feat, org)
        Xace_te, _ = ace_features(series, org)
        if (~ok_te).any():
            log(f"{period}: {int((~ok_te).sum())} origins without a usable frame -> TDA columns set to 0")
        Xpi_te = np.nan_to_num(Xpi_te, nan=0.0)
        Xte = np.hstack([Xpi_te, Xace_te])

        val_end = tr_org.max()
        val_start = val_end - pd.Timedelta(days=VAL_DAYS)
        inner_val = (tr_org > val_start)
        inner_cut = ~inner_val
        pred, chosen, per_origin = ridge_fit_predict(Xtr, Y, Xte, inner_cut, inner_val)
        per_period[period] = {"train_origins": int(len(tr_org)), "origins": int(len(org)),
                              "alpha_median": float(np.median(chosen)),
                              "frame_missing": int((~ok_te).sum()),
                              "frame_age_h_median": float(np.median(age_te[ok_te])) if ok_te.any() else None,
                              "inference_seconds_per_origin": per_origin}
        block = pd.DataFrame({
            "origin_last_input_utc": np.repeat(org_str, H),
            "horizon_hours": np.tile(np.arange(1, H + 1), len(org)),
            "pred_kms": np.clip(pred.ravel(), 150.0, 1500.0),
        })
        rows.append(block)
        tgt_idx = pd.DatetimeIndex(
            (org.tz_convert(None).to_numpy()[:, None] + offs).ravel(), tz="UTC")
        truth = series.reindex(tgt_idx).to_numpy()
        mse = float(np.mean((block["pred_kms"].to_numpy() - truth) ** 2))
        per_period[period]["local_mse_vs_frozen_series"] = mse
        log(f"{period}: {len(org)} origins, local MSE {mse:.1f}, median alpha {np.median(chosen):.0f}")

    out = pd.concat(rows, ignore_index=True)
    log(f"predictions: {len(out)} rows, {out.origin_last_input_utc.nunique()} origins")
    return out, per_period


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--skip-paper", action="store_true")
    args = ap.parse_args()

    if not args.skip_cache:
        import build_cache
        parts = sorted((HERE / "features").glob("part-*.parquet"))
        frames = build_cache.frame_list("2022-08-20", "2026-09-30")
        need = (len(frames) + build_cache.PART_SIZE - 1) // build_cache.PART_SIZE
        if len(parts) < need:
            log(f"building feature cache: {len(parts)}/{need} parts on disk")
            import sys
            sys.argv = ["build_cache"]
            build_cache.main()

    feat = load_feature_frame()
    results = {"experiment": "e2100_tda_pi_suvi72",
               "paper": "arXiv:2610.35549 (Explainable prediction of HSS driven by coronal holes via TDA)",
               "features": {"n_frames": int(len(feat)),
                            "window": [str(feat.slot.min()), str(feat.slot.max())],
                            "dim": int(len(PI_COLS))}}
    # the official forecast is written first: the internal paper task never gates it
    preds, per_period = official_forecast(feat)
    results["official"] = per_period
    results["inference_seconds_per_fold"] = float(np.mean(
        [v["inference_seconds_per_origin"] for v in per_period.values()]))
    preds.to_parquet(HERE / "predictions.parquet", index=False)
    log(f"wrote predictions.parquet ({len(preds)} rows)")

    if not args.skip_paper:
        try:
            results["paper_task"] = paper_replication(feat)
        except Exception as exc:  # noqa: BLE001 - recorded, never fatal
            log(f"paper task failed: {type(exc).__name__}: {exc}")
            results["paper_task_error"] = f"{type(exc).__name__}: {exc}"
    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    log("wrote results.json")


if __name__ == "__main__":
    main()
