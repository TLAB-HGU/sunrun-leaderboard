"""B. Information-headroom checks on the frozen candidate (selection Dec-2025..Feb-2026, audit Mar..May-2026).

B2  timing oracle: per-origin best time shift of the 72h curve within +-k h (uses truth, an upper bound only).
B1  residual predictability: forward-in-time model trained on selection, scored on audit, predicting the
    candidate's residual from causal features. Arms: control (speed history), existing (all inputs we already own),
    existing+donki (DONKI WSA-ENLIL runs with modelCompletionTime <= origin).
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

LB = Path("/home/t-lab01/sunrun/leaderboard")
RUN = LB / "store/mse20-four-regimes/run-20261003"
OUT = LB / "store/headroom-audit"
SEL, AUD = RUN / "selected-selection-predictions.parquet", RUN / "audit-predictions.parquet"
H = np.arange(1, 73)
LEADS = [(1, 6), (7, 24), (25, 48), (49, 72)]


def hourly(df, col="timestamp_utc"):
    df = df.copy()
    df.index = pd.to_datetime(df.pop(col), utc=True).dt.floor("h")
    return df[~df.index.duplicated(keep="last")].sort_index()


def load_truth():
    ace = hourly(pd.read_parquet(LB / "store/ch-v1/ace.parquet"))
    return ace.loc[~ace["was_missing"].astype(bool), "filled_speed_kms"]


def wide(path, truth):
    p = pd.read_parquet(path)
    p["origin"] = pd.to_datetime(p["origin_last_input_utc"], utc=True)
    pred = p.pivot(index="origin", columns="horizon_hours", values="pred_kms")[H]
    tt = pred.index.values[:, None] + (H * 3600 * 10**9).astype("timedelta64[ns]")
    y = truth.reindex(pd.DatetimeIndex(tt.ravel(), tz="UTC")).to_numpy().reshape(pred.shape)
    return pred, y


def timing_oracle(pred, y):
    P = pred.to_numpy()
    obs = ~np.isnan(y)
    res = {}
    for k in (0, 6, 12, 24):
        best = np.full(len(P), np.inf)
        for s in range(-k, k + 1):
            idx = np.clip(np.arange(72) - s, 0, 71)
            se = np.where(obs, (P[:, idx] - np.nan_to_num(y)) ** 2, 0).sum(1)
            best = np.minimum(best, se)
        res[f"shift_pm{k}h"] = float(best.sum() / obs.sum())
    bias = np.nanmean(y - P, axis=1, keepdims=True)  # per-origin constant level oracle
    res["level_oracle"] = float(np.nanmean((P + bias - y) ** 2))
    return res


def ace_features(origins):
    raw = hourly(pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(
        LB / "store/ch-breakthrough-v2/upload/raw-ace-snapshot/year=*/part.parquet")))))
    cols = ["ace_speed_kms", "ace_density_cm3", "ace_temperature_k", "ace_bt_nt", "ace_bz_gsm_nt", "ace_by_gsm_nt"]
    logs = ["ace_epam_e_channel1", "ace_epam_e_channel2", "ace_epam_p_channel1", "ace_epam_p_channel3",
            "ace_epam_p_channel5", "ace_sis_gt10_flux"]
    x = raw[cols].astype(float)
    x[logs] = np.log10(raw[logs].astype(float).clip(lower=1e-3))
    x = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq="h", tz="UTC"))
    f = {}
    for c in x.columns:  # rolling windows end at the origin hour: causal
        f[f"{c}_last"] = x[c].ffill(limit=6)
        f[f"{c}_m6"] = x[c].rolling(6, min_periods=1).mean()
        f[f"{c}_m24"] = x[c].rolling(24, min_periods=1).mean()
    f["speed_d24"] = f["ace_speed_kms_m6"] - f["ace_speed_kms_m6"].shift(24)
    f["epam_p1_d6"] = f["ace_epam_p_channel1_m6"] - f["ace_epam_p_channel1_m6"].shift(6)
    return pd.DataFrame(f).reindex(origins)


def ch_features(origins):
    ch = pd.read_parquet(LB / "store/ch-v1/ch-hourly.parquet")
    if not isinstance(ch.index, pd.DatetimeIndex):
        tcol = next(c for c in ch.columns if "time" in c.lower())
        ch = ch.set_index(pd.to_datetime(ch.pop(tcol), utc=True))
    ch = ch.sort_index()
    keep = [c for c in ch.columns if any(s in c for s in ("meridian", "equatorial", "_area")) and "lon" not in c]
    ch = ch[keep].select_dtypes("number")
    return ch.reindex(origins, method="ffill", tolerance=pd.Timedelta("6h")).add_prefix("ch_")


def donki_features(origins):
    sims = []
    for f in sorted(glob.glob(str(OUT / "donki/WSAEnlilSimulations_*.json"))):
        sims += json.load(open(f))
    rows = []
    for s in sims:
        if not s.get("estimatedShockArrivalTime"):
            continue
        rows.append((pd.Timestamp(s["modelCompletionTime"]), pd.Timestamp(s["estimatedShockArrivalTime"]),
                     max([s.get("kp_90") or 0, s.get("kp_135") or 0, s.get("kp_180") or 0])))
    ev = pd.DataFrame(rows, columns=["done", "arrival", "kp"]).sort_values("done")
    return ev


def donki_matrix(ev, origins, horizon):
    """Signed hours from target time to the nearest Earth shock predicted by runs completed <= origin."""
    out = np.full((len(origins), 3), np.nan)
    for i, o in enumerate(origins):
        e = ev.iloc[: ev["done"].searchsorted(o, side="right")]
        e = e[(e["done"] >= o - pd.Timedelta("7D")) & (e["arrival"] >= o - pd.Timedelta("3D"))]
        if len(e):
            out[i] = [(e["arrival"] - o).dt.total_seconds().min() / 3600, len(e), e["kp"].max()]
    base = pd.DataFrame(out, index=origins, columns=["enlil_next_arr_h", "enlil_n_runs", "enlil_kp"])
    return base, horizon


def long_table(pred, y, feats, origins):
    n = len(origins)
    df = pd.DataFrame({"origin": np.repeat(origins, 72), "h": np.tile(H, n),
                       "pred": pred.to_numpy().ravel(), "y": y.ravel()})
    df = df.join(feats, on="origin")
    if "enlil_next_arr_h" in df:
        df["enlil_dt_target"] = df["h"] - df["enlil_next_arr_h"]  # target time minus predicted arrival
    return df.dropna(subset=["y"])


def recurrence(df, truth):
    for d in (26, 27, 28):
        t = df["origin"] + pd.to_timedelta(df["h"] - 24 * d, unit="h")
        df[f"rec{d}"] = truth.reindex(pd.DatetimeIndex(t)).to_numpy()
    df["rec_mean"] = df[["rec26", "rec27", "rec28"]].mean(1)
    df["rec_minus_pred"] = df["rec_mean"] - df["pred"]
    return df


def fit_score(tr, te, cols, seed=0):
    m = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7,
                         min_child_weight=50, reg_lambda=10, tree_method="hist", random_state=seed, n_jobs=8)
    m.fit(tr[cols], tr["y"] - tr["pred"])
    corr = te["pred"] + m.predict(te[cols])
    base_se, corr_se = (te["pred"] - te["y"]) ** 2, (corr - te["y"]) ** 2
    r = {"base_mse": float(base_se.mean()), "corrected_mse": float(corr_se.mean())}
    r["gain_pct"] = 100 * (1 - r["corrected_mse"] / r["base_mse"])
    for a, b in LEADS:
        m_ = te["h"].between(a, b)
        r[f"gain_pct_{a}-{b}h"] = float(100 * (1 - corr_se[m_].mean() / base_se[m_].mean()))
    imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
    r["top_features"] = imp.head(8).round(4).to_dict()
    return r


def main():
    truth = load_truth()
    out = {"B2_timing_oracle": {}, "B1_residual_predictability": {}}
    data = {}
    ev = donki_features(None)
    for name, path in (("selection", SEL), ("audit", AUD)):
        pred, y = wide(path, truth)
        out["B2_timing_oracle"][name] = timing_oracle(pred, y)
        print(name, out["B2_timing_oracle"][name])
        o = pred.index
        feats = ace_features(o).join(ch_features(o)).join(donki_matrix(ev, o, None)[0])
        data[name] = recurrence(long_table(pred, y, feats, o), truth)
    tr, te = data["selection"], data["audit"]
    control = ["h", "pred", "ace_speed_kms_last", "ace_speed_kms_m6", "ace_speed_kms_m24", "speed_d24"]
    donki = ["enlil_next_arr_h", "enlil_n_runs", "enlil_kp", "enlil_dt_target"]
    existing = [c for c in tr.columns if c not in ("origin", "y", *donki)]
    arms = {"control_speed": control, "existing_all": existing, "existing_plus_donki": existing + donki,
            "control_plus_donki": control + donki}
    for arm, cols in arms.items():
        out["B1_residual_predictability"][arm] = {
            "forward_sel_to_audit": fit_score(tr, te, cols),
            "reverse_audit_to_sel_sensitivity": fit_score(te, tr, cols)}
        f, r = out["B1_residual_predictability"][arm].values()
        print(f"{arm:22s} fwd gain {f['gain_pct']:6.2f}%  rev gain {r['gain_pct']:6.2f}%  n_feat {len(cols)}")
    out["notes"] = {"donki_runs_with_earth_arrival": int(len(ev)),
                    "n_rows": {"selection": int(len(tr)), "audit": int(len(te))}}
    (OUT / "headroom.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
