"""B1'. Information ablation on the full history: one fixed direct XGBoost recipe, only the input families change.

Train on labels before each test period (72h purge, expanding), test on dev / selection / audit. Origins every 3h.
Arms add information sources one at a time so the MSE drop measures what each source carries.
DONKI WSA-ENLIL features only use runs with modelCompletionTime <= origin.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from headroom import LB, OUT, ace_features, ch_features, hourly, load_truth  # noqa: E402

H = np.arange(1, 73)
PERIODS = {"dev": ("2025-08-04", "2025-11-27 23:00"), "selection": ("2025-12-04", "2026-02-25 23:00"),
           "audit": ("2026-03-04", "2026-05-28 23:00")}
START = "2022-10-01"
PARAMS = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
              min_child_weight=20, reg_lambda=5, tree_method="hist", n_jobs=8, random_state=0)


def speed_features(origins, truth_filled):
    s = truth_filled
    f = {f"v_lag{l}": s.shift(l) for l in (0, 1, 3, 6, 12, 24, 48, 72, 120)}
    for w in (6, 24, 72):
        f[f"v_m{w}"] = s.rolling(w, min_periods=1).mean()
        f[f"v_sd{w}"] = s.rolling(w, min_periods=2).std()
    return pd.DataFrame(f).reindex(origins)


def enlil_runs():
    rows = []
    for f in sorted(glob.glob(str(OUT / "donki/WSAEnlilSimulations_*.json"))):
        for s in json.load(open(f)):
            if s.get("estimatedShockArrivalTime"):
                rows.append((pd.Timestamp(s["modelCompletionTime"]), pd.Timestamp(s["estimatedShockArrivalTime"]),
                             max(s.get("kp_90") or 0, s.get("kp_135") or 0, s.get("kp_180") or 0),
                             bool(s.get("isEarthGB"))))
    return pd.DataFrame(rows, columns=["done", "arrival", "kp", "glancing"]).drop_duplicates().sort_values("done")


def enlil_features(origins, ev):
    out = np.full((len(origins), 5), np.nan)
    for i, o in enumerate(origins):
        e = ev.iloc[: ev["done"].searchsorted(o, side="right")]
        e = e[e["done"] >= o - pd.Timedelta("7D")]
        dt = (e["arrival"] - o).dt.total_seconds() / 3600
        fut, past = dt[dt >= -6], dt[(dt < -6) & (dt >= -96)]
        out[i] = [fut.min() if len(fut) else np.nan, -past.max() if len(past) else np.nan,
                  len(fut), e.loc[fut.index, "kp"].max() if len(fut) else 0,
                  float(e.loc[fut.index, "glancing"].all()) if len(fut) else np.nan]
    return pd.DataFrame(out, index=origins,
                        columns=["enlil_next_arr_h", "enlil_since_arr_h", "enlil_n_pending", "enlil_kp", "enlil_glancing"])


def build(freq="3h", start=None, end=None):
    truth = load_truth()
    ace = hourly(pd.read_parquet(LB / "store/ch-v1/ace.parquet"))["filled_speed_kms"]
    origins = pd.date_range(start or START, end or PERIODS["audit"][1], freq=freq, tz="UTC")
    feats = (speed_features(origins, ace).join(ace_features(origins), rsuffix="_raw")
             .join(ch_features(origins)).join(enlil_features(origins, enlil_runs())))
    n = len(origins)
    df = pd.DataFrame({"origin": np.repeat(origins, 72), "h": np.tile(H, n)})
    df["target_t"] = df["origin"] + pd.to_timedelta(df["h"], unit="h")
    df["y"] = truth.reindex(pd.DatetimeIndex(df["target_t"])).to_numpy()
    for d in (26, 27, 28):
        df[f"rec{d}"] = ace.reindex(pd.DatetimeIndex(df["target_t"] - pd.Timedelta(days=d))).to_numpy()
    df["rec_mean"] = df[["rec26", "rec27", "rec28"]].mean(1)
    df = df.join(feats, on="origin").dropna(subset=["y"])
    df["enlil_dt_target"] = df["h"] - df["enlil_next_arr_h"]
    return df


def arms(cols):
    speed = ["h"] + [c for c in cols if c.startswith(("v_", "rec"))]
    plasma = [c for c in cols if c.startswith("ace_") or c in ("speed_d24", "epam_p1_d6")]
    ch = [c for c in cols if c.startswith("ch_")]
    enlil = [c for c in cols if c.startswith("enlil_")]
    return {"speed_only": speed, "speed+plasma": speed + plasma, "speed+ch": speed + ch,
            "existing_all": speed + plasma + ch, "speed+enlil": speed + enlil,
            "existing+enlil": speed + plasma + ch + enlil}


def main():
    df = build()
    res = {"n_rows": int(len(df)), "periods": PERIODS, "params": PARAMS, "results": {}}
    A = arms(df.columns)
    for pname, (a, b) in PERIODS.items():
        a, b = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        tr = df[df["target_t"] < a - pd.Timedelta("72h")]
        te = df[df["origin"].between(a, b)]
        res["results"][pname] = {"n_train": int(len(tr)), "n_test": int(len(te))}
        for arm, cols in A.items():
            m = xgb.XGBRegressor(**PARAMS).fit(tr[cols], tr["y"])
            se = (m.predict(te[cols]) - te["y"]) ** 2
            r = {"mse": float(se.mean())}
            for lo, hi in ((1, 6), (7, 24), (25, 48), (49, 72)):
                r[f"mse_{lo}-{hi}h"] = float(se[te["h"].between(lo, hi)].mean())
            if arm == "existing+enlil":
                imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
                r["top_features"] = imp.head(10).round(4).to_dict()
            res["results"][pname][arm] = r
            print(f"{pname:9s} {arm:15s} MSE {r['mse']:8.1f} | 25-48h {r['mse_25-48h']:8.1f} | 49-72h {r['mse_49-72h']:8.1f}", flush=True)
    (OUT / "info_ablation.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
