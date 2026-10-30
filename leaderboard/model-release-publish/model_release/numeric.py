"""Frozen numeric feature arithmetic extracted from submitted source recipes."""
import numpy as np
import pandas as pd

def hourly(df, col="timestamp_utc"):
    df = df.copy()
    df.index = pd.to_datetime(df.pop(col), utc=True).dt.floor("h")
    return df[~df.index.duplicated(keep="last")].sort_index()

def ace_features(origins, raw_frame):
    raw = hourly(raw_frame)
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

def ch_features(origins, ch):
    ch = ch.copy()
    if not isinstance(ch.index, pd.DatetimeIndex):
        tcol = next(c for c in ch.columns if "time" in c.lower())
        ch = ch.set_index(pd.to_datetime(ch.pop(tcol), utc=True))
    ch = ch.sort_index()
    keep = [c for c in ch.columns if any(s in c for s in ("meridian", "equatorial", "_area")) and "lon" not in c]
    ch = ch[keep].select_dtypes("number")
    return ch.reindex(origins, method="ffill", tolerance=pd.Timedelta("6h")).add_prefix("ch_")
def speed_features(origins, truth_filled):
    s = truth_filled
    f = {f"v_lag{l}": s.shift(l) for l in (0, 1, 3, 6, 12, 24, 48, 72, 120)}
    for w in (6, 24, 72):
        f[f"v_m{w}"] = s.rolling(w, min_periods=1).mean()
        f[f"v_sd{w}"] = s.rolling(w, min_periods=2).std()
    return pd.DataFrame(f).reindex(origins)

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
