"""Frozen longitude-strip arithmetic; availability comes from an explicit manifest."""
import numpy as np
import pandas as pd

OMEGA = 0.55
D_KM = 147858620.7
SPEEDS = (350, 450, 550, 650, 750)
TAU = {v: D_KM / v / 3600.0 for v in SPEEDS}
PARKER_WEST = {v: OMEGA * TAU[v] for v in SPEEDS}
CELL_W = 15.0
STRIP_HALF = 5.0
COVER_MIN = 0.5
STALE_H = 6.0
MARGIN_H = 1.0
H_ALL = np.arange(1, 73)
STAT_SUFFIX = {"logmed": "log_median", "dark": "dark_fraction", "cover": "valid_fraction"}

def strip_columns() -> list:
    cols = [f"strip_v{v}_{s}" for v in SPEEDS for s in STAT_SUFFIX]
    return cols + ["img_age_h", "img_stale"]

def lam(h, v, age_h) -> float:
    """West-positive strip centre (deg) for horizon h, speed v, image age."""
    return OMEGA * (TAU[v] - (np.asarray(h, dtype=float) + age_h))

def _cell_centers() -> np.ndarray:
    return -90.0 + CELL_W * (np.arange(12) + 0.5)

def strip_weights(lam_c: float) -> tuple:
    """Overlap weights of the [lam_c-5, lam_c+5] strip over the 12 lon cells."""
    lo, hi = lam_c - STRIP_HALF, lam_c + STRIP_HALF
    edges = -90.0 + CELL_W * np.arange(13)
    w = np.clip(np.minimum(hi, edges[1:]) - np.maximum(lo, edges[:-1]), 0, None) / (2 * STRIP_HALF)
    return tuple(float(x) for x in w)

def select_slots(origins, meta, block) -> pd.DataFrame:
    """Newest proxy-eligible slot per origin (vectorised walk-back).

    Eligible: status ok, slot <= O-1h, s3_modified <= O, lat1 cells finite,
    local file present. Falls back to the newest earlier qualifying slot;
    no qualifying slot -> missing (NaN + stale).
    """
    slots = meta["slot"].to_numpy(dtype="datetime64[ns]")
    obs = meta["obs_end"].to_numpy(dtype="datetime64[ns]")
    s3m = meta["s3_modified"].to_numpy(dtype="datetime64[ns]")
    okstat = (meta["status"] == "ok").to_numpy()
    finite = block.reindex(pd.DatetimeIndex(meta["slot"])).notna().all(axis=1).to_numpy()
    present = meta["available"].to_numpy(dtype=bool)
    o = pd.DatetimeIndex(pd.to_datetime(origins, utc=True)).to_numpy(dtype="datetime64[ns]")
    margin = np.timedelta64(int(MARGIN_H * 3600), "s")
    j = np.searchsorted(slots, o - margin, side="right") - 1
    o_ns = o.astype("datetime64[ns]").astype(np.int64)
    s3_ns = s3m.astype(np.int64)
    sel = np.full(len(o), -1, dtype=np.int64)
    for i in range(len(o)):
        k = int(j[i])
        while k >= 0 and not (okstat[k] and s3_ns[k] <= o_ns[i] and finite[k] and present[k]):
            k -= 1
        sel[i] = k
    out = pd.DataFrame({"origin": pd.DatetimeIndex(o, tz="UTC")})
    out["slot_idx"] = sel
    out["slot"] = pd.to_datetime(np.where(sel >= 0, slots[np.maximum(sel, 0)], np.datetime64("NaT")), utc=True)
    out["obs_end"] = pd.to_datetime(np.where(sel >= 0, obs[np.maximum(sel, 0)], np.datetime64("NaT")), utc=True)
    age = (o_ns - np.where(sel >= 0, obs[np.maximum(sel, 0)].astype(np.int64), np.int64(0))) / 3.6e12
    out["img_age_h"] = np.where(sel >= 0, age, np.nan)
    out["img_stale"] = np.where(sel < 0, 1.0, np.where(out["img_age_h"] > STALE_H, 1.0, 0.0))
    out["missing"] = sel < 0
    return out

def strip_features(origins, meta=None, block=None, selection=None) -> tuple:
    """17 strip columns for every (origin, h) row + per-origin selection."""
    if meta is None or block is None:
        raise ValueError("Explicit SUVI metadata and numeric block are required")
    if selection is None:
        selection = select_slots(origins, meta, block)
    lat1 = list(block.columns)
    stats = {s: np.array([[block[c].to_numpy()[k] for c in
                           sorted(lat1, key=lambda c: int(c[3:5])) if c.endswith(suf)]
                          for k in range(len(block))], dtype=float)
             for s, suf in STAT_SUFFIX.items()}
    slot_of = selection["slot_idx"].to_numpy()
    ages = selection["img_age_h"].to_numpy()
    stale = (selection["img_stale"].to_numpy() > 0)
    n = len(selection)
    feats = np.full((n * 72, len(strip_columns()) - 2), np.nan)
    edges = -90.0 + CELL_W * np.arange(13)
    tau_arr = np.array([TAU[v] for v in SPEEDS])
    for i in range(n):
        if slot_of[i] < 0 or stale[i]:
            continue
        k, age = int(slot_of[i]), float(ages[i])
        base = i * 72
        LAM = OMEGA * (tau_arr[None, :] - (H_ALL[:, None] + age))  # (72, 5)
        W = np.clip(np.minimum(LAM[..., None] + STRIP_HALF, edges[1:])
                    - np.maximum(LAM[..., None] - STRIP_HALF, edges[:-1]), 0, None) / (2 * STRIP_HALF)
        wsum = W.sum(-1)  # (72, 5)
        for vi in range(len(SPEEDS)):
            w = W[:, vi, :]
            ws = wsum[:, vi]
            cov = np.where(ws > 0, (w * stats["cover"][k]).sum(-1) / np.maximum(ws, 1e-12), np.nan)
            feats[base:base + 72, vi * 3 + 2] = cov
            ok = cov >= COVER_MIN
            lm = np.where(ws > 0, (w * stats["logmed"][k]).sum(-1) / np.maximum(ws, 1e-12), np.nan)
            dk = np.where(ws > 0, (w * stats["dark"][k]).sum(-1) / np.maximum(ws, 1e-12), np.nan)
            feats[base:base + 72, vi * 3] = np.where(ok, lm, np.nan)
            feats[base:base + 72, vi * 3 + 1] = np.where(ok, dk, np.nan)
    df = pd.DataFrame({"origin": np.repeat(selection["origin"].to_numpy(), 72),
                       "h": np.tile(H_ALL, n)})
    for ci, name in enumerate(strip_columns()[:-2]):
        df[name] = feats[:, ci].astype(np.float32)
    df["img_age_h"] = np.repeat(np.where(slot_of < 0, np.nan, ages).astype(np.float32), 72)
    df["img_stale"] = np.repeat(stale.astype(np.float32), 72)
    df["origin"] = pd.to_datetime(df["origin"], utc=True)
    return df, selection
