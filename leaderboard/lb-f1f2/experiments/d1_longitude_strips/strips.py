"""D1 mechanism A: forecast-time-aligned SUVI longitude strips (proxy rule).

Each (origin O, horizon h) row gets strip features from the SINGLE newest
proxy-eligible SUVI image: newest hourly slot S with S <= O-1h, S3
LastModified <= O, local bytes present, and finite equatorial cells; else the
newest earlier qualifying slot plus an age feature; none -> NaN + stale flag.

Geometry (west-positive, single formula, never double-counted):
  lambda(h,v) = OMEGA * (TAU[v] - (T - t_img)),  T = O + h (hours),
  t_img = obs_end of the selected image, OMEGA = 0.55 deg/h prograde.
A 10deg-wide strip (+-5deg) at 1deg sampling would need deprojected pixels;
the hourly SUVI cache only resolves 15deg lon cells, so each strip is the
overlap-weighted mean of the intersected equatorial (lat1 ~ -30..+30) cells.
Latitude band and 15deg resolution are documented approximations, not fits.

Image source (read-only, preofficial only): the causal hourly SUVI cache
(store/suvi-fusion/suvi-hourly-v2.parquet, obs_end <= slot enforced at
build) joined to the slot metadata export
(/home/t-lab01/.local/state/sundb/stage/suvi_hourly) for obs_end,
s3_modified (S3 LastModified) and the local file path. No official-window
truth is read anywhere in this module.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urlrequest

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[2]
SUVI_CACHE = LB / "store/suvi-fusion/suvi-hourly-v2.parquet"
SUNIMG = LB.parent / "sun-img"
META_GLOB = "/home/t-lab01/.local/state/sundb/stage/suvi_hourly/year=*/part.parquet"
S3_BASE = "https://noaa-goes18.s3.amazonaws.com"


def resolve_nc(path: str) -> str:
    """Stage nc_path may be relative (resolved under sun-img/) or absolute."""
    if not path:
        return ""
    p = Path(path)
    return str(p if p.is_absolute() else SUNIMG / p)

# Frozen geometry (matches the approved image plan arithmetic).
OMEGA = 0.55  # deg/h prograde
D_KM = 147858620.7  # AU - 2.5 Rsun
SPEEDS = (350, 450, 550, 650, 750)
TAU = {v: D_KM / v / 3600.0 for v in SPEEDS}  # hours; ~117.35/91.27/74.68/63.19/54.76
PARKER_WEST = {v: OMEGA * TAU[v] for v in SPEEDS}  # h=0, fresh image
CELL_W = 15.0  # cache lon cells cover [-90,+90] at 15deg, west-positive
STRIP_HALF = 5.0  # 10deg full width
COVER_MIN = 0.5  # strip with <50% valid pixels -> NaN for that (h,v)
STALE_H = 6.0  # O - obs_end > 6h -> stale -> all-NaN + flag
MARGIN_H = 1.0  # proxy: image slot must satisfy S <= O - 1h
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


def load_inputs() -> tuple:
    """Slot metadata (obs_end, s3_modified, path) + equatorial cache block."""
    meta = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(META_GLOB))], ignore_index=True)
    meta["slot"] = pd.to_datetime(meta["slot"], utc=True)
    meta["obs_end"] = pd.to_datetime(meta["obs_end"], utc=True)
    meta["s3_modified"] = pd.to_datetime(meta["s3_modified"], utc=True)
    meta = meta.sort_values("slot").reset_index(drop=True)
    cache = pd.read_parquet(SUVI_CACHE)
    cache.index = pd.to_datetime(cache.index, utc=True).tz_convert("UTC")
    lat1 = [c for c in cache.columns if "_lat1_" in c]
    block = cache[sorted(lat1)]
    return meta, block


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
    raw_paths = meta["nc_path"].where(meta["nc_path"].notna(), "").astype(str).tolist()
    paths = [resolve_nc(p) for p in raw_paths]
    present = np.array([bool(p) and os.path.isfile(p) and os.path.getsize(p) > 0 for p in paths])
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
        meta, block = load_inputs()
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


def _head(url: str, timeout=20) -> dict:
    req = urlrequest.Request(url, method="HEAD")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as r:
            h = {k.lower(): v for k, v in r.headers.items()}
            return {"status": r.status, "etag": h.get("etag", "").strip('"'),
                    "last_modified": h.get("last-modified", ""),
                    "length": h.get("content-length", "")}
    except Exception as exc:  # noqa: BLE001 - recorded, never raised
        return {"status": None, "error": f"{type(exc).__name__}: {exc}"}


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(periods, n_per=200, seed=0, workers=24) -> dict:
    """200 samples/period: local MD5 vs S3 ETag HEAD + proxy-timing record."""
    meta, _ = load_inputs()
    meta["slot"] = pd.to_datetime(meta["slot"], utc=True)
    rng = np.random.default_rng(seed)
    jobs = []
    for name, (a, b) in periods.items():
        pool = meta[meta["status"].eq("ok") & meta["slot"].between(a, b)].reset_index(drop=True)
        take = rng.choice(len(pool), size=min(n_per, len(pool)), replace=False)
        for t in take:
            jobs.append((name, pool.loc[int(t)]))

    def one(job):
        name, row = job
        key = str(row["s3_key"]).rsplit(".", 1)[0] + ".nc"
        rec = {"period": name, "slot": pd.Timestamp(row["slot"]).isoformat(),
               "obs_end": pd.Timestamp(row["obs_end"]).isoformat(),
               "stage_s3_modified": pd.Timestamp(row["s3_modified"]).isoformat(),
               "s3_key": key, "nc_path": str(row["nc_path"])}
        p = resolve_nc(str(row["nc_path"]))
        if not (p and os.path.isfile(p)):
            return rec | {"local": "missing"}
        rec["local_md5"] = _md5(p)
        rec["local_bytes"] = os.path.getsize(p)
        head = _head(f"{S3_BASE}/{key}")
        rec["head"] = head
        if head.get("status") == 200:
            etag = head.get("etag", "")
            rec["etag_match"] = ("multipart" if "-" in etag else (etag == rec["local_md5"]))
            try:
                lm = pd.Timestamp(head["last_modified"]).tz_convert("UTC")
                rec["head_last_modified"] = lm.isoformat()
                rec["head_agrees_stage"] = bool(abs((lm - pd.Timestamp(row["s3_modified"])).total_seconds()) < 5)
                rec["obs_end_le_slot"] = bool(pd.Timestamp(row["obs_end"]) <= pd.Timestamp(row["slot"]))
            except Exception as exc:  # noqa: BLE001
                rec["time_parse_error"] = str(exc)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, jobs))
    out = {"n_per_period": n_per, "seed": seed, "samples": rows}
    ok = [r for r in rows if r.get("etag_match") is True]
    out["summary"] = {"n": len(rows), "md5_etag_match": len(ok),
                      "multipart": sum(1 for r in rows if r.get("etag_match") == "multipart"),
                      "head_fail": sum(1 for r in rows if r.get("head", {}).get("status") != 200),
                      "local_missing": sum(1 for r in rows if r.get("local") == "missing")}
    return out


def poison_test(origins, cutoff, perturb=10.0, meta=None, block=None) -> dict:
    """Real future-poison test: perturb post-cutoff inputs, check pre-cutoff
    invariance (exact, NaN-aware) plus positive-control movement."""
    if meta is None or block is None:
        meta, block = load_inputs()
    base, sel = strip_features(origins, meta, block)
    cut = pd.to_datetime(cutoff, utc=True)
    pert = block.copy()
    after = pert.index > cut
    pert.loc[after] = pert.loc[after] + perturb
    mod, _ = strip_features(origins, meta, pert, selection=sel)
    cols = strip_columns()
    slot_time = pd.to_datetime(sel["slot"])
    pre_mask = np.repeat(((sel["slot_idx"].to_numpy() >= 0) & (slot_time <= cut).fillna(False).to_numpy()), 72)
    post_mask = np.repeat(((sel["slot_idx"].to_numpy() >= 0) & (slot_time > cut).fillna(False).to_numpy()), 72)
    pre_inv = all(np.array_equal(base.loc[pre_mask, c].to_numpy(), mod.loc[pre_mask, c].to_numpy(),
                                 equal_nan=True) for c in cols)
    moved = [c for c in cols if not np.array_equal(base.loc[post_mask, c].to_numpy(),
                                                   mod.loc[post_mask, c].to_numpy(), equal_nan=True)]
    passed = bool(pre_inv and len(moved) > 0 and post_mask.any())
    return {"passed": passed, "cutoff_utc": cut.isoformat(), "perturbation": perturb,
            "n_pre_rows": int(pre_mask.sum()), "pre_invariant": bool(pre_inv),
            "n_post_rows": int(post_mask.sum()), "moved_columns": moved,
            "note": "perturbed post-cutoff cache inputs; pre-cutoff strip features bitwise identical"}


def manifest_json_default(o):
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    raise TypeError(repr(o))


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
