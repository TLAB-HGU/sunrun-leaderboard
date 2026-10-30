"""Physical-quality lane: ACE physics + CH mask-quality vs speed-only control.

Three matched-parameter arms (identical XGBoost params, identical splits):
  speed_only               ACE filled-speed history only (foundation ``ace_*``).
  physical_ace             speed + raw ACE physics (density / temperature / |B|,
                           valid derived dynamic pressure, short causal
                           gradients, missingness / status / age features).
  physical_ace_ch_quality  physical_ACE + causal CH mask-quality features
                           (t035/t045/t055 threshold agreement, valid_fraction /
                           count / max-area quality, geometry + area continuity).

Causality: every feature row at origin ``o`` uses source timestamps ``<= o``
only. Training never fits observations or label targets past the declared
cutoff (``PooledResidualForecaster.fit`` enforces it); split boundaries keep a
>=72h purge via ``purged_split``. No in-sample fitted-residual shortcut: the
residual target is anchored on the fixed mean-reversion baseline.

Raw ACE inputs come from the staged ``ace_solar`` store via
``common.load_raw_ace``. Its ``point_in_time_vintage_verified`` column is
identically zero: this operational-vintage limitation is recorded in every
manifest/metrics file and the output is NEVER marked verified.

CH mask reliability: ``store/ch-v1/masks/`` holds rendered SVG/PNG overlays
only (no per-object mask tables or FITS headers), so full object-mask
re-extraction is out of budget. Quality is therefore assessed from causal
compact-cache summaries (threshold agreement, valid_fraction, count,
max_area, equatorial/meridian geometry, area continuity), and that
delimitation is recorded explicitly. Header conventions consumed by the
ch-v1 extraction pipeline are validated functionally (synthetic-header probe
of ``coronal_holes.geometry``) without asserting any unproven WCS
sign/scale error.
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.ch_breakthrough_v2"

from .common import (  # noqa: E402
    HORIZON_HOURS,
    HORIZONS,
    N_THREADS,
    SEED,
    PooledResidualForecaster,
    acquire_gpu_lease,
    as_speed_series,
    build_features,
    ensure_device,
    file_identity,
    load_ace_hourly,
    load_ch_hourly,
    load_raw_ace,
    observed_mask,
    purged_split,
    score_forecast,
    subsample_origins,
    timed_predict,
    utc,
    write_manifest,
)

ARMS = ("speed_only", "physical_ace", "physical_ace_ch_quality")

# Matched parameters: identical for every arm (no per-arm tuning).
PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 4,
    "reg_lambda": 1.0,
}

# Proton-only dynamic pressure: P[nPa] = m_p * n[m^-3] * v[m/s]^2 * 1e9
# = 1.6726e-6 * n[cm^-3] * v[km/s]^2. Documented, no helium correction.
PDYN_FACTOR = 1.6726e-6

PHYS_CHANNELS = ("dens", "temp", "bt", "pdyn")
CH_Q_TAGS = ("t035", "t045", "t055")

SPLITS = {
    "dev-A": {"cutoff": "2024-12-31T23:00:00Z", "start": "2025-01-01T00:00:00Z",
              "end": "2025-03-31T23:00:00Z", "screen": True},
    "dev-B": {"cutoff": "2025-04-30T23:00:00Z", "start": "2025-05-01T00:00:00Z",
              "end": "2025-07-31T23:00:00Z", "screen": True},
    "dev-C": {"cutoff": "2025-08-31T23:00:00Z", "start": "2025-09-01T00:00:00Z",
              "end": "2025-11-30T23:00:00Z", "screen": True},
    "selection": {"cutoff": "2025-11-30T23:00:00Z", "start": "2025-12-01T00:00:00Z",
                  "end": "2026-02-28T23:00:00Z", "screen": True},
    "holdout": {"cutoff": "2026-02-28T23:00:00Z", "start": "2026-03-01T00:00:00Z",
                "end": "2026-05-31T23:00:00Z", "screen": True},
    # Pinned official folds (matches ch-v1: 2833 hourly origins starting at
    # the cutoff). Single pass, refit through cutoff, label exploratory.
    "official": {"cutoff": "2026-05-31T23:00:00Z", "start": "2026-05-31T23:00:00Z",
                 "end": "2026-09-26T23:00:00Z", "screen": False},
}

VINTAGE_LIMITATION = (
    "Raw ACE point_in_time_vintage_verified is identically 0 over the full "
    "staged history: operational (not reprocessed-vintage) data. Results are "
    "NEVER marked vintage-verified; treat as an operational-vintage "
    "validation limitation, not proven leakage."
)

MASK_DELIMITATION = (
    "store/ch-v1/masks/ holds rendered SVG/PNG overlays only (no per-object "
    "mask tables or FITS headers). Full object-mask re-extraction was out of "
    "budget, so CH reliability is assessed from causal compact-cache quality "
    "/ continuity summaries only."
)

HEADER_NOTE = (
    "ch-v1 extraction consumes CRPIX1/CRPIX2/RSUN (+optional CROTA/SOLAR_B0/"
    "PC1_1/PC1_2/PC2_1/PC2_2) via coronal_holes.geometry; functionally probed "
    "with a synthetic header (shape/disk/lat/lon/area contract only). No WCS "
    "sign/scale error is asserted."
)


# ---------------------------------------------------------------------------
# Causal window helper (lane feature engineering; protocol pieces stay in common)
# ---------------------------------------------------------------------------

def _window(source: pd.Series, origins: pd.DatetimeIndex, width: int) -> np.ndarray:
    """Causal (origin-width+1 .. origin) values; NaN where unknown."""
    grid = pd.date_range(origins.min() - pd.Timedelta(hours=width - 1),
                         origins.max(), freq="h")
    pos = grid.get_indexer(origins) - (width - 1)
    base = source.reindex(grid).to_numpy(dtype=float)
    return np.lib.stride_tricks.sliding_window_view(base, width)[pos]


def _nanstat(func, values, axis=1):
    # All-NaN windows stay NaN by design (native XGBoost missingness);
    # silence the expected empty-slice warnings (mirrors common._nanstat).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func(values, axis=axis)


def _finite_or_nan(frame: pd.DataFrame) -> pd.DataFrame:
    """Pointwise inf -> NaN. Ratios over zero-area masks yield inf; XGBoost
    rejects inf, while NaN stays native missingness. Causal (pointwise)."""
    out = frame.replace([np.inf, -np.inf], np.nan)
    return out.astype(np.float32, errors="ignore")
def _since_valid(mask: np.ndarray) -> np.ndarray:
    """Hours since last valid (finite/True) sample in each causal window."""
    time = np.arange(mask.shape[1])[None, :]
    last = np.maximum.accumulate(np.where(mask, time, -1), axis=1)
    age = (mask.shape[1] - 1 - last[:, -1]).astype(float)
    return np.where(last[:, -1] < 0, np.nan, age)


# ---------------------------------------------------------------------------
# Raw ACE physics features
# ---------------------------------------------------------------------------

def load_staged_raw_ace(pattern: str = "/home/t-lab01/.local/state/sundb/stage/ace_solar/year=*/part.parquet") -> pd.DataFrame:
    """Concat staged raw ACE year parts via common.load_raw_ace (oldest first)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f"no staged raw ACE files match {pattern}")
    parts = [load_raw_ace(path) for path in files]
    raw = pd.concat(parts)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return raw


def vintage_record(raw: pd.DataFrame) -> dict:
    """Report the vintage_verified=0 limitation; NEVER mark verified."""
    col = "point_in_time_vintage_verified"
    unique = sorted(pd.to_numeric(raw[col], errors="coerce").dropna().unique().tolist()) \
        if col in raw.columns else []
    return {"column": col, "unique_values": unique, "verified": False,
            "limitation": VINTAGE_LIMITATION}


def _clean(raw: pd.DataFrame, column: str, minimum: float | None = None) -> pd.Series:
    series = pd.to_numeric(raw[column], errors="coerce").astype(float)
    if minimum is not None:
        series = series.where(series > minimum if minimum == 0 else series >= minimum)
        if minimum == 0:
            series = series.where(series > 0)
    return series


def build_physics_frame(raw: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Causal ACE physics features indexed by origin.

    Channels: raw density [cm^-3], raw temperature [K], raw |B| (ace_bt_nt),
    derived dynamic pressure [nPa]. Per channel: lag0, 1h/6h causal
    gradients, 24h mean, 24h missingness count, age since last valid.
    Plus swepam/mag status + 24h bad-status counts + ages since nominal.
    """
    raw = raw.sort_index()
    origins = pd.DatetimeIndex(utc(origins)).sort_values().unique()
    dens = _clean(raw, "ace_density_cm3", minimum=0)
    temp = _clean(raw, "ace_temperature_k", minimum=0)
    bt = _clean(raw, "ace_bt_nt", minimum=0)
    vel = _clean(raw, "ace_speed_kms", minimum=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdyn = pd.Series(PDYN_FACTOR * dens.to_numpy() * vel.to_numpy() ** 2,
                         index=raw.index)
        pdyn = pdyn.where(np.isfinite(pdyn) & (dens.to_numpy() > 0)
                          & (vel.to_numpy() > 0))
    channels = {"dens": dens, "temp": temp, "bt": bt, "pdyn": pdyn}
    out: dict[str, np.ndarray] = {}
    for name, series in channels.items():
        vals = _window(series, origins, 25)
        out[f"phys_{name}_lag0"] = vals[:, -1]
        out[f"phys_{name}_diff_1"] = vals[:, -1] - vals[:, -2]
        out[f"phys_{name}_diff_6"] = vals[:, -1] - vals[:, -7]
        out[f"phys_{name}_mean24"] = _nanstat(np.nanmean, vals[:, -24:])
        out[f"phys_{name}_missing24"] = np.isnan(vals[:, -24:]).sum(axis=1).astype(float)
        out[f"phys_{name}_age_hours"] = _since_valid(np.isfinite(vals))
    for key, column in (("sw", "ace_swepam_status"), ("mag", "ace_mag_status")):
        status = pd.to_numeric(raw[column], errors="coerce") if column in raw.columns \
            else pd.Series(np.nan, index=raw.index)
        sval = _window(status, origins, 25)
        out[f"phys_{key}_status_lag0"] = sval[:, -1]
        out[f"phys_{key}_bad_24"] = ((sval[:, -24:] != 0)
                                    | np.isnan(sval[:, -24:])).sum(axis=1).astype(float)
        out[f"phys_{key}_age_hours"] = _since_valid(sval == 0)
    frame = pd.DataFrame(out, index=origins)
    frame.index.name = "origin_last_input_utc"
    return _finite_or_nan(frame)


# ---------------------------------------------------------------------------
# CH mask-quality features (causal compact-cache summaries)
# ---------------------------------------------------------------------------

def _ch_series(ch: pd.DataFrame, column: str) -> pd.Series:
    if column not in ch.columns:
        return pd.Series(np.nan, index=ch.index, dtype=float)
    return pd.to_numeric(ch[column], errors="coerce").astype(float)


def build_ch_quality_frame(ch: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Causal CH mask-reliability summaries indexed by origin.

    Threshold agreement (t035/t045/t055 compact area columns), per-threshold
    valid_fraction / count / max-area quality, equatorial-share geometry,
    and area/count/coverage continuity. Point-in-time values are causal
    (source timestamps <= origin); changes compare causal lags only.
    """
    ch = ch.sort_index()
    ch.index = pd.DatetimeIndex(utc(ch.index))
    origins = pd.DatetimeIndex(utc(origins)).sort_values().unique()
    area = {tag: _window(_ch_series(ch, f"ch_{tag}_area"), origins, 73)
            for tag in CH_Q_TAGS}
    out: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        a035, a045, a055 = area["t035"][:, -1], area["t045"][:, -1], area["t055"][:, -1]
        out["chq_agree_045_035"] = a045 / a035
        out["chq_agree_055_045"] = a055 / a045
        out["chq_agree_055_035"] = a055 / a035
        out["chq_spread_045_035"] = a045 - a035
        out["chq_spread_055_045"] = a055 - a045
        out["chq_spread_055_035"] = a055 - a035
    rank_ok = ((a035 <= a045) & (a045 <= a055)).astype(float)
    rank_ok[~(np.isfinite(a035) & np.isfinite(a045) & np.isfinite(a055))] = np.nan
    out["chq_rank_ok"] = rank_ok
    for tag in CH_Q_TAGS:
        vals = area[tag]
        out[f"chq_{tag}_area_absdiff_1"] = np.abs(vals[:, -1] - vals[:, -2])
        out[f"chq_{tag}_area_absdiff_24"] = np.abs(vals[:, -1] - vals[:, -25])
        vf = _window(_ch_series(ch, f"ch_{tag}_valid_fraction"), origins, 25)
        out[f"chq_{tag}_vf_lag0"] = vf[:, -1]
        out[f"chq_{tag}_vf_min24"] = _nanstat(np.nanmin, vf[:, -24:])
        cnt = _window(_ch_series(ch, f"ch_{tag}_count"), origins, 25)
        out[f"chq_{tag}_count_lag0"] = cnt[:, -1]
        out[f"chq_{tag}_count_jump24"] = np.abs(cnt[:, -1] - cnt[:, -25])
        out[f"chq_{tag}_maxarea_lag0"] = _window(
            _ch_series(ch, f"ch_{tag}_max_area"), origins, 1)[:, -1]
        with np.errstate(invalid="ignore", divide="ignore"):
            eq = _ch_series(ch, f"ch_{tag}_equatorial_area").reindex(
                origins).to_numpy(dtype=float)
            ar = vals[:, -1]
            share = np.where((np.isfinite(eq) & np.isfinite(ar)) & (ar > 0), eq / ar, np.nan)
        out[f"chq_{tag}_eq_share_lag0"] = share
    frame = pd.DataFrame(out, index=origins)
    frame.index.name = "origin_last_input_utc"
    return _finite_or_nan(frame)


def header_conventions_report() -> dict:
    """Functionally probe real header conventions; assert no WCS error claims."""
    from experiments.xgboost_suvi_fusion import coronal_holes as _ch
    header = {"CRPIX1": 5.5, "CRPIX2": 5.5, "RSUN": 4.0, "CROTA": 0.0,
              "SOLAR_B0": 0.0, "PC1_1": 1.0, "PC1_2": 0.0,
              "PC2_1": 0.0, "PC2_2": 1.0}
    disk, lat, lon, _area = _ch.geometry((10, 10), header)
    assert disk.shape == (10, 10) and lat.shape == (10, 10) and lon.shape == (10, 10)
    assert bool(disk.any()) and bool((~disk).any())
    masks = sorted(str(p) for p in Path("store/ch-v1/masks").iterdir()) \
        if Path("store/ch-v1/masks").is_dir() else []
    return {
        "consumed_keys": ["CRPIX1", "CRPIX2", "RSUN", "CROTA", "SOLAR_B0",
                          "PC1_1", "PC1_2", "PC2_1", "PC2_2"],
        "probe": "synthetic-header geometry() shape/disk/lat/lon/area contract OK",
        "masks_dir_listing": [Path(p).name for p in masks],
        "masks_are_rendered_overlays_only": all(
            p.endswith((".svg", ".png")) for p in masks),
        "wcs_sign_scale_error_asserted": False,
        "note": HEADER_NOTE,
        "delimitation": MASK_DELIMITATION,
    }


# ---------------------------------------------------------------------------
# Arm assembly + experiment driver
# ---------------------------------------------------------------------------

def arm_columns(base: pd.DataFrame, physics: pd.DataFrame,
                quality: pd.DataFrame, arm: str) -> list[str]:
    """Column sets per arm; speed_only is the matched control."""
    ace_cols = [c for c in base.columns if c.startswith("ace_")]
    if arm == "speed_only":
        return ace_cols
    if arm == "physical_ace":
        return ace_cols + list(physics.columns)
    if arm == "physical_ace_ch_quality":
        ch_cols = [c for c in base.columns if c.startswith("ch_")]
        return ace_cols + list(physics.columns) + ch_cols + list(quality.columns)
    raise ValueError(f"unknown arm {arm}")


def _free_gpu() -> int | None:
    """Non-blocking probe of the two lease files; None when both are busy."""
    lease_dir = Path("store/ch-breakthrough-v2/gpu-leases")
    lease_dir.mkdir(parents=True, exist_ok=True)
    for gpu in (0, 1):
        try:
            handle = open(lease_dir / f"gpu{gpu}.lock", "w")
        except OSError:
            continue
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            continue
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
        return gpu
    return None


def resolve_device(preferred: str) -> tuple[str, str]:
    """Pick the training device; record what is actually used.

    ``auto`` takes a GPU lease via ``common.acquire_gpu_lease`` when a card
    is free (fail-closed CUDA probe) and falls back to CPU otherwise.
    Returns (device_for_xgb, device_record).
    """
    if preferred != "auto":
        ensure_device(preferred)
        return preferred, preferred
    if _free_gpu() is None:
        return "cpu", "cpu (leases busy)"
    with acquire_gpu_lease() as gpu:
        device = f"cuda:{gpu}"
        try:
            ensure_device(device)
        except Exception:
            return "cpu", "cpu (cuda probe failed)"
        return device, device


def run_split(split: str, *, ace_paths, ch_path, raw_pattern, outdir: Path,
              origin_step="24h", max_origins=120, max_train_origins=500,
              device="cpu", seed=SEED) -> dict:
    """Run all three arms on one protocol split; returns per-arm MSEs."""
    cfg = SPLITS[split]
    cutoff = utc(cfg["cutoff"])
    ace = load_ace_hourly(ace_paths)
    ch = load_ch_hourly(ch_path)
    raw = load_staged_raw_ace(raw_pattern)
    vintage = vintage_record(raw)
    speed = as_speed_series(ace)
    base = build_features(ace, ch, None)
    origins_all = base.index.intersection(
        pd.DatetimeIndex(utc(raw.index)).floor("h")).sort_values()
    physics = build_physics_frame(raw, origins_all)
    quality = build_ch_quality_frame(ch, origins_all)
    frames = {"base": base.reindex(origins_all),
              "physics": physics, "quality": quality}
    observed = observed_mask(speed, speed.index)
    raw_files = sorted(glob.glob(raw_pattern))
    raw_parts = [file_identity(path) for path in raw_files]
    raw_digest = hashlib.sha256(
        "".join(part["sha256"] for part in raw_parts).encode()).hexdigest()
    identities = {"ace": file_identity(ace_paths[0]),
                  "ch_hourly": file_identity(ch_path),
                  "raw_ace_years": {"pattern": raw_pattern,
                                    "files": raw_files,
                                    "file_identities": raw_parts,
                                    "sha256": raw_digest,
                                    "bytes": sum(part["bytes"] for part in raw_parts)}}
    header_report = header_conventions_report()

    # Chronological OOF: train pool strictly before the cutoff (labels
    # complete at/before it); eval grid per protocol window.
    train_pool, _ = purged_split(frames["base"].index, cutoff)
    grid = pd.date_range(utc(cfg["start"]), utc(cfg["end"]), freq="h")
    if split == "official":
        # Pinned official folds: every hourly origin in range (matches the
        # 2833 ch-v1 official origins); training still ends at the cutoff.
        eval_origins = grid.intersection(frames["base"].index)
        sampling = {"step": "hourly", "requested": len(grid),
                    "kept": len(eval_origins), "seed": seed,
                    "note": "all supplied origins, no subsampling"}
    else:
        purge_after = cutoff + pd.Timedelta(hours=HORIZON_HOURS)
        grid = grid[(grid > purge_after)
                    & (grid + pd.Timedelta(hours=HORIZON_HOURS) <= utc(cfg["end"]))]
        grid = grid.intersection(frames["base"].index)
        # NOTE: frozen common.subsample_origins raises TypeError on tz-aware
        # origins when its random max_origins branch triggers, so screening
        # caps stay above the daily-floor count (no random draw taken).
        eval_origins, sampling = subsample_origins(
            grid, step=origin_step, max_origins=max_origins, seed=seed)
    if len(eval_origins) == 0:
        raise ValueError(f"{split}: no eval origins after purge/subsampling")
    if len(train_pool) == 0:
        raise ValueError(f"{split}: empty train pool before cutoff")

    outdir.mkdir(parents=True, exist_ok=True)
    result = {"split": split, "cutoff": cutoff.isoformat(),
              "n_train_pool": len(train_pool),
              "n_eval_origins": len(eval_origins),
              "sampling": sampling, "device": device,
              "vintage": vintage, "arms": {}}
    for arm in ARMS:
        cols = arm_columns(frames["base"], frames["physics"], frames["quality"], arm)
        # Assemble arm features aligned to shared origins; map any inf from
        # foundation ratio columns (e.g. zero-area agreement) to NaN.
        arm_features = _finite_or_nan(pd.concat(
            [frames["base"][[c for c in cols if c in frames["base"].columns]],
             frames["physics"][[c for c in cols if c in frames["physics"].columns]],
             frames["quality"][[c for c in cols if c in frames["quality"].columns]]],
            axis=1)[cols])
        fore = PooledResidualForecaster(params=dict(PARAMS), device=device, seed=seed)
        fore.fit(arm_features, speed, cutoff, max_origins=max_train_origins)
        preds, secs = timed_predict(
            lambda outs, f=fore, af=arm_features: f.predict(af, speed, outs),
            eval_origins)
        name = f"{split}-{arm}"
        preds.to_parquet(outdir / f"{name}.parquet", index=False)
        metrics = score_forecast(preds, speed, observed, origins=eval_origins)
        (outdir / f"{name}.metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n")
        extra = {"origins": len(eval_origins), "mse": metrics["mse"],
                 "mse_observed": metrics["mse_observed"],
                 "imputed_fraction": metrics.get("imputed_fraction"),
                 "unique_observed_peak_timestamps":
                     metrics.get("unique_observed_peak_timestamps"),
                 "overlap_block_ci95_7day": metrics.get("overlap_block_ci95_7day"),
                 "split": split, "arm": arm,
                 "eval_window": [cfg["start"], cfg["end"]],
                 "n_train_pool": len(train_pool),
                 "n_short_history_skipped": fore.n_short_history_skipped,
                 "vintage": vintage, "header_conventions": header_report}
        if split == "official":
            extra["label"] = ("exploratory: evaluation period was previously "
                              "inspected during ch-v1; single pass only")
        write_manifest(outdir / f"{name}.manifest.json", mode=name,
                       train_cutoff=cutoff, params=fore.params,
                       tree_counts={"n_estimators": PARAMS["n_estimators"]},
                       columns=list(fore.columns),
                       input_identities=identities, seed=seed,
                       inference_seconds_per_origin=secs, sampling=sampling,
                       extra=extra, caller_file=__file__,
                       actual_device=fore.actual_device,
                       prediction_path=outdir / f"{name}.parquet")
        result["arms"][arm] = {"mse": metrics["mse"],
                               "mse_observed": metrics["mse_observed"],
                               "n_columns": len(fore.columns),
                               "secs_per_origin": secs}
    (outdir / f"{split}.summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", nargs="+", default=["dev-A"],
                   choices=[*SPLITS, "all"])
    p.add_argument("--ace", nargs="+", default=["store/ch-v1/ace.parquet"])
    p.add_argument("--ch-hourly", default="store/ch-v1/ch-hourly.parquet")
    p.add_argument("--raw-pattern",
                   default="/home/t-lab01/.local/state/sundb/stage/ace_solar/year=*/part.parquet")
    p.add_argument("--origin-step", default="24h")
    p.add_argument("--max-origins", type=int, default=120)
    p.add_argument("--max-train-origins", type=int, default=500)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--output-dir", default="store/ch-breakthrough-v2/physical_quality")
    p.add_argument("--threads", type=int, default=N_THREADS)
    return p


def run(args) -> dict:
    device, device_record = resolve_device(args.device)
    splits = list(SPLITS) if args.splits == ["all"] else args.splits
    outdir = Path(args.output_dir)
    summary = {"device": device_record, "params": PARAMS, "splits": {}}
    for split in splits:
        res = run_split(split, ace_paths=args.ace, ch_path=args.ch_hourly,
                        raw_pattern=args.raw_pattern, outdir=outdir,
                        origin_step=args.origin_step,
                        max_origins=args.max_origins,
                        max_train_origins=args.max_train_origins,
                        device=device, seed=args.seed)
        summary["splits"][split] = res
    (outdir / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    return summary


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
