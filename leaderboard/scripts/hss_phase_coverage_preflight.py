"""Data-only phase coverage preflight; no fits, no backfill, no performance claims."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LEADERBOARD = Path(__file__).resolve().parents[1]
if str(LEADERBOARD) not in sys.path:
    sys.path.insert(0, str(LEADERBOARD))

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

HOLE_ISO = "2024-06-17T18:00:00+00:00"
DEFAULT_RUN_DIR = LEADERBOARD / "store/mse20-four-regimes/run-20261003/hss-phase"


def _utc(values):
    import pandas as pd

    return pd.DatetimeIndex(pd.to_datetime(values, utc=True))


def frame_coverage(frames):
    """Per-frame index coverage plus NaN/inf row counts; no filling, no drops."""
    import numpy as np
    import pandas as pd

    out = {}
    for name, frame in frames.items():
        try:
            if isinstance(frame, pd.Series):
                index = _utc(frame.index)
                vals = frame.to_numpy()
                nan_rows = int(pd.isna(frame).sum())
                with np.errstate(all="ignore"):
                    flat = pd.to_numeric(vals, errors="coerce")
                    inf_rows = int(np.isinf(np.asarray(flat, dtype=float)).sum())
            elif isinstance(frame, pd.DataFrame):
                if isinstance(frame.index, pd.DatetimeIndex):
                    index = _utc(frame.index)
                elif "timestamp_utc" in frame.columns:
                    index = _utc(frame["timestamp_utc"])
                else:
                    out[name] = {"index_type": "non_datetime", "n_rows": int(len(frame))}
                    continue
                nan_rows = int(frame.isna().any(axis=1).sum()) if len(frame.columns) else 0
                num = frame.select_dtypes(include=[np.number])
                if len(num.columns):
                    with np.errstate(all="ignore"):
                        inf_rows = int(np.isinf(num.to_numpy(dtype=float)).any(axis=1).sum())
                else:
                    inf_rows = 0
            else:
                out[name] = {"index_type": type(frame).__name__, "n_rows": int(len(frame))}
                continue
        except Exception as error:
            out[name] = {"error": f"{type(error).__name__}: {error}"}
            continue
        entry = {
            "n_rows": int(len(index)),
            "index_min": index.min().isoformat() if len(index) else None,
            "index_max": index.max().isoformat() if len(index) else None,
            "n_nan_rows": nan_rows,
            "n_inf_rows": inf_rows,
        }
        if name in ("speed", "base") and len(index):
            expected = pd.date_range(index.min(), index.max(), freq="h")
            missing = expected.difference(index)
            entry["n_missing_hourly"] = int(len(missing))
            entry["first_missing_hourly"] = [t.isoformat() for t in missing[:10]]
        out[name] = entry
    return out


def _base_speed_indexes(frames):
    base = frames.get("base")
    speed = frames.get("speed")
    base_index = _utc(base.index) if base is not None else _utc([])
    speed_index = _utc(speed.index) if speed is not None else _utc([])
    return base_index.sort_values(), speed_index.sort_values()


def deployment_coverage(frames, folds, fold_origins_fn):
    """Every issuance kept; missing ones carry previous-base provenance and lag."""
    base_index, speed_index = _base_speed_indexes(frames)
    base_set, speed_set = set(base_index), set(speed_index)
    per_fold, missing = {}, []
    total = 0
    for fold in folds:
        origins = _utc(fold_origins_fn(fold))
        total += len(origins)
        n_missing = 0
        for origin in origins:
            in_base, in_speed = origin in base_set, origin in speed_set
            if in_base:
                continue
            n_missing += 1
            prev = base_index[base_index <= origin]
            prev_utc = prev.max().isoformat() if len(prev) else None
            lag = (origin - prev.max()).total_seconds() / 3600 if len(prev) else None
            missing.append(
                {
                    "issuance_utc": origin.isoformat(),
                    "fold_id": fold.get("id"),
                    "in_speed": bool(in_speed),
                    "in_base": False,
                    "prev_base_utc": prev_utc,
                    "lag_hours": lag,
                }
            )
        per_fold[fold.get("id")] = {"n_origins": int(len(origins)), "n_missing_base": int(n_missing)}
    return {"n_issuances": int(total), "n_missing": int(len(missing)), "per_fold": per_fold, "missing": missing}


def training_oof_audit(frames, folds, training_origins_fn, calibration_fn=None, config=None):
    """Origin-pool eligibility only; never fits, never touches torch models."""
    import pandas as pd

    out = {}
    for fold in folds:
        cutoff = pd.Timestamp(fold["start_utc"]) - pd.Timedelta(hours=1)
        bounded = {}
        for name, frame in frames.items():
            try:
                if hasattr(frame, "loc") and isinstance(getattr(frame, "index", None), pd.DatetimeIndex):
                    bounded[name] = frame.loc[frame.index <= cutoff]
                else:
                    bounded[name] = frame
            except Exception:
                bounded[name] = frame
        try:
            pool = _utc(training_origins_fn(bounded, cutoff, config) if config is not None else training_origins_fn(bounded, cutoff))
            entry = {"cutoff": cutoff.isoformat(), "n_pool": int(len(pool)),
                     "pool_min": pool.min().isoformat(), "pool_max": pool.max().isoformat()}
            if calibration_fn is not None:
                _, assignment = calibration_fn(pool, cutoff)
                entry["oof_blocks"] = {str(int(k)): int((assignment == k).sum()) for k in sorted(set(assignment.tolist()))}
            entry["ok"] = True
        except Exception as error:
            entry = {"cutoff": cutoff.isoformat(), "ok": False, "error": f"{type(error).__name__}: {error}"}
        out[fold.get("id")] = entry
    return out


def analyze_hole(frames, folds, fold_origins_fn, hole_iso=HOLE_ISO):
    """Explicit provenance for the 2024-06-17T18Z issuance hole."""
    import pandas as pd

    hole = pd.Timestamp(hole_iso)
    base_index, speed_index = _base_speed_indexes(frames)
    base_set, speed_set = set(base_index), set(speed_index)
    containing, on_grid = None, False
    for fold in folds:
        origins = set(_utc(fold_origins_fn(fold)))
        if hole in origins:
            containing, on_grid = fold.get("id"), True
            break
        if pd.Timestamp(fold["start_utc"]) <= hole < pd.Timestamp(fold["end_utc"]):
            containing = fold.get("id")
    prev = base_index[base_index <= hole]
    nxt = base_index[base_index > hole]
    window = pd.date_range(hole - pd.Timedelta(hours=24), hole + pd.Timedelta(hours=24), freq="h")
    gap_missing = [t.isoformat() for t in window if t not in base_set]
    return {
        "issuance_utc": hole.isoformat(),
        "on_issuance_grid": bool(on_grid),
        "containing_fold": containing,
        "in_speed": bool(hole in speed_set),
        "in_base": bool(hole in base_set),
        "prev_base_utc": prev.max().isoformat() if len(prev) else None,
        "lag_hours": (hole - prev.max()).total_seconds() / 3600 if len(prev) else None,
        "next_base_utc": nxt.min().isoformat() if len(nxt) else None,
        "window_missing_base_48h": gap_missing,
        "n_window_missing_base_48h": len(gap_missing),
    }


def _refuse_inside_run_dir(run_dir, output):
    run = Path(run_dir).resolve()
    out = Path(output).resolve()
    try:
        if out == run or out.is_relative_to(run):
            raise ValueError("output must be outside the frozen run directory")
    except AttributeError:
        if out == run or str(out).startswith(str(run) + "/"):
            raise ValueError("output must be outside the frozen run directory")
    return run, out


def run_preflight(run_dir, output, synthetic=False):
    import pandas as pd

    run_dir = Path(run_dir)
    if synthetic:
        idx = pd.date_range("2024-06-15T00:00:00Z", periods=144, freq="h", tz="UTC")
        speed = pd.Series(range(len(idx)), index=idx)
        base = pd.DataFrame({"f": range(len(idx))}, index=idx).drop(pd.Timestamp(HOLE_ISO))
        frames = {"speed": speed, "base": base}
        folds = [{"id": "synth", "start_utc": idx[48].isoformat(), "end_utc": idx[-1].isoformat()}]
        origins_fn = lambda fold: pd.date_range(fold["start_utc"], pd.Timestamp(fold["end_utc"]) - pd.Timedelta(hours=73), freq="6h")  # noqa: E731
        registration = {"synthetic": True, "n_folds": 1, "fold_ids": ["synth"]}
        cutoff = "2026-05-31T23:00:00Z"
        oof = training_oof_audit(frames, folds, lambda fr, co, *a: _utc(fr["base"].index), None, None)
    else:
        from experiments.mse20_hss_phase import dev
        from experiments.mse20_hss_phase import protocol

        registration_record = protocol.verify_registration(run_dir)
        folds = registration_record["folds"]
        if len(folds) != 14:
            raise ValueError("registration must hold exactly 14 folds")
        frames = protocol.load_frames(protocol.CUTOFF, run_dir)
        origins_fn = dev.fold_origins
        registration = {
            "registered_utc": registration_record.get("registered_utc"),
            "n_folds": len(folds),
            "fold_ids": [f["id"] for f in folds],
        }
        cutoff = str(protocol.CUTOFF)
        from experiments.mse20_hss_phase import model as phase_model

        oof = training_oof_audit(frames, folds, phase_model.training_origins, phase_model.calibration_assignments,
                                 registration_record["configs"][0])
    deploy = deployment_coverage(frames, folds, origins_fn)
    report = {
        "no_model_fit": True,
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "preofficial_cutoff": cutoff,
        "run_dir": str(Path(run_dir).resolve()),
        "registration_reference": registration,
        "fold_counts": deploy["per_fold"],
        "n_issuances": deploy["n_issuances"],
        "n_missing": deploy["n_missing"],
        "missing_issuances": deploy["missing"],
        "frame_coverage": frame_coverage(frames),
        "hole_2024_06_17T18Z": analyze_hole(frames, folds, origins_fn),
        "training_oof": oof,
        "notes": "data-only audit; no backfill, no dropped issuances, no performance claims",
    }
    _, out = _refuse_inside_run_dir(run_dir, output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as handle:
        json.dump(report, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    print(f"folds={len(folds)} issuances={deploy['n_issuances']} missing={deploy['n_missing']}")
    hole = report["hole_2024_06_17T18Z"]
    print(f"hole {hole['issuance_utc']} in_base={hole['in_base']} fold={hole['containing_fold']}")
    print(f"prev_base={hole['prev_base_utc']} lag_hours={hole['lag_hours']}")
    print(f"window_missing_48h={hole['n_window_missing_base_48h']}")
    print(f"frames={sorted(report['frame_coverage'])}")
    print(f"no_model_fit=true output={out}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic-test", action="store_true")
    args = parser.parse_args(argv)
    run_preflight(args.run_dir, args.output, synthetic=args.synthetic_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
