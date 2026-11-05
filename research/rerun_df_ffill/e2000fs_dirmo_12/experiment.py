"""The original E2000 block/recursive strategies on the new, causally filled df.

Source: analysis/A0014/strategies.py and the frozen e2000fs strategy submissions.
The model recipe, feature order, recursive inputs, four-fold/72h OOF correction,
and final [1,3000] clipping are retained. Only the dataset, official period masks,
and missing-target loss masking change. No evaluation labels enter fitting.

Run with --experiment e2000fs_dirmo_24 --data input/df_ffill.parquet --out DIR.
--raw-data is the *same new input df before ffill*, used solely to recover the
original img_stale = no image or image age > 6h feature. No raw images are read.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import subprocess
import time

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import xgboost as xgb

H, LAG, FOLDS, PURGE_H = 72, 24, 4, 72
CUTOFFS = {"F1": "2025-12-29", "F2": "2026-03-29", "F3": "2026-06-29"}
QUERY_COUNTS = {"F1": 2089, "F2": 2113, "F3": 2113}
PARAMS = dict(max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
              min_child_weight=20, reg_lambda=5, seed=42, objective="reg:squarederror",
              tree_method="hist", device="cpu", verbosity=0, disable_default_eval_metric=1)
CONSTANT = ['v_lag0', 'v_lag1', 'v_lag3', 'v_lag6', 'v_lag12', 'v_lag24', 'v_lag48', 'v_lag72', 'v_lag120', 'v_m6', 'v_sd6', 'v_m24', 'v_sd24', 'v_m72', 'v_sd72', 'ace_speed_kms_last', 'ace_speed_kms_m6', 'ace_speed_kms_m24', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_density_cm3_m24', 'ace_temperature_k_last', 'ace_temperature_k_m6', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_bt_nt_m24', 'ace_bz_gsm_nt_last', 'ace_bz_gsm_nt_m6', 'ace_bz_gsm_nt_m24', 'ace_by_gsm_nt_last', 'ace_by_gsm_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_e_channel1_last', 'ace_epam_e_channel1_m6', 'ace_epam_e_channel1_m24', 'ace_epam_e_channel2_last', 'ace_epam_e_channel2_m6', 'ace_epam_e_channel2_m24', 'ace_epam_p_channel1_last', 'ace_epam_p_channel1_m6', 'ace_epam_p_channel1_m24', 'ace_epam_p_channel3_last', 'ace_epam_p_channel3_m6', 'ace_epam_p_channel3_m24', 'ace_epam_p_channel5_last', 'ace_epam_p_channel5_m6', 'ace_epam_p_channel5_m24', 'ace_sis_gt10_flux_last', 'ace_sis_gt10_flux_m6', 'ace_sis_gt10_flux_m24', 'speed_d24', 'epam_p1_d6', 'ch_ch_t035_area', 'ch_ch_t035_max_area', 'ch_ch_t035_top1_area', 'ch_ch_t035_top2_area', 'ch_ch_t035_top3_area', 'ch_ch_t035_meridian15_area', 'ch_ch_t035_meridian15_equatorial_area', 'ch_ch_t035_meridian30_area', 'ch_ch_t035_meridian30_equatorial_area', 'ch_ch_t035_equatorial_area', 'ch_ch_t045_area', 'ch_ch_t045_max_area', 'ch_ch_t045_top1_area', 'ch_ch_t045_top2_area', 'ch_ch_t045_top3_area', 'ch_ch_t045_meridian15_area', 'ch_ch_t045_meridian15_equatorial_area', 'ch_ch_t045_meridian30_area', 'ch_ch_t045_meridian30_equatorial_area', 'ch_ch_t045_equatorial_area', 'ch_ch_t055_area', 'ch_ch_t055_max_area', 'ch_ch_t055_top1_area', 'ch_ch_t055_top2_area', 'ch_ch_t055_top3_area', 'ch_ch_t055_meridian15_area', 'ch_ch_t055_meridian15_equatorial_area', 'ch_ch_t055_meridian30_area', 'ch_ch_t055_meridian30_equatorial_area', 'ch_ch_t055_equatorial_area', 'img_age_h', 'img_stale', 'enlil_next_arr_h', 'enlil_since_arr_h', 'enlil_n_pending', 'enlil_kp', 'enlil_glancing']
VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v350_dark', 'strip_v350_cover', 'strip_v450_logmed', 'strip_v450_dark', 'strip_v450_cover', 'strip_v550_logmed', 'strip_v550_dark', 'strip_v550_cover', 'strip_v650_logmed', 'strip_v650_dark', 'strip_v650_cover', 'strip_v750_logmed', 'strip_v750_dark', 'strip_v750_cover', 'enlil_dt_target']

STRATEGIES = {
    "e2000fs_recursive": ("RecMO", 1), "e2000fs_dirrec": ("DirRecMO", 1),
    "e2000fs_mimo": ("DirMO", 72),
    **{f"e2000fs_{kind.lower()}_{s}": (kind, s)
       for kind in ("RecMO", "DirMO", "DirRecMO") for s in (6, 12, 24)},
    "e2000fs_rectify": (("RecMO", 1), ("DirMO", 1)),
    "e2000fs_rectifymo_24": (("RecMO", 24), ("DirMO", 24)),
    "e2000fs_stratify_dirrecmo_dirrecmo_24": (("DirRecMO", 24), ("DirRecMO", 24)),
}


def mapped(name):
    return name.replace("ch_ch_", "suvi_", 1) if name.startswith("ch_ch_") else name


def past_window(series, origins):
    """24 causal input snapshots ending at origin (last snapshot is O-1h ACE)."""
    origins = pd.DatetimeIndex(origins)
    times = origins.asi8[:, None] + np.arange(-LAG + 1, 1)[None, :] * 3_600_000_000_000
    return series.reindex(pd.DatetimeIndex(times.ravel(), tz="UTC")).to_numpy(np.float32).reshape(-1, LAG)


def stale_series(raw):
    """Reconstruct original no-image/age>6 rule from raw table image observation ends."""
    raw = raw.sort_values("origin", kind="stable")
    duplicates = raw[raw.duplicated("origin", keep=False)]
    if len(duplicates) and (duplicates.groupby("origin")["img_age_h"].nunique(dropna=False) > 1).any():
        raise ValueError("train/query image timestamps disagree")
    raw = raw.drop_duplicates("origin").set_index("origin")
    ends = pd.Series(raw.index, index=raw.index) - pd.to_timedelta(raw.img_age_h, unit="h")
    if (ends.dropna() > ends.dropna().index).any():
        raise ValueError("image observation lies after its input origin")
    last_end = ends.ffill()
    age = (pd.Series(raw.index, index=raw.index) - last_end).dt.total_seconds() / 3600
    return (last_end.isna() | age.gt(6.0)).astype(np.float32)


@dataclass
class Inputs:
    ctx: np.ndarray
    E: np.ndarray
    past: np.ndarray

    def __len__(self):
        return len(self.ctx)

    def take(self, indices):
        return Inputs(self.ctx[indices], self.E[indices], self.past[indices])


def blocks(sigma):
    if sigma <= 0 or H % sigma:
        raise ValueError("block size must divide 72")
    return [(s, s + sigma - 1) for s in range(1, H + 1, sigma)]


def training_trajectory(inp, y):
    """Fill teacher-forcing *inputs* causally; targets passed to the loss stay missing."""
    return pd.DataFrame(np.hstack([inp.past, y])).ffill(axis=1).to_numpy(np.float32)


def design(kind, inp, trajectory, s, e):
    exogenous = inp.E[:, s - 1:e].reshape(len(inp), -1)
    if kind == "RecMO":
        return np.hstack([inp.ctx, trajectory[:, s - 1:s - 1 + LAG], exogenous,
                          np.full((len(inp), 1), s, np.float32)]).astype(np.float32)
    if kind == "DirMO":
        return np.hstack([inp.ctx, inp.past, exogenous]).astype(np.float32)
    if kind == "DirRecMO":
        return np.hstack([inp.ctx, inp.past, trajectory[:, LAG:LAG + s - 1], exogenous]).astype(np.float32)
    raise ValueError(kind)


def masked_l2(observed):
    observed = np.asarray(observed, np.float32)

    def objective(pred, matrix):
        p = np.asarray(pred).reshape(observed.shape)
        y = matrix.get_label().reshape(observed.shape)
        return ((p - y) * observed).astype(np.float32), observed
    return objective


def fit_booster(X, Y, rounds, threads, output, tag, feature_names):
    """The frozen squared-error learner, with zero loss mass on absent labels."""
    Y = np.asarray(Y, np.float32).reshape(len(Y), -1)
    keep = np.isfinite(Y).any(axis=1)
    if not keep.any():
        raise ValueError(f"{tag}: no observed training targets")
    X, Y = X[keep], Y[keep]
    observed = np.isfinite(Y)
    if not observed.any(axis=0).all():
        raise ValueError(f"{tag}: a fitted output has no observed target")
    if np.isinf(X).any():
        raise ValueError(f"{tag}: infinite feature value")
    if output is not None and len(feature_names) != X.shape[1]:
        raise ValueError(f"{tag}: feature names do not match the design width")
    params = dict(PARAMS, nthread=threads, base_score=float(np.nanmean(Y)))
    if Y.shape[1] > 1:
        params["multi_strategy"] = "multi_output_tree"
    matrix = xgb.DMatrix(X, label=np.where(observed, Y, 0.0), nthread=threads)
    trace = {}

    def metric(pred, dm):
        error = np.asarray(pred).reshape(Y.shape) - dm.get_label().reshape(Y.shape)
        return "observed_rmse", float(np.sqrt(np.square(error[observed]).mean()))

    start = time.perf_counter()
    model = xgb.train(params, matrix, num_boost_round=rounds, obj=masked_l2(observed),
                      custom_metric=metric, evals=[(matrix, "train")], evals_result=trace, verbose_eval=False)
    duration = time.perf_counter() - start
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        model.save_model(output / f"{tag}.ubj")
        pd.DataFrame({"round": np.arange(1, rounds + 1),
                      "train_observed_rmse": trace["train"]["observed_rmse"]}).to_csv(output / f"{tag}_loss.csv", index=False)
        report = {"tag": tag, "train_rows": len(Y), "removed_all_missing_rows": int((~keep).sum()),
                  "observed_targets": int(observed.sum()), "outputs": Y.shape[1],
                  "feature_count": X.shape[1], "feature_names": feature_names,
                  "parameters": params, "rounds": rounds, "seconds": duration}
        (output / f"{tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"fit": tag, "rows": len(Y), "outputs": Y.shape[1], "seconds": duration}), flush=True)
    return model


def block_feature_names(kind, s, e, context_names=None, varying_names=None):
    context_names = list(CONSTANT if context_names is None else context_names)
    varying_names = VARYING if varying_names is None else varying_names
    past = [f"speed_input_lag{k}" for k in range(LAG - 1, -1, -1)]
    exogenous = [f"{name}@h{h}" for h in range(s, e + 1) for name in varying_names]
    if kind == "RecMO":
        return context_names + [f"trajectory_lag{k}" for k in range(LAG - 1, -1, -1)] + exogenous + ["block_start_h"]
    if kind == "DirMO":
        return context_names + past + exogenous
    return context_names + past + [f"earlier_target_h{h}" for h in range(1, s)] + exogenous


class Block:
    def __init__(self, kind, sigma, rounds=900, threads=1, output=None, tag="fit", context_names=None, varying_names=None):
        self.kind, self.sigma, self.rounds, self.threads = kind, sigma, rounds, threads
        self.output, self.tag = output, tag
        self.context_names, self.varying_names = context_names, varying_names
        self.models = {}

    def fit(self, inp, Y):
        T = training_trajectory(inp, Y)
        spans = blocks(self.sigma)
        if self.kind == "RecMO":
            first = design(self.kind, inp, T, *spans[0])
            X = np.empty((len(inp) * len(spans), first.shape[1]), dtype=np.float32)
            y = np.empty((len(inp) * len(spans), self.sigma), dtype=np.float32)
            for b, (s, e) in enumerate(spans):
                sl = slice(b * len(inp), (b + 1) * len(inp))
                X[sl] = first if b == 0 else design(self.kind, inp, T, s, e)
                y[sl] = Y[:, s - 1:e]
            names = block_feature_names(self.kind, *spans[0], self.context_names, self.varying_names)
            self.models[0] = fit_booster(X, y, self.rounds, self.threads, self.output, self.tag, names)
        else:
            for b, (s, e) in enumerate(spans):
                names = block_feature_names(self.kind, s, e, self.context_names, self.varying_names)
                self.models[b] = fit_booster(design(self.kind, inp, T, s, e), Y[:, s - 1:e],
                                             self.rounds, self.threads, self.output, f"{self.tag}_b{b:02}", names)
        return self

    def predict(self, inp):
        T = np.hstack([inp.past, np.full((len(inp), H), np.nan, np.float32)])
        for b, (s, e) in enumerate(blocks(self.sigma)):
            model = self.models[0 if self.kind == "RecMO" else b]
            T[:, LAG + s - 1:LAG + e] = model.inplace_predict(design(self.kind, inp, T, s, e)).reshape(len(inp), e - s + 1)
        return T[:, LAG:].astype(np.float64)


def purged_folds(origins, k=FOLDS, purge_h=PURGE_H):
    origins = pd.DatetimeIndex(origins)
    if len(origins) < k:
        raise ValueError("not enough origins for four OOF folds")
    for held in np.array_split(np.arange(len(origins)), k):
        gap = pd.Timedelta(hours=purge_h)
        keep = (origins < origins[held[0]] - gap) | (origins > origins[held[-1]] + gap)
        train = np.flatnonzero(keep)
        if not len(train):
            raise ValueError("purging leaves an empty OOF training set")
        yield train, held


class Rectify:
    def __init__(self, base, residual, rounds=900, threads=1, output=None, tag="fit"):
        self.base_spec, self.res_spec = base, residual
        self.rounds, self.threads, self.output, self.tag = rounds, threads, output, tag

    @staticmethod
    def residual_inputs(inp, B):
        return Inputs(np.hstack([inp.ctx, inp.past, B]), inp.E, np.zeros_like(inp.past))

    def fit(self, inp, Y, origins):
        oof = np.full_like(Y, np.nan, dtype=np.float32)
        for f, (tr, held) in enumerate(purged_folds(origins)):
            base = Block(*self.base_spec, self.rounds, self.threads, self.output, f"{self.tag}_oof{f}")
            base.fit(inp.take(tr), Y[tr])
            oof[held] = base.predict(inp.take(held))
        if not np.isfinite(oof).all():
            raise ValueError("OOF forecasts incomplete")
        self.base = Block(*self.base_spec, self.rounds, self.threads, self.output, f"{self.tag}_base").fit(inp, Y)
        residual_names = CONSTANT + [f"original_speed_lag{k}" for k in range(LAG - 1, -1, -1)] + [f"base_forecast_h{h}" for h in range(1, H + 1)]
        self.residual = Block(*self.res_spec, self.rounds, self.threads, self.output, f"{self.tag}_residual", residual_names)
        self.residual.fit(self.residual_inputs(inp, oof), Y - oof)
        if self.output is not None:
            np.savez_compressed(Path(self.output) / f"{self.tag}_oof.npz", predictions=oof, origins=pd.DatetimeIndex(origins).asi8)
        return self

    def predict(self, inp):
        base = self.base.predict(inp)
        return base + self.residual.predict(self.residual_inputs(inp, base))


def build(experiment, rounds=900, threads=1, output=None, tag="fit"):
    spec = STRATEGIES[experiment]
    if isinstance(spec[0], str):
        return Block(*spec, rounds, threads, output, tag)
    return Rectify(*spec, rounds, threads, output, tag)


def read_origin_table(path, columns):
    frame = pd.read_parquet(path, columns=["origin", *columns], filters=[("h", "==", 1)])
    frame["origin"] = pd.to_datetime(frame.origin, utc=True)
    return frame


def timeline(path):
    raw = read_origin_table(path, ["v_lag0"])
    if (raw.groupby("origin").v_lag0.nunique(dropna=False) > 1).any():
        raise ValueError("train/query speed inputs disagree")
    return raw.drop_duplicates("origin").sort_values("origin").set_index("origin").v_lag0


def load_grid(path, period, training, speeds, stale):
    filters = [(f"fit_{period}", "==", True)] if training else [("set", "==", "query"), ("period", "==", period)]
    available = [mapped(c) for c in CONSTANT if c != "img_stale"]
    constants = pd.read_parquet(path, columns=["origin", *available], filters=[*filters, ("h", "==", 1)]).sort_values("origin", kind="stable")
    origins = pd.DatetimeIndex(pd.to_datetime(constants.origin, utc=True))
    if origins.has_duplicates:
        raise ValueError("duplicate origins in selected grid")
    constants["img_stale"] = stale.reindex(origins).to_numpy()
    C = constants[[mapped(c) for c in CONSTANT]].to_numpy(np.float32)
    rows = pd.read_parquet(path, columns=["origin", "h", "y", *VARYING], filters=filters).sort_values(["origin", "h"], kind="stable")
    if len(rows) != len(origins) * H or not np.array_equal(rows.h.to_numpy().reshape(-1, H), np.broadcast_to(np.arange(1, H + 1), (len(origins), H))):
        raise ValueError("incomplete origin/h grid")
    row_origins = pd.DatetimeIndex(pd.to_datetime(rows.origin.iloc[::H], utc=True))
    if not origins.equals(row_origins):
        raise ValueError("constant and varying origin grids differ")
    E = rows[VARYING].to_numpy(np.float32).reshape(len(origins), H, len(VARYING))
    Y = rows.y.to_numpy(np.float32).reshape(len(origins), H)
    if training and not (origins + pd.Timedelta(hours=144) <= pd.Timestamp(CUTOFFS[period], tz="UTC")).all():
        raise ValueError("training maturity mask exceeds cutoff")
    return origins, Inputs(C, E, past_window(speeds, origins)), Y


def run_period(experiment, data, raw_data, out, period, threads=1, smoke=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    speeds = timeline(data)
    stale = stale_series(read_origin_table(raw_data, ["img_age_h"]))
    origins, inp, Y = load_grid(data, period, True, speeds, stale)
    query_origins, query, _ = load_grid(data, period, False, speeds, stale)
    if len(query_origins) != QUERY_COUNTS[period]:
        raise ValueError(f"{period}: expected {QUERY_COUNTS[period]} queries, got {len(query_origins)}")
    if smoke:
        take = np.linspace(0, len(origins) - 1, min(128, len(origins))).astype(int)
        origins, inp, Y = origins[take], inp.take(take), Y[take]
        query_origins, query = query_origins[:8], query.take(slice(0, 8))
    rounds = 2 if smoke else 900
    model = build(experiment, rounds, threads, out / f"{period}_models", period)
    if isinstance(model, Rectify):
        model.fit(inp, Y, origins)
    else:
        model.fit(inp, Y)
    inference_start = time.perf_counter()
    predictions = np.clip(model.predict(query), 1.0, 3000.0)
    inference_seconds = time.perf_counter() - inference_start
    if predictions.shape != (len(query_origins), H) or not np.isfinite(predictions).all():
        raise ValueError("invalid predictions")
    np.savez_compressed(out / f"{period}_predictions.npz", pred=predictions, origins=query_origins.asi8)
    report = {"experiment": experiment, "period": period, "smoke": smoke,
              "source": str(Path(data).resolve()), "unfilled_source": str(Path(raw_data).resolve()),
              "source_bytes": Path(data).stat().st_size, "source_mtime_ns": Path(data).stat().st_mtime_ns,
              "constant_features": CONSTANT, "mapped_constant_features": [mapped(c) for c in CONSTANT],
              "varying_features": VARYING, "original_feature_count_including_h": 111,
              "history_length": LAG, "history_rule": "origin input snapshots O-23..O; newest ACE observation O-1h",
              "img_stale_rule": "no prior image or origin minus last observed image end > 6h; recovered from unfilled new df",
              "target_rule": "campaign training-y ffill; remaining missing loss weight=0; teacher input trajectories causal ffill",
              "training_cutoff": CUTOFFS[period], "maturity_hours": 144,
              "train_origins": len(origins), "train_start": str(origins.min()), "train_end": str(origins.max()),
              "query_origins": len(query_origins), "query_start": str(query_origins.min()), "query_end": str(query_origins.max()),
              "parameters": PARAMS, "rounds": rounds, "threads": threads,
              "inference_seconds_per_fold": inference_seconds / len(query_origins),
              "end_to_end_seconds": time.perf_counter() - start,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgb.__version__}}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    report["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
    (out / f"{period}_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def merge(out):
    out = Path(out)
    frames = []
    for period in CUTOFFS:
        z = np.load(out / f"{period}_predictions.npz")
        origins = pd.DatetimeIndex(z["origins"], tz="UTC")
        pred = z["pred"]
        if len(origins) != QUERY_COUNTS[period] or pred.shape != (len(origins), H) or not np.isfinite(pred).all():
            raise ValueError(f"{period}: incomplete full-run predictions")
        frames.append(pd.DataFrame({"origin_last_input_utc": np.repeat(origins.strftime("%Y-%m-%dT%H:%M:%SZ"), H),
                                    "horizon_hours": np.tile(np.arange(1, H + 1), len(origins)), "pred_kms": pred.ravel()}))
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["origin_last_input_utc", "horizon_hours"]).any():
        raise ValueError("duplicate output keys")
    result.to_parquet(out / "predictions.parquet", index=False)
    return len(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(STRATEGIES))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, default=Path("data-v2/input/df.parquet"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--period", choices=list(CUTOFFS))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.merge:
        print(json.dumps({"prediction_rows": merge(args.out)}))
        return
    for period in [args.period] if args.period else (["F1"] if args.smoke else list(CUTOFFS)):
        print(json.dumps(run_period(args.experiment, args.data, args.raw_data, args.out, period, args.threads, args.smoke)), flush=True)
    if not args.period and not args.smoke:
        print(json.dumps({"prediction_rows": merge(args.out)}))


if __name__ == "__main__":
    import sys
    if "--experiment" not in sys.argv:
        sys.argv.extend(["--experiment", 'e2000fs_dirmo_12'])
    main()
