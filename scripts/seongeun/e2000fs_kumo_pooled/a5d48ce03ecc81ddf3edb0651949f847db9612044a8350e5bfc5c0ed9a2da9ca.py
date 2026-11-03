"""Frozen Kumo/TabICLv2/RealMLP strategies on causal ffilled input; optional original CUDA backend.
The scalar architecture, latest-5000 context, eight ICL estimators and RealMLP
256-epoch budget are retained. Epoch selection uses a purged earlier window
inside each official fit mask, since the old window overlaps new evaluation.
Smoke results are intentionally incomplete and cannot pass merge validation.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import numpy as np
import pandas as pd
H, LAG = 72, 24
EXPERIMENTS = {f"e2000fs_{family}_{strategy}": (family, strategy)
               for family in ("kumo", "tabiclv2", "realmlp")
               for strategy in ("direct", "recursive", "dirrec", "pooled")}


CUTOFFS = {"F1": "2025-12-29", "F2": "2026-03-29", "F3": "2026-06-29"}

QUERY_COUNTS = {"F1": 2089, "F2": 2113, "F3": 2113}

CONSTANT = ['v_lag0', 'v_lag1', 'v_lag3', 'v_lag6', 'v_lag12', 'v_lag24', 'v_lag48', 'v_lag72', 'v_lag120', 'v_m6', 'v_sd6', 'v_m24', 'v_sd24', 'v_m72', 'v_sd72', 'ace_speed_kms_last', 'ace_speed_kms_m6', 'ace_speed_kms_m24', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_density_cm3_m24', 'ace_temperature_k_last', 'ace_temperature_k_m6', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_bt_nt_m24', 'ace_bz_gsm_nt_last', 'ace_bz_gsm_nt_m6', 'ace_bz_gsm_nt_m24', 'ace_by_gsm_nt_last', 'ace_by_gsm_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_e_channel1_last', 'ace_epam_e_channel1_m6', 'ace_epam_e_channel1_m24', 'ace_epam_e_channel2_last', 'ace_epam_e_channel2_m6', 'ace_epam_e_channel2_m24', 'ace_epam_p_channel1_last', 'ace_epam_p_channel1_m6', 'ace_epam_p_channel1_m24', 'ace_epam_p_channel3_last', 'ace_epam_p_channel3_m6', 'ace_epam_p_channel3_m24', 'ace_epam_p_channel5_last', 'ace_epam_p_channel5_m6', 'ace_epam_p_channel5_m24', 'ace_sis_gt10_flux_last', 'ace_sis_gt10_flux_m6', 'ace_sis_gt10_flux_m24', 'speed_d24', 'epam_p1_d6', 'ch_ch_t035_area', 'ch_ch_t035_max_area', 'ch_ch_t035_top1_area', 'ch_ch_t035_top2_area', 'ch_ch_t035_top3_area', 'ch_ch_t035_meridian15_area', 'ch_ch_t035_meridian15_equatorial_area', 'ch_ch_t035_meridian30_area', 'ch_ch_t035_meridian30_equatorial_area', 'ch_ch_t035_equatorial_area', 'ch_ch_t045_area', 'ch_ch_t045_max_area', 'ch_ch_t045_top1_area', 'ch_ch_t045_top2_area', 'ch_ch_t045_top3_area', 'ch_ch_t045_meridian15_area', 'ch_ch_t045_meridian15_equatorial_area', 'ch_ch_t045_meridian30_area', 'ch_ch_t045_meridian30_equatorial_area', 'ch_ch_t045_equatorial_area', 'ch_ch_t055_area', 'ch_ch_t055_max_area', 'ch_ch_t055_top1_area', 'ch_ch_t055_top2_area', 'ch_ch_t055_top3_area', 'ch_ch_t055_meridian15_area', 'ch_ch_t055_meridian15_equatorial_area', 'ch_ch_t055_meridian30_area', 'ch_ch_t055_meridian30_equatorial_area', 'ch_ch_t055_equatorial_area', 'img_age_h', 'img_stale', 'enlil_next_arr_h', 'enlil_since_arr_h', 'enlil_n_pending', 'enlil_kp', 'enlil_glancing']

VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v350_dark', 'strip_v350_cover', 'strip_v450_logmed', 'strip_v450_dark', 'strip_v450_cover', 'strip_v550_logmed', 'strip_v550_dark', 'strip_v550_cover', 'strip_v650_logmed', 'strip_v650_dark', 'strip_v650_cover', 'strip_v750_logmed', 'strip_v750_dark', 'strip_v750_cover', 'enlil_dt_target']

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


def teacher_inputs(inp, y):
    return pd.DataFrame(np.hstack([inp.past, y])).ffill(axis=1).to_numpy(np.float32)[:, LAG:]


def features(strategy, inp, previous, rows, horizons):
    pieces = [inp.ctx[rows]]
    if strategy == "recursive":
        trajectory = np.hstack([inp.past, previous])
        pieces.append(trajectory[rows[:, None], horizons[:, None] + np.arange(LAG)])
    else:
        pieces.append(inp.past[rows])
        if strategy == "dirrec":
            if len(np.unique(horizons)) != 1:
                raise ValueError("DirRec design requires a single head")
            pieces.append(previous[rows, :int(horizons[0])])
    pieces.append(inp.E[rows, horizons])
    if strategy in ("recursive", "pooled"):
        pieces.append((horizons + 1)[:, None])
    return np.hstack(pieces).astype(np.float32)


def context_cells(origins, y, head, shared, limit):
    order = np.argsort(pd.DatetimeIndex(origins).asi8, kind="stable")
    rows = np.repeat(order, H) if shared else order
    horizons = np.tile(np.arange(H), len(order)) if shared else np.full(len(order), head)
    observed = np.isfinite(y[rows, horizons])
    rows, horizons = rows[observed][-limit:], horizons[observed][-limit:]
    if not len(rows):
        raise ValueError("no observed context targets")
    return rows, horizons


def selection_split(origins):
    dates = pd.DatetimeIndex(origins)
    start = dates.max() - pd.Timedelta(days=81)
    train = np.flatnonzero(dates + pd.Timedelta(hours=144) < start)
    valid = np.flatnonzero(dates >= start)
    if not len(train) or not len(valid):
        raise ValueError("insufficient history for purged epoch selection")
    return train, valid


def checkpoint(family, smoke=False):
    from huggingface_hub import hf_hub_download
    if family == "kumo":
        repo, filename, revision = "nvidia/Kumo-Tabular", f"{'small' if smoke else 'large'}/regressor.pt", "v1.0.1"
    elif family == "tabiclv2":
        repo, filename, revision = "jingang/TabICL", "tabicl-regressor-v2-20260212.ckpt", "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
    else:
        return None
    path = hf_hub_download(repo, filename, revision=revision, local_files_only=True)
    return dict(repo=repo, filename=filename, revision=revision, local_path=path)


class Learner:
    def __init__(self, family, threads=1, smoke=False, stop_epoch=None, device="cpu"):
        if device not in ("cpu", "cuda:0"):
            raise ValueError("device must be cpu or cuda:0")
        self.device = device
        import torch
        torch.set_num_threads(threads)
        self.family, self.smoke = family, smoke
        self.identity = checkpoint(family, smoke)
        if family == "realmlp":
            from pytabkit import RealMLP_TD_Regressor
            params = dict(random_state=42, device=device, n_threads=threads, n_cv=1,
                          n_refit=0, n_repeats=1, n_ens=1, val_fraction=0.,
                          n_epochs=2 if smoke else 256, verbosity=0,
                          train_metric_name="mse", val_metric_name="rmse")
            if stop_epoch is not None:
                params["stop_epoch"] = stop_epoch
            self.model = RealMLP_TD_Regressor(**params)
        else:
            if family == "kumo":
                import sdm.models.kumo.tabular.model as module
                constructor = lambda: module.KumoTabular(task="regression", size="small" if smoke else "large", device=device)
            else:
                import sdm.models.tabiclv2.model as module
                constructor = lambda: module.TabICLv2(task="regression", device=device)
            with patch.object(module, "download_checkpoint", lambda *a, **k: self.identity["local_path"]):
                self.model = constructor()

    def table(self, x):
        import sdm
        frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
        return sdm.TableTensor.from_pandas(frame, stypes={c: sdm.Stype.numerical for c in frame}, device=self.device)

    def autocast(self):
        import torch
        return torch.amp.autocast("cuda", dtype=torch.float16) if self.device.startswith("cuda") else nullcontext()

    def fit(self, x, y, validation=None):
        if not np.isfinite(y).all():
            raise ValueError("missing targets may not enter model fitting")
        if np.isinf(x).any():
            raise ValueError("infinite features")
        if self.family == "realmlp":
            kw = {} if validation is None else dict(X_val=validation[0], y_val=validation[1])
            self.model.fit(x, y, **kw)
        else:
            import torch
            with torch.inference_mode(), self.autocast():
                self.model.fit(x=self.table(x), y=self.table(y[:, None]),
                               num_estimators=1 if self.smoke else 8, estimator_batch_size=1,
                               generator=torch.Generator(device=self.device).manual_seed(42))
        return self

    def predict(self, x):
        if self.family == "realmlp":
            return np.asarray(self.model.predict(x), dtype=np.float64).reshape(-1)
        import torch
        with torch.inference_mode(), self.autocast():
            values = self.model.predict(self.table(x)).numerical
            if self.device.startswith("cuda"):
                # Exact original adapter: cast quantiles to float32, then NumPy float64 mean.
                return np.asarray(values.float().cpu().numpy(), dtype=np.float64).mean(axis=1)
            return values.mean(dim=1).cpu().numpy().astype(np.float64)


def fit_head(family, strategy, inp, y, origins, head, output, threads, smoke, device="cpu"):
    shared = strategy in ("recursive", "pooled")
    limit = 32 if smoke else 5000
    previous = teacher_inputs(inp, y)
    selected = None
    selection = None
    if family == "realmlp":
        tr, va = selection_split(origins)
        if smoke:
            tr, va = tr[-40:], va[:2]
        selection = dict(train_start=str(origins[tr].min()), train_end=str(origins[tr].max()),
                         valid_start=str(origins[va].min()), valid_end=str(origins[va].max()))
        sr, sh = context_cells(origins[tr], y[tr], head, shared, limit)
        vr = np.repeat(va, H) if shared else va
        vh = np.tile(np.arange(H), len(va)) if shared else np.full(len(va), head)
        keep = np.isfinite(y[vr, vh])
        vr, vh = vr[keep], vh[keep]
        if not len(vr):
            raise ValueError("no observed epoch-selection labels")
        selector = Learner(family, threads, smoke, device=device).fit(
            features(strategy, inp, previous, tr[sr], sh), y[tr[sr], sh],
            (features(strategy, inp, previous, vr, vh), y[vr, vh]))
        stop = selector.model.fit_params_["stop_epoch"]
        selected = int(stop["rmse"] if isinstance(stop, dict) else stop)
        del selector
    rows, horizons = context_cells(origins, y, head, shared, limit)
    x, target = features(strategy, inp, previous, rows, horizons), y[rows, horizons]
    start = time.perf_counter()
    learner = Learner(family, threads, smoke, selected, device=device).fit(x, target)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tag = f"head{head:02}"
    np.savez_compressed(output / f"{tag}_context.npz", x=x, y=target,
                        origins=pd.DatetimeIndex(origins[rows]).asi8, horizons=horizons + 1)
    if family == "realmlp":
        import joblib
        joblib.dump(learner.model, output / f"{tag}.joblib")
    report = dict(family=family, strategy=strategy, head=head + 1, shared=shared,
                  seed=42, device=device, context_rows=len(rows), feature_count=x.shape[1],
                  selection_window=selection, selected_epoch=selected, epoch_budget=2 if smoke else 256,
                  estimators=1 if smoke else 8, checkpoint=learner.identity,
                  seconds=time.perf_counter() - start,
                  model_artifact="joblib" if family == "realmlp" else "pinned checkpoint plus exact fitted context and seed",
                  selection_rule="latest 81 days within fit mask; earlier training purged by 144h" if family == "realmlp" else None)
    (output / f"{tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return learner


def run_period(experiment, data, raw_data, out, period, threads=1, smoke=False, device="cpu"):
    family, strategy = EXPERIMENTS[experiment]
    checkpoint(family, smoke)  # fail before expensive loading if checkpoint is unavailable
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    speeds = timeline(data)
    stale = stale_series(read_origin_table(raw_data, ["img_age_h"]))
    origins, inp, y = load_grid(data, period, True, speeds, stale)
    query_origins, query, _ = load_grid(data, period, False, speeds, stale)
    if len(query_origins) != QUERY_COUNTS[period]:
        raise ValueError("unexpected query grid")
    if smoke:
        query_origins, query = query_origins[:2], query.take(slice(0, 2))
    pred = np.full((len(query_origins), H), np.nan)
    model = None
    inference = 0.
    for head in range(3 if smoke else H):
        if model is None:
            model = fit_head(family, strategy, inp, y, origins, head, out / f"{period}_models", threads, smoke, device)
        begin = time.perf_counter()
        x = features(strategy, query, pred, np.arange(len(query)), np.full(len(query), head))
        batch = 8 if smoke else 256
        pred[:, head] = np.clip(np.concatenate([model.predict(x[i:i + batch]) for i in range(0, len(x), batch)]), 1., 3000.)
        inference += time.perf_counter() - begin
        if strategy not in ("recursive", "pooled"):
            model = None
        print(f"{period} head {head+1} complete", flush=True)
    if not np.isfinite(pred[:, :3 if smoke else H]).all():
        raise ValueError("nonfinite predictions")
    np.savez_compressed(out / f"{period}_predictions.npz", pred=pred, origins=query_origins.asi8)
    import importlib.metadata as metadata
    report = dict(experiment=experiment, period=period, smoke=smoke, device=device, seed=42,
                  source=str(Path(data).resolve()), unfilled_source=str(Path(raw_data).resolve()),
                  train_start=str(origins.min()), train_end=str(origins.max()), train_origins=len(origins),
                  training_cutoff=CUTOFFS[period], maturity_hours=144,
                  constant_features=CONSTANT, mapped_constant_features=[mapped(c) for c in CONSTANT], varying_features=VARYING,
                  history_length=LAG, target_rule="campaign training-y ffill; remaining missing labels excluded; teacher inputs causally ffilled",
                  backend_change=("original CUDA FP16 autocast restored" if family != "realmlp" else "original RealMLP CUDA backend restored") if device.startswith("cuda") else "CPU arithmetic instead of original CUDA backend",
                  inference_seconds_per_fold=inference / len(query_origins), end_to_end_seconds=time.perf_counter()-start,
                  versions={name:metadata.version(name) for name in ("numpy", "pandas", "torch", "pytabkit", "structured-data-models")})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
    (out / f"{period}_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw-data", type=Path, default=Path("data-v2/input/df.parquet"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--period", choices=list(CUTOFFS))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.merge:
        print(json.dumps({"prediction_rows": merge(args.out)}))
        return
    for period in [args.period] if args.period else (["F1"] if args.smoke else list(CUTOFFS)):
        print(json.dumps(run_period(args.experiment, args.data, args.raw_data, args.out, period, args.threads, args.smoke, args.device)), flush=True)
    if not args.period and not args.smoke:
        print(json.dumps({"prediction_rows": merge(args.out)}))


if __name__ == "__main__":
    import sys
    if "--experiment" not in sys.argv:
        sys.argv.extend(["--experiment", 'e2000fs_kumo_pooled'])
    main()
