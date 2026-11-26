"""The sequence-model runner shared by the vanila DL board rows (GRU, iTransformer, TSMixer). PyTorch, GPU only.

Input: data-v2/input/df_ts.parquet (one row per hour bin). For an origin O every model sees one
(120 steps x channels) tensor, step j = hour bin O-119+j:
  ACE       21 in-situ bin values (log10 for densities, temperatures and particle fluxes), causal forward fill of
            at most 119 h, plus one observed/missing mask per instrument (SWEPAM, MAG, EPAM, SIS)
  rec       27-day recurrence: speed(t + 72 h - d days), d = 26/27/28, so the last 72 steps hold the previous
            rotation's speed at leads 1..72 (always older than t)
  ENLIL     the five hourly WSA-ENLIL arrival features plus a past-arrival mask
  image     longitude sweep of ONE SUVI frame (the frame in bin O): stripe k = 119-j, the longitude that crossed
            the central meridian k hours before O (logmed, dark, cover) plus a frame mask. When bin O has no frame
            the latest frame within 24 h (m hours old) is used with stripes k-m (rotation-corrected walk-back).
Channels are standardized with statistics of the period's training span only (stored in the checkpoint).

Target: speed at O+1..O+72; training labels are forward filled on the target time (as vanila_mse). Loss: unweighted MSE.
With SeqSpec.persistence_residual the forecast is the last observed speed of the window plus the network output
(the network learns the departure from persistence); without it the network outputs the speed itself.
With SeqSpec.tail_weight the squared error of targets >= 550 km/s weighs 4 (objectives.tail_weights, normalised to
mean 1 per lead), in training and in the early-stopping criterion on the calibration window.
With SeqSpec.summary the model also receives the XGBoost rows' summary features (research/_module/summary.py):
model(x, constant (B, 23), per lead (B, 72, 8)) instead of model(x).
With SeqSpec.swa_start the network is the running average of the epoch-end weights from that epoch on
(torch.optim.swa_utils.AveragedModel, batch-norm statistics recomputed on the training rows); early stopping and the
full model use the averaged weights. With SeqSpec.lead_checkpoints each lead bucket (h1-6, h7-24, h25-48, h49-72)
gets its own early-stopping epoch on the calibration window; the full model keeps a snapshot at each of those epochs
and the forecast of a bucket comes from its snapshot.
With SeqSpec.base_forecast (a folder of {period}.parquet raw 72-lead forecasts per origin, e.g. out-of-fold XGBoost on
the training origins and the full model on the query origins) the forecast is that base plus the network output.

For each seed (42, 43, 44) and period (F1, F2, F3), the vanila_mse protocol with a network in place of the boosters:
a prior model on the reference window (origins + 144 h <= last 3 months) with epoch early stopping on the calibration
window gives the epoch count E* and the frozen per-lead offset; a full model on all training origins runs E* epochs.
Query forecast = full model, clip [1, 3000], + offset, clip [1, 3000]; board forecast = fixed equal-weight mean of the
three seeds; scored by the official scorer. Leaderboard-period truth is used for scoring only.

An experiment.py declares a SeqSpec (model builder, hyperparameters) and calls main(SPEC, __doc__, HERE).
Outputs: see common.output_paths; checkpoints weights/F{n}/s{seed}.npz, logs/{data,run,s42,s43,s44,mean3}.jsonl and
logs/F{n}_s{seed}_metadata.json, results/{s42,s43,s44,mean3}_*.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass, field
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Callable

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch

from . import calibration as cal
from . import summary as summary_features
from .objectives import tail_weights
from .common import (CUTOFFS, H, QUERY_COUNTS, ROOT, SEEDS, file_identity, git_commit, launch, mean3,
                     observations_path, output_paths, prediction_frame, run_logs, score, versions, write_json,
                     write_results)
from .table import PURGE_HOURS

DATA_TS = ROOT / 'data-v2/input/df_ts.parquet'
LOOKBACK = 120
FILL_LIMIT = LOOKBACK - 1
WALK_BACK = 24
ACE_LINEAR = ['ace_speed_kms', 'ace_bx_gsm_nt', 'ace_by_gsm_nt', 'ace_bz_gsm_nt', 'ace_bt_nt', 'ace_lat_gsm_deg',
              'ace_lon_gsm_deg']
ACE_LOG = ['ace_density_cm3', 'ace_temperature_k', 'ace_epam_e_channel1', 'ace_epam_e_channel2',
           'ace_epam_p_channel1', 'ace_epam_p_channel2', 'ace_epam_p_channel3', 'ace_epam_p_channel4',
           'ace_epam_p_channel5', 'ace_sis_gt10_flux', 'ace_sis_gt30_flux']
ACE = ACE_LINEAR + ACE_LOG
MASKS = {'mask_swepam': 'ace_speed_kms', 'mask_mag': 'ace_bt_nt', 'mask_epam': 'ace_epam_p_channel1',
         'mask_sis': 'ace_sis_gt10_flux'}
REC_DAYS = (26, 27, 28)
REC = [f'rec{d}_lead{H}' for d in REC_DAYS]
ENLIL = ['enlil_next_arr_h', 'enlil_since_arr_h', 'enlil_n_pending', 'enlil_kp', 'enlil_glancing']
STRIP_STATS = ('logmed', 'dark', 'cover')
STRIP_FIRST_K = -72
IMAGE = [f'suvi_sweep_{stat}' for stat in STRIP_STATS]
SERIES = [*ACE, *MASKS, *REC, *ENLIL, 'mask_enlil_past']
CHANNELS = [*SERIES, *IMAGE, 'mask_image']
UNSCALED = {*MASKS, 'mask_enlil_past', 'mask_image'}
SPEED = CHANNELS.index('ace_speed_kms')
TRAINING = dict(optimizer='AdamW', learning_rate=1e-4, weight_decay=1e-2, batch_size=256, max_epochs=60, patience=8,
                gradient_clip=1.0, loss='unweighted MSE on the standardized 72-lead target')


@dataclass
class SeqSpec:
    experiment: str                      # submission ID (board row)
    family: str                          # gru / itransformer / tsmixer (sunrun family, log component)
    build_model: Callable                # (number of channels, speed channel index) -> nn.Module: (B, 120, C) -> (B, 72)
    hyperparameters: dict                # architecture hyperparameters for logs and metadata
    libraries: tuple = ()                # extra distributions whose versions are recorded
    persistence_residual: bool = False   # forecast = last observed speed + network output
    tail_weight: bool = False            # squared error of targets >= 550 km/s weighs 4 (objectives.tail_weights)
    summary: bool = False                # model(x, constant, per-lead) with the XGBoost rows' summary features
    swa_start: int | None = None         # average the epoch-end weights from this epoch on (SWA)
    lead_checkpoints: bool = False       # one early-stopping epoch and weight snapshot per lead bucket
    base_forecast: str | None = None     # folder of {period}.parquet base forecasts; forecast = base + network
    alpha: float = cal.ALPHA
    offset_window: int = cal.OFFSET_WINDOW
    threads: int = 2
    training: dict = field(default_factory=lambda: dict(TRAINING))


def strip_columns(stat):
    return [f"suvi_strip_{stat}_k{'m' if k < 0 else 'p'}{abs(k):03d}" for k in range(STRIP_FIRST_K, LOOKBACK + WALK_BACK)]


class Table:
    """df_ts.parquet as hourly arrays: unscaled series channels, stripe cube, frame walk-back and filled labels."""

    def __init__(self, path, logs, summary=False):
        began = time.perf_counter()
        columns = ['time', 'train_origin', 'query_origin', 'period', *[f'fit_{p}' for p in CUTOFFS], 'y',
                   *ACE, *ENLIL, 'img_age_h', *[c for stat in STRIP_STATS for c in strip_columns(stat)]]
        frame = pd.read_parquet(path, columns=columns)
        self.time = pd.DatetimeIndex(frame['time'])
        if self.time.has_duplicates or not (np.diff(self.time.asi8) == 3600 * 10 ** 9).all():
            raise ValueError('df_ts needs a gap-free hourly time axis')
        self.frame = frame[['train_origin', 'query_origin', 'period', *[f'fit_{p}' for p in CUTOFFS]]]
        raw = frame[ACE].astype(np.float64)
        raw[ACE_LOG] = np.log10(raw[ACE_LOG].where(raw[ACE_LOG] > 0))
        filled = raw.ffill(limit=FILL_LIMIT)
        series = {name: filled[name].to_numpy() for name in ACE}
        for name, source in MASKS.items():
            series[name] = raw[source].notna().to_numpy(np.float64)
        speed = filled['ace_speed_kms']
        for name, days in zip(REC, REC_DAYS):
            # value at t = speed bin (t + 72 h - days): always older than t
            series[name] = speed.shift(days * 24 - H).to_numpy()
        for name in ENLIL:
            series[name] = frame[name].to_numpy(np.float64)
        series['mask_enlil_past'] = frame['enlil_since_arr_h'].notna().to_numpy(np.float64)
        self.series = np.stack([series[name] for name in SERIES], 1).astype(np.float32)
        self.strips = np.stack([frame[strip_columns(stat)].to_numpy(np.float32) for stat in STRIP_STATS], 2)
        have = frame['img_age_h'].notna().to_numpy()
        position = np.arange(len(frame))
        latest = np.maximum.accumulate(np.where(have, position, -1))
        self.frame_age = np.where((latest >= 0) & (position - latest <= WALK_BACK), position - latest, -1)
        observed = frame['y'].to_numpy(np.float64)
        self.y_filled = pd.Series(observed).ffill().to_numpy()
        self.summary = None
        if summary:
            first = int(np.flatnonzero(frame['train_origin'].to_numpy() | frame['query_origin'].to_numpy())[0])
            constant, varying = summary_features.build(path, first)
            self.summary = (summary_features.scaled(constant).astype(np.float32), varying)
        logs('data', 'table_ready', source=file_identity(path), rows=len(frame), start=str(self.time[0]),
             end=str(self.time[-1]), channels=CHANNELS, channel_count=len(CHANNELS),
             frames_present=float(have.mean()), frame_within_walk_back=float((self.frame_age >= 0).mean()),
             y_missing=float(np.isnan(observed).mean()),
             **(dict(summary_constant=summary_features.CONSTANT, summary_varying=summary_features.VARYING,
                     summary_log10=summary_features.LOG10) if summary else {}),
             policy=f'ACE causal ffill <= {FILL_LIMIT} h, log10 of {ACE_LOG}; recurrence = filled speed at '
                    f't + {H} h - 26/27/28 d; image = stripes 0..{LOOKBACK - 1} of the bin-O frame or of the latest frame '
                    f'within {WALK_BACK} h shifted by its age; training y ffill on the target time; '
                    'remaining NaN = 0 after standardization',
             seconds=round(time.perf_counter() - began, 3))

    def origins(self, period, training):
        if training:
            mask = self.frame[f'fit_{period}'].to_numpy()
        else:
            mask = (self.frame['query_origin'] & self.frame['period'].eq(period)).to_numpy()
        index = np.flatnonzero(mask)
        if len(index) == 0 or index.min() < LOOKBACK + max(REC_DAYS) * 24 or index.max() + H >= len(self.time):
            raise ValueError(f'{period}: origins need a full lookback, recurrence and target span inside df_ts')
        if training and not (self.time[index] + pd.Timedelta(hours=PURGE_HOURS)
                             <= pd.Timestamp(CUTOFFS[period], tz='UTC')).all():
            raise ValueError('training cutoff violated')
        return index

    def targets(self, index):
        """Filled training labels at leads 1..72 (training origins only)."""
        values = self.y_filled[index[:, None] + np.arange(1, H + 1)[None, :]]
        if not np.isfinite(values).all():
            raise ValueError('Prepared training targets must be finite')
        return values


class Windows:
    """GPU copy of the standardized table; batch(index) gathers (B, 120, C) inputs on the device."""

    def __init__(self, table, norm, device, base=None):
        mean, std = np.asarray(norm['mean'], np.float32), np.asarray(norm['std'], np.float32)
        count = len(SERIES)
        series = (table.series - mean[None, :count]) / std[None, :count]
        strips = (table.strips - mean[None, None, count:count + 3]) / std[None, None, count:count + 3]
        self.series = torch.from_numpy(np.nan_to_num(series, nan=0.)).to(device)
        self.strips = torch.from_numpy(np.nan_to_num(strips, nan=0.)).to(device)
        self.frame_age = torch.from_numpy(table.frame_age).to(device)
        self.steps = torch.arange(-(LOOKBACK - 1), 1, device=device)
        self.sweep = torch.arange(LOOKBACK - 1, -1, -1, device=device) - STRIP_FIRST_K
        self.y_mean, self.y_std = float(norm['y_mean']), float(norm['y_std'])
        self.base = None
        if base is not None:
            self.base = torch.from_numpy(((base - self.y_mean) / self.y_std).astype(np.float32)).to(device)
        self.summary = None
        if 'summary' in norm:
            stats = {key: np.asarray(value, np.float32) for key, value in norm['summary'].items()}
            constant = (table.summary[0] - stats['constant_mean'][None]) / stats['constant_std'][None]
            varying = (table.summary[1] - stats['varying_mean'][None, None]) / stats['varying_std'][None, None]
            self.summary = (torch.from_numpy(np.nan_to_num(constant, nan=0.)).to(device),
                            torch.from_numpy(np.nan_to_num(varying, nan=0.)).to(device))
        # last-step speed channel -> standardized target units
        self.speed_scale = float(std[SPEED]) / self.y_std
        self.speed_shift = (float(mean[SPEED]) - self.y_mean) / self.y_std
        self.device = device

    def batch(self, index):
        index = torch.as_tensor(index, device=self.device)
        series = self.series[index[:, None] + self.steps[None, :]]
        age = self.frame_age[index]
        present = age >= 0
        source = index - age.clamp(min=0)
        image = self.strips[source[:, None], self.sweep[None, :] - age.clamp(min=0)[:, None]]
        image = image * present[:, None, None]
        mask = present[:, None, None].expand(-1, LOOKBACK, 1).to(series.dtype)
        return torch.cat([series, image, mask], 2)


def inputs(windows, index):
    """Model inputs of a batch of origins: (x,) or, with summary features, (x, constant, per lead)."""
    x = windows.batch(index)
    index = torch.as_tensor(index, device=windows.device)
    batch = (x,) if windows.summary is None else (x, windows.summary[0][index], windows.summary[1][index])
    if windows.base is not None:
        base = windows.base[index]
        if not torch.isfinite(base).all():
            raise ValueError('Base forecast missing for some origins')
        batch = (*batch, base)
    return batch


def fit_norm(table, index):
    """Channel and target statistics of the training span only (first lookback step .. last training origin)."""
    first, last = index.min() - (LOOKBACK - 1), index.max()
    series = table.series[first:last + 1].astype(np.float64)
    mean, std = np.nanmean(series, 0), np.nanstd(series, 0)
    frames = np.flatnonzero(table.frame_age[first:last + 1] == 0) + first
    sweep = table.strips[frames][:, -STRIP_FIRST_K:-STRIP_FIRST_K + LOOKBACK].astype(np.float64).reshape(-1, 3)
    mean, std = np.r_[mean, np.nanmean(sweep, 0), 0.], np.r_[std, np.nanstd(sweep, 0), 1.]
    for position, name in enumerate(CHANNELS):
        if name in UNSCALED:
            mean[position], std[position] = 0., 1.
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Degenerate channel statistics')
    targets = table.targets(index)
    norm = dict(channels=CHANNELS, mean=mean.tolist(), std=std.tolist(), y_mean=float(targets.mean()),
                y_std=float(targets.std()), span_start=str(table.time[first]), span_end=str(table.time[last]))
    if table.summary is not None:
        # statistics over the training origins only
        constant, varying = (values[index].astype(np.float64) for values in table.summary)
        stats = dict(constant_mean=np.nanmean(constant, 0), constant_std=np.nanstd(constant, 0),
                     varying_mean=np.nanmean(varying, (0, 1)), varying_std=np.nanstd(varying, (0, 1)))
        if not all(np.isfinite(v).all() for v in stats.values()) or any((stats[k] <= 0).any() for k in stats if k.endswith('std')):
            raise ValueError('Degenerate summary-feature statistics')
        norm['summary'] = {key: value.tolist() for key, value in stats.items()}
    return norm


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def state_arrays(model):
    return {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}


def load_state(model, arrays, device):
    model.load_state_dict({name: torch.from_numpy(np.asarray(value)) for name, value in arrays.items()})
    return model.to(device).eval()


def forward(spec, model, windows, batch):
    """Standardized 72-lead forecast of one batch of inputs(); adds the window's last observed speed for residual rows."""
    x = batch[0]
    if spec.base_forecast is not None:
        return model(*batch[:-1]) + batch[-1]
    output = model(*batch)
    if spec.persistence_residual:
        output = output + (x[:, -1, SPEED] * windows.speed_scale + windows.speed_shift)[:, None]
    return output


LEAD_BUCKETS = {'h1_6': slice(0, 6), 'h7_24': slice(6, 24), 'h25_48': slice(24, 48), 'h49_72': slice(48, 72)}


class LeadComposite(torch.nn.Module):
    """One network per lead bucket; the forecast of a bucket's leads comes from that bucket's network."""

    def __init__(self, networks):
        super().__init__()
        self.networks = torch.nn.ModuleList(networks)

    def forward(self, *batch):
        outputs = [network(*batch) for network in self.networks]
        return torch.cat([output[:, leads] for output, leads in zip(outputs, LEAD_BUCKETS.values())], 1)


def network(spec):
    """The inference network of a row: one model, or one per lead bucket for lead_checkpoints rows."""
    if spec.lead_checkpoints:
        return LeadComposite([spec.build_model(len(CHANNELS), SPEED) for _ in LEAD_BUCKETS])
    return spec.build_model(len(CHANNELS), SPEED)


def composite_state(states):
    """State arrays of a LeadComposite from one state per lead bucket."""
    return {f'networks.{i}.{name}': value for i, state in enumerate(states) for name, value in state.items()}


@torch.no_grad()
def update_batch_norm(spec, model, windows, rows, batch_size):
    """Recompute batch-norm running statistics of averaged weights on the training rows (cumulative average, as
    torch.optim.swa_utils.update_bn) without touching the random stream of the training loop."""
    norms = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not norms:
        return
    momenta = {m: m.momentum for m in norms}
    for m in norms:
        m.reset_running_stats()
        m.momentum = None
    with torch.random.fork_rng(devices=[windows.device] if windows.device.type == 'cuda' else []):
        model.train()
        for start in range(0, len(rows), batch_size):
            forward(spec, model, windows, inputs(windows, rows[start:start + batch_size]))
    for m in norms:
        m.momentum = momenta[m]


def by_lead(squared):
    """Mean squared error of a (rows x 72) array per lead bucket."""
    return {name: float(squared[:, leads].mean()) for name, leads in LEAD_BUCKETS.items()}


@torch.no_grad()
def predict(spec, model, windows, index, batch_size=1024):
    model.eval()
    parts = [forward(spec, model, windows, inputs(windows, index[start:start + batch_size]))
             for start in range(0, len(index), batch_size)]
    values = torch.cat(parts).double().cpu().numpy() * windows.y_std + windows.y_mean
    if values.shape != (len(index), H) or not np.isfinite(values).all():
        raise ValueError('Nonfinite or misshaped forecasts')
    return values


def train(spec, table, windows, index, holdout, epochs, seed, logs, stream, record):
    """MSE training. With a holdout: early stopping, returns the best-epoch weights and E*. Without: exactly
    `epochs` epochs, returns the final weights. With swa_start the weights are the running average of the epoch-end
    weights from that epoch on. With lead_checkpoints, E* is one epoch per lead bucket ({bucket: epoch}) and the
    weights are a LeadComposite of the snapshots at those epochs; `epochs` is then that mapping for the full model."""
    settings = spec.training
    seed_everything(seed)
    model = spec.build_model(len(CHANNELS), SPEED).to(windows.device)
    buckets = list(LEAD_BUCKETS)
    snapshots = None
    if spec.lead_checkpoints and holdout is None:
        snapshots, epochs = dict(epochs), max(epochs.values())
    averaged, swa_start = None, None
    if spec.swa_start is not None:
        swa_start = min(spec.swa_start, epochs)
        averaged = torch.optim.swa_utils.AveragedModel(model)
    lead_best = {name: (np.inf, 0, None) for name in buckets}
    saved = {}
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'],
                                  weight_decay=settings['weight_decay'])
    target = torch.from_numpy(((table.targets(index) - windows.y_mean) / windows.y_std).astype(np.float32)).to(windows.device)
    rows = torch.from_numpy(index).to(windows.device)
    if holdout is not None:
        held = torch.from_numpy(((table.targets(holdout) - windows.y_mean) / windows.y_std).astype(np.float32)).to(windows.device)
    weights = held_weights = None
    if spec.tail_weight:
        weights = torch.from_numpy(tail_weights(table.targets(index))[1].astype(np.float32)).to(windows.device)
        if holdout is not None:
            held_weights = torch.from_numpy(tail_weights(table.targets(holdout))[1].astype(np.float32)).to(windows.device)
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state, history = np.inf, 0, None, []
    for epoch in range(1, epochs + 1):
        began = time.perf_counter()
        model.train()
        order = torch.randperm(len(index), generator=generator).to(windows.device)
        total, count = 0., 0
        for start in range(0, len(order), settings['batch_size']):
            pick = order[start:start + settings['batch_size']]
            if len(pick) < 2:
                continue
            output = forward(spec, model, windows, inputs(windows, rows[pick]))
            if weights is None:
                loss = torch.nn.functional.mse_loss(output, target[pick])
            else:
                loss = (weights[pick] * (output - target[pick]) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'])
            optimizer.step()
            total, count = total + float(loss.detach()) * len(pick), count + len(pick)
        entry = dict(epoch=epoch, train_mse=total / count)
        if not np.isfinite(entry['train_mse']):
            raise ValueError('Training diverged')
        current = model
        if averaged is not None:
            if epoch < swa_start:
                current = None
            else:
                averaged.update_parameters(model)
                update_batch_norm(spec, averaged.module, windows, rows, settings['batch_size'])
                current = averaged.module
                entry['averaged_epochs'] = epoch - swa_start + 1
        if snapshots is not None:
            for name in buckets:
                if snapshots[name] == epoch:
                    saved[name] = copy.deepcopy(state_arrays(current))
        if holdout is not None and current is not None:
            current.eval()
            with torch.no_grad():
                error = torch.cat([(forward(spec, current, windows, inputs(windows, holdout[s:s + 1024]))
                                    - held[s:s + 1024]) ** 2 for s in range(0, len(holdout), 1024)])
            entry['holdout_mse'] = float(error.mean())
            entry['holdout_by_lead'] = by_lead(error)
            criterion = entry['holdout_mse']
            if held_weights is not None:
                entry['holdout_loss'] = criterion = float((held_weights * error).mean())
            if spec.lead_checkpoints:
                for name in buckets:
                    if entry['holdout_by_lead'][name] < lead_best[name][0]:
                        lead_best[name] = (entry['holdout_by_lead'][name], epoch, copy.deepcopy(state_arrays(current)))
            elif criterion < best:
                best, best_epoch, best_state = criterion, epoch, copy.deepcopy(state_arrays(current))
        history.append(entry)
        if spec.lead_checkpoints and holdout is not None:
            logs(stream, 'epoch', **record, **entry, best_epoch={n: lead_best[n][1] or None for n in buckets},
                 seconds=round(time.perf_counter() - began, 3))
            if all(epoch - lead_best[n][1] >= settings['patience'] for n in buckets if lead_best[n][1]) \
                    and all(lead_best[n][1] for n in buckets):
                break
            continue
        logs(stream, 'epoch', **record, **entry, best_epoch=best_epoch or None,
             seconds=round(time.perf_counter() - began, 3))
        started = best_epoch if swa_start is None else max(best_epoch, swa_start - 1)
        if holdout is not None and epoch - started >= settings['patience']:
            break
    if holdout is None:
        if snapshots is not None:
            return composite_state([saved[name] for name in buckets]), dict(snapshots), history
        final = model if averaged is None else averaged.module
        return state_arrays(final), epochs, history
    if spec.lead_checkpoints:
        return (composite_state([lead_best[name][2] for name in buckets]),
                {name: lead_best[name][1] for name in buckets}, history)
    return best_state, best_epoch, history


def fit_report(spec, table, windows, model, reference, calibration, prior, history, best_epoch):
    """Training-data-only fit check of the prior model at its early-stopping epoch, per lead bucket (standardized MSE):
    a 4,000-origin sample of the reference window and the calibration holdout, each next to persistence and to the
    training-mean forecast, plus the holdout error relative to its own minimum over the epochs."""
    sample = reference[np.linspace(0, len(reference) - 1, min(len(reference), 4000)).astype(int)]
    report = {}
    for name, index, forecast in (('train_sample', sample, predict(spec, model, windows, sample)),
                                  ('holdout', calibration, prior)):
        truth = (table.targets(index) - windows.y_mean) / windows.y_std
        last = windows.batch(index)[:, -1, SPEED].double().cpu().numpy() * windows.speed_scale + windows.speed_shift
        report[name] = dict(model=by_lead(((forecast - windows.y_mean) / windows.y_std - truth) ** 2),
                            persistence=by_lead((last[:, None] - truth) ** 2), training_mean=by_lead(truth ** 2))
    scored = [entry for entry in history if 'holdout_by_lead' in entry]
    chosen = {name: (history[(best_epoch[name] if isinstance(best_epoch, dict) else best_epoch) - 1]
                     ['holdout_by_lead'][name]) for name in LEAD_BUCKETS}
    report['holdout_over_own_minimum'] = {name: chosen[name] / min(entry['holdout_by_lead'][name] for entry in scored)
                                          for name in LEAD_BUCKETS}
    report['epoch'] = best_epoch
    return report


def load_base(spec, table, period):
    """(time x 72) raw base forecasts of one period on the df_ts time axis (NaN where the folder has no origin)."""
    if spec.base_forecast is None:
        return None, None
    path = Path(spec.base_forecast) / f'{period}.parquet'
    frame = pd.read_parquet(path)
    position = table.time.get_indexer(pd.DatetimeIndex(pd.to_datetime(frame['origin'], utc=True)))
    if (position < 0).any():
        raise ValueError('Base forecast origins outside df_ts')
    base = np.full((len(table.time), H), np.nan)
    base[position] = frame[[f'b{h}' for h in range(1, H + 1)]].to_numpy(np.float64)
    return base, dict(file=file_identity(path), rows=len(frame), sets=frame['set'].value_counts().to_dict())


def save_checkpoint(path, stages, norm, state, epochs):
    """{stage}__{parameter} arrays of the prior and full networks, normalization, calibration and epoch count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f'{stage}__{name}': value for stage, weights in stages.items() for name, value in weights.items()}
    for name, value in (('norm', norm), ('calibration', state), ('epochs', epochs)):
        arrays[name] = np.frombuffer(json.dumps(value, allow_nan=False).encode(), np.uint8)
    temporary = path.with_name(path.stem + '.partial.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def load_checkpoint(path, stage):
    with np.load(path, allow_pickle=False) as arrays:
        prefix = f'{stage}__'
        weights = {name[len(prefix):]: arrays[name] for name in arrays.files if name.startswith(prefix)}
        norm, state = (json.loads(arrays[name].tobytes().decode()) for name in ('norm', 'calibration'))
    return weights, norm, state


def library_versions(spec):
    return versions(torch=torch.__version__, cuda=torch.version.cuda,
                    **{name: package_metadata.version(name) for name in spec.libraries})


def train_period(spec, table, period, seed, args, logs, device):
    stream, began = f's{seed}', time.perf_counter()
    train_index, query_index = table.origins(period, True), table.origins(period, False)
    if len(query_index) != QUERY_COUNTS[period]:
        raise ValueError('Unexpected official query grid')
    origins = table.time[train_index]
    reference, calibration, boundary = cal.prior_split(origins)
    full = np.arange(len(train_index))
    max_epochs = spec.training['max_epochs']
    if args.smoke:
        reference = reference[np.linspace(0, len(reference) - 1, min(len(reference), 512)).astype(int)]
        calibration = calibration[np.linspace(0, len(calibration) - 1, min(len(calibration), 128)).astype(int)]
        full = np.linspace(0, len(train_index) - 1, min(len(train_index), 640)).astype(int)
        query_index, max_epochs = query_index[:8], 2
    norm = fit_norm(table, train_index)
    base, base_record = load_base(spec, table, period)
    windows = Windows(table, norm, device, base)
    progress = dict(unit=0, units=2)
    record = lambda stage: dict(period=period, stage=stage, component=spec.family, **progress)  # noqa: E731
    logs(stream, 'period_start', period=period, seed=seed, smoke=args.smoke, training_cutoff=CUTOFFS[period],
         maturity_hours=PURGE_HOURS, train_origins=len(full), train_start=str(origins[full].min()),
         train_end=str(origins[full].max()), query_origins=len(query_index),
         query_start=str(table.time[query_index].min()), query_end=str(table.time[query_index].max()), **progress)

    prior_state, best_epoch, prior_history = train(spec, table, windows, train_index[reference],
                                                   train_index[calibration], max_epochs, seed, logs, stream,
                                                   record('prior'))
    model = load_state(network(spec), prior_state, device)
    prior = predict(spec, model, windows, train_index[calibration])
    truth = table.targets(train_index[calibration])
    fit_check = fit_report(spec, table, windows, model, train_index[reference], train_index[calibration], prior,
                           prior_history, best_epoch)
    offset, rows = cal.estimate_offset(prior, truth, spec.offset_window, spec.alpha, smoke=args.smoke)
    progress['unit'] += 1
    logs(stream, 'fit', **record('prior'), epochs=best_epoch, train_rows=len(reference),
         holdout_rows=len(calibration), holdout_mse=min(e['holdout_mse'] for e in prior_history if 'holdout_mse' in e),
         fit_seconds=round(time.perf_counter() - began, 3))
    state = dict(calibration_rows=len(calibration), reference_rows=len(reference),
                 reference_start=str(origins[reference].min()), reference_end=str(origins[reference].max()),
                 calibration_start=str(origins[calibration].min()), calibration_end=str(origins[calibration].max()),
                 split_boundary=str(boundary), purge_hours=PURGE_HOURS, policy=cal.POLICY,
                 window=spec.offset_window, alpha=spec.alpha, scope='horizon', applied=True,
                 offset=offset.tolist(), offset_rows=rows,
                 offset_start=str(origins[calibration][-rows]), offset_end=str(origins[calibration][-1]))

    stage_began = time.perf_counter()
    full_state, epochs, full_history = train(spec, table, windows, train_index[full], None, best_epoch, seed, logs,
                                             stream, record('full'))
    model = load_state(network(spec), full_state, device)
    tick = time.perf_counter()
    raw = predict(spec, model, windows, query_index)
    seconds = time.perf_counter() - tick
    progress['unit'] += 1
    logs(stream, 'fit', **record('full'), epochs=epochs, train_rows=len(full), query_rows=len(query_index),
         train_mse=full_history[-1]['train_mse'], fit_seconds=round(time.perf_counter() - stage_began, 3))

    checkpoint = args.weights / period / f's{seed}.npz'
    save_checkpoint(checkpoint, dict(prior=prior_state, full=full_state), norm, state, epochs)
    weights, saved_norm, _ = load_checkpoint(checkpoint, 'full')
    replay = predict(spec, load_state(network(spec), weights, device),
                     Windows(table, saved_norm, device, base), query_index)
    replay_error = float(np.max(np.abs(replay - raw)))
    if replay_error > 1e-3:
        raise ValueError(f'Saved network did not reproduce its forecasts (max abs error {replay_error})')
    prediction = cal.apply_offset(raw, offset)
    component = cal.component_values(raw)
    clipping = dict(component_cells=int(np.count_nonzero(raw != component)),
                    final_cells=int(np.count_nonzero(component + offset[None, :] != prediction)))
    metadata = dict(experiment=spec.experiment, family=spec.family, period=period, seed=seed, smoke=args.smoke,
                    objective=spec.training['loss'], hyperparameters=spec.hyperparameters,
                    persistence_residual=spec.persistence_residual, tail_weight=spec.tail_weight, fit_check=fit_check,
                    **(dict(swa_start=spec.swa_start) if spec.swa_start is not None else {}),
                    **(dict(base_forecast=base_record) if base_record else {}),
                    **(dict(lead_buckets={k: [v.start + 1, v.stop] for k, v in LEAD_BUCKETS.items()})
                       if spec.lead_checkpoints else {}),
                    **(dict(summary_constant=summary_features.CONSTANT, summary_varying=summary_features.VARYING)
                       if spec.summary else {}),
                    training=dict(spec.training, seed=seed), epochs=epochs, prior_history=prior_history,
                    full_history=full_history, parameter_count=int(sum(v.size for v in full_state.values())),
                    lookback_hours=LOOKBACK, channel_count=len(CHANNELS), channels=CHANNELS,
                    channel_groups=dict(ace=ACE, masks=list(MASKS), recurrence=REC, enlil=[*ENLIL, 'mask_enlil_past'],
                                        image=[*IMAGE, 'mask_image']),
                    normalization=norm, training_cutoff=CUTOFFS[period], maturity_hours=PURGE_HOURS,
                    train_origins=len(full), train_start=str(origins[full].min()),
                    train_end=str(origins[full].max()), query_origins=len(query_index),
                    query_start=str(table.time[query_index].min()), query_end=str(table.time[query_index].max()),
                    inference_seconds_per_fold=seconds / len(prediction), replay_max_abs_error=replay_error,
                    versions=library_versions(spec), device=torch.cuda.get_device_name(0),
                    input_file=file_identity(args.data), policy=cal.POLICY, git_commit=git_commit(),
                    frozen_calibration=state)
    write_json(args.out / 'logs' / f'{period}_s{seed}_metadata.json', metadata)
    if not args.board:
        # campaign runs (one seed per --out): period completion for the sunrun TUI
        write_json(args.out / f'{period}_metadata.json', metadata)
    logs(stream, 'period_complete', period=period, seed=seed, checkpoint=str(checkpoint), epochs=epochs,
         frozen_calibration=state, inference_seconds_per_fold=seconds / len(prediction), clipping=clipping,
         replay_max_abs_error=replay_error, end_to_end_seconds=round(time.perf_counter() - began, 3))
    return table.time[query_index], prediction


def predict_period(spec, table, period, seed, args, logs, device):
    """Inference only: the full network, normalization and frozen offset from weights/F{p}/s{seed}.npz."""
    stream, began = f's{seed}', time.perf_counter()
    query_index = table.origins(period, False)
    checkpoint = args.weights / period / f's{seed}.npz'
    weights, norm, state = load_checkpoint(checkpoint, 'full')
    if norm['channels'] != CHANNELS:
        raise ValueError('Checkpoint channels differ from this runner')
    model = load_state(network(spec), weights, device)
    tick = time.perf_counter()
    raw = predict(spec, model, Windows(table, norm, device, load_base(spec, table, period)[0]), query_index)
    seconds = time.perf_counter() - tick
    prediction = cal.apply_offset(raw, np.asarray(state['offset'], np.float64))
    logs(stream, 'predict_only', period=period, seed=seed, checkpoint=str(checkpoint),
         query_origins=len(prediction), offset=state['offset'], inference_seconds_per_fold=seconds / len(prediction),
         seconds=round(time.perf_counter() - began, 3))
    return table.time[query_index], prediction


def main(spec, doc, here):
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--experiment', choices=[spec.experiment], default=spec.experiment)
    parser.add_argument('--data', type=Path, default=DATA_TS)
    parser.add_argument('--observations', type=Path, help='default: the leaderboard dataset copy')
    parser.add_argument('--out', type=Path, help='run folder (default: research/result/<experiment>)')
    parser.add_argument('--weights', type=Path,
                        help='checkpoint folder (default: <experiment folder>/weights, or OUT/weights with --out)')
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    parser.add_argument('--threads', type=int, default=spec.threads)
    parser.add_argument('--seeds', type=int, nargs='+', choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--periods', nargs='+', choices=list(CUTOFFS), default=list(CUTOFFS))
    parser.add_argument('--predict-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or not torch.cuda.is_available():
        raise ValueError('DL runs are GPU only: start with CUDA_VISIBLE_DEVICES=<gpu index>; no CPU fallback')
    if args.threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(args.threads)
    device = torch.device('cuda')
    args.board = output_paths(args, here, spec.experiment)
    if args.smoke and (args.board or args.predict_only):
        raise ValueError('--smoke writes a throwaway run: give a separate --out and train')
    logs = run_logs(args, spec.experiment)
    launch(logs, args, mode='predict_only' if args.predict_only else ('smoke' if args.smoke else 'train'),
           versions=library_versions(spec), family=spec.family, hyperparameters=spec.hyperparameters,
           persistence_residual=spec.persistence_residual, tail_weight=spec.tail_weight, summary=spec.summary,
           swa_start=spec.swa_start, lead_checkpoints=spec.lead_checkpoints, base_forecast=spec.base_forecast,
           training=spec.training, lookback_hours=LOOKBACK, channels=CHANNELS, channel_count=len(CHANNELS),
           alpha=spec.alpha, offset_window=spec.offset_window, cutoffs=CUTOFFS, seeds=args.seeds,
           cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], gpu=torch.cuda.get_device_name(0))
    table = Table(args.data, logs, spec.summary)
    observations = None if args.smoke else observations_path(args.observations)
    frames = {}
    for seed in args.seeds:
        parts = {}
        for period in args.periods:
            run = predict_period if args.predict_only else train_period
            parts[period] = run(spec, table, period, seed, args, logs, device)
        frames[seed] = prediction_frame(parts)
        complete = not args.smoke and args.periods == list(CUTOFFS)
        scores = score(frames[seed], observations) if complete else None
        write_results(args.out / 'results', f's{seed}', frames[seed], scores)
        logs(f's{seed}', 'seed_complete', seed=seed, rows=len(frames[seed]),
             total_s=scores['competition']['total']['s'] if scores else None)
    if sorted(frames) == list(SEEDS):
        ensemble = mean3([frames[seed] for seed in SEEDS])
        scores = score(ensemble, observations) if observations is not None and args.periods == list(CUTOFFS) else None
        write_results(args.out / 'results', 'mean3', ensemble, scores)
        logs('mean3', 'mean3_complete', seeds=list(SEEDS), weights=[1 / 3] * 3, rows=len(ensemble),
             total_s=scores['competition']['total']['s'] if scores else None,
             periods={p: scores['competition'][p.lower()]['s'] for p in CUTOFFS} if scores else None)
    logs('run', 'complete', status='complete')
