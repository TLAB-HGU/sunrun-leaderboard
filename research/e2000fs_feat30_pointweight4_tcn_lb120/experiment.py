"""e2000fs_feat30_pointweight4_tcn_lb120: causal TCN on 120 h of Top30 history, end to end.

A causal dilated TCN (receptive field 127 h, 64 channels) reads the last 120 h of the 30 features; one head conditions each of
the 72 horizons on the 9 known future features.
Training uses the point4 weights (target at or above 550 km/s counts four times, normalized per horizon) on z-scored speed.
An internal run on 2025-12-04..2026-02-22 picks the epoch count; F1 and F2 then refit that many epochs on targets before
2026-02-26 and 2026-05-29. The Top30 list is post-hoc, so it is official-exposed.

usage: experiment.py --out DIR [--smoke]
Data is only read from data-v2. Results depend on the GPU stack; the archive is results/forecast.npz (mse 6330.02).
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')  # GPU0 is reserved

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATA = Path('/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/e2000fs-lag0')
FEATURES = ['v_lag0', 'rec_mean', 'enlil_kp', 'ch_ch_t035_meridian30_equatorial_area', 'ace_bt_nt_last', 'ace_speed_kms_last',
            'ch_ch_t035_equatorial_area', 'ch_ch_t035_meridian15_equatorial_area', 'ace_bt_nt_m6', 'rec26',
            'strip_v450_logmed', 'strip_v350_logmed', 'ch_ch_t055_equatorial_area', 'rec27', 'ace_density_cm3_last',
            'ch_ch_t055_meridian30_equatorial_area', 'ace_by_gsm_nt_m24', 'ace_density_cm3_m6',
            'ace_temperature_k_last', 'enlil_n_pending', 'ace_temperature_k_m24', 'strip_v750_logmed',
            'ch_ch_t055_meridian15_area', 'ch_ch_t035_top2_area', 'strip_v550_dark', 'strip_v650_dark',
            'ace_sis_gt10_flux_last', 'ace_epam_p_channel3_last', 'img_age_h', 'rec28']
FIRST_HOUR = '2022-09-26 01:00:00+00:00'  # first row of hourly_history.npy
LOOKBACK = 120
KERNEL, DILATIONS, WIDTH = 3, (1, 2, 4, 8, 16, 32), 64
BATCH_SIZE, MAX_EPOCHS, PATIENCE = 256, 100, 10
SEED = 42
STAGES = (('f1', '2026-02-26'), ('official', '2026-05-29'))  # forecast grid, training target cutoff


def targets(speed, origins):
    times = origins.values[:, None] + np.arange(1, 73).astype('timedelta64[h]')[None, :]
    return speed.reindex(pd.DatetimeIndex(times.ravel(), tz='UTC')).to_numpy().reshape(-1, 72)


class CausalBlock(nn.Module):
    """One left-padded convolution; output t only uses inputs at or before t."""

    def __init__(self, cin, cout, dilation):
        super().__init__()
        self.pad = (KERNEL - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, KERNEL, dilation=dilation)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(nn.functional.pad(x, (self.pad, 0))))


class TCN(nn.Module):
    """Encode each origin once, then condition its 72 forecasts on the known future inputs and the lead."""

    def __init__(self):
        super().__init__()
        widths = [30] + [WIDTH] * len(DILATIONS)
        self.encoder = nn.Sequential(*[CausalBlock(widths[i], widths[i + 1], d) for i, d in enumerate(DILATIONS)])
        self.head = nn.Sequential(nn.Linear(WIDTH + 9 + 1, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer('horizon', torch.arange(1, 73, dtype=torch.float32).view(1, 72, 1) / 72)

    def forward(self, history, future):
        context = self.encoder(history.transpose(1, 2))[:, :, -1][:, None, :].expand(-1, 72, -1)
        horizon = self.horizon.expand(len(history), -1, -1)
        return self.head(torch.cat((context, future, horizon), dim=-1)).squeeze(-1)


def normalization(history, positions, future, y):
    """Medians, means and stds from the hours inside the training windows and the training future inputs only."""
    used = np.zeros(len(history), dtype=bool)
    for end in positions:
        used[end - LOOKBACK + 1:end + 1] = True
    stats = {}
    for tag, array in (('history', history[used]), ('future', future.reshape(-1, 9))):
        median = np.nanmedian(array, axis=0)
        filled = np.where(np.isfinite(array), array, median)
        stats[tag] = dict(median=median.astype(np.float32), mean=filled.mean(axis=0, dtype=np.float64).astype(np.float32),
                          std=np.maximum(filled.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32))
    stats['y_mean'], stats['y_std'] = float(y.mean()), max(float(y.std()), 1e-6)
    stats['weight_mean'] = np.where(y >= 550., 4., 1.).mean(axis=0).astype(np.float32)
    return stats


def transform(array, stats):
    return ((np.where(np.isfinite(array), array, stats['median']) - stats['mean']) / stats['std']).astype(np.float32)


class Windows(Dataset):
    def __init__(self, history, positions, future, y, stats):
        self.history, self.positions, self.future, self.y, self.stats = history, positions, future, y, stats

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        end = int(self.positions[i])
        h = transform(self.history[end - LOOKBACK + 1:end + 1], self.stats['history'])
        f = transform(self.future[i], self.stats['future'])
        actual = self.y[i]
        target = ((actual - self.stats['y_mean']) / self.stats['y_std']).astype(np.float32)
        return h, f, target, (np.where(actual >= 550., 4., 1.) / self.stats['weight_mean']).astype(np.float32)


def loader(windows, shuffle=False, generator=None):
    return DataLoader(windows, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator, num_workers=0, pin_memory=True)


def weighted_mse(pred, target, weights):
    return (weights * (pred - target).square()).mean()


def train(windows, val_windows, epochs, device, curve, new_net):
    """Seeded training; with ``val_windows`` it stops after PATIENCE epochs without improvement and returns the best epoch."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = torch.Generator().manual_seed(SEED)
    net = new_net().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    batches = loader(windows, True, generator)
    best, best_epoch, wait, rows = float('inf'), 0, 0, []
    for epoch in range(1, epochs + 1):
        if val_windows is not None and wait >= PATIENCE:
            break
        net.train()
        total = count = 0
        for arrays in batches:
            hist, future, target, weights = [a.to(device, non_blocking=True) for a in arrays]
            optimizer.zero_grad(set_to_none=True)
            loss = weighted_mse(net(hist, future), target, weights)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * target.numel()
            count += target.numel()
        row = dict(epoch=epoch, train_weighted_mse=total / count * windows.stats['y_std'] ** 2)
        if val_windows is not None:
            net.eval()
            total = count = 0
            with torch.inference_mode():
                for arrays in loader(val_windows):
                    hist, future, target, weights = [a.to(device, non_blocking=True) for a in arrays]
                    total += float(weighted_mse(net(hist, future), target, weights)) * target.numel()
                    count += target.numel()
            row['validation_weighted_mse'] = total / count * windows.stats['y_std'] ** 2
            if row['validation_weighted_mse'] < best:
                best, best_epoch, wait = row['validation_weighted_mse'], epoch, 0
            else:
                wait += 1
        rows.append(row)
        print(json.dumps(row), flush=True)
    pd.DataFrame(rows).to_csv(curve, index=False)
    return net, best_epoch


def main(out, smoke):
    # Load
    names = json.loads((DATA / 'names.json').read_text())
    train_grid = dict(np.load(DATA / 'train.npz'))
    grids = {stage: dict(np.load(DATA / f'{stage}.npz')) for stage, _ in STAGES}
    speed = pd.read_parquet(DATA / 'speed.parquet')['speed']
    history = np.load(DATA / 'hourly_history.npy', mmap_mode='r')
    hourly = pd.date_range(FIRST_HOUR, periods=len(history), freq='h', tz='UTC')
    origins = pd.DatetimeIndex(train_grid['origins'], tz='UTC')
    y = targets(speed, origins)
    varying = [n for n in FEATURES if n in names['varying']]
    columns = [names['varying'].index(n) for n in varying]
    future = np.asarray(train_grid['E'][:, :, columns], dtype=np.float32)
    positions = hourly.searchsorted(origins)
    device = torch.device('cuda')
    epochs = 2 if smoke else MAX_EPOCHS
    step = 50 if smoke else 1
    new_net = TCN

    # Pick the epoch count on the internal window
    start = pd.Timestamp('2025-12-04', tz='UTC')
    fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < start)[::step]
    val = np.flatnonzero((origins >= start) & (origins <= pd.Timestamp('2026-02-22T23:00:00Z')))[::step]
    stats = normalization(history, positions[fit], future[fit], y[fit])
    _, best_epoch = train(Windows(history, positions[fit], future[fit], y[fit], stats),
                          Windows(history, positions[val], future[val], y[val], stats), epochs, device, out / 'internal_loss.csv', new_net)

    # Refit that many epochs and forecast
    frames, errors = [], {}
    for stage, cutoff in STAGES:
        fit = np.flatnonzero(origins + pd.Timedelta(hours=72) < pd.Timestamp(cutoff, tz='UTC'))[::step]
        stats = normalization(history, positions[fit], future[fit], y[fit])
        net, _ = train(Windows(history, positions[fit], future[fit], y[fit], stats), None, best_epoch, device, out / f'{stage}_loss.csv', new_net)
        torch.save(net.state_dict(), out / f'{stage}.pt')
        grid = grids[stage]
        query = pd.DatetimeIndex(grid['origins'], tz='UTC')
        windows = Windows(history, hourly.searchsorted(query), np.asarray(grid['E'][:, :, columns], dtype=np.float32), np.zeros((len(query), 72)), stats)
        net.eval()
        parts = []
        with torch.inference_mode():
            for arrays in loader(windows):
                hist, future_query = [a.to(device, non_blocking=True) for a in arrays[:2]]
                parts.append(net(hist, future_query).cpu().numpy() * stats['y_std'] + stats['y_mean'])
        pred = np.clip(np.concatenate(parts).astype(np.float64), 1., 3000.)
        print(f'{stage}: fit {len(fit)} origins, {best_epoch} epochs', flush=True)
        errors[stage] = (pred - targets(speed, query)) ** 2
        frames.append(pd.DataFrame({'origin_last_input_utc': np.repeat(query.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                                    'horizon_hours': np.tile(np.arange(1, 73), len(query)),
                                    'pred_kms': pred.ravel()}))

    # Score: the hourly speed series is the official truth series
    frame = pd.concat(frames, ignore_index=True)
    assert len(frame) == 357840 and np.isfinite(frame.pred_kms).all()
    frame.to_parquet(out / 'predictions.parquet', index=False)
    scores = dict(mse=float(np.concatenate([e.ravel() for e in errors.values()]).mean()),
                  mse_f1=float(errors['f1'].mean()), mse_f2=float(errors['official'].mean()))
    (out / 'scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    print(json.dumps(scores), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    main(args.out, args.smoke)
