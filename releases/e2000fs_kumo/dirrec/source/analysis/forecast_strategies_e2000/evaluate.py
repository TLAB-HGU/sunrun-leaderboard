"""Scoring only: selection-stage validation MSE and the leaderboard row (F1+F2, + MSE for targets >= 550 km/s).

This is the only module that reads the leaderboard snapshot; it never feeds training or selection.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data import H, LB

if str(LB) not in sys.path:
    sys.path.insert(0, str(LB))
from expos import evaluate as EV  # noqa: E402

HIGH = 550.
BANDS = EV.BANDS
CELLS = {'LLH': 'low_low_high', 'LHH': 'low_high_high', 'HLH': 'high_low_high', 'HHH': 'high_high_high'}
COMPARATORS = {'E1813': 3277.1166, 'E1789': 3180.2038}


def prediction_frame(origins, pred):
    pred = np.asarray(pred, dtype=float)
    if pred.shape != (len(origins), H) or not np.isfinite(pred).all():
        raise ValueError('invalid 72-horizon predictions')
    return pd.DataFrame({'origin_last_input_utc': np.repeat(pd.DatetimeIndex(origins).strftime('%Y-%m-%dT%H:%M:%SZ'), H),
                         'horizon_hours': np.tile(np.arange(1, H + 1), len(origins)),
                         'pred_kms': pred.ravel()})


def observed_mask(origins):
    """Observed (not forward-filled) target hours, from the hourly ACE series."""
    _, observed = EV._ace()
    o = pd.DatetimeIndex(origins)
    t = o.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    return observed.reindex(pd.DatetimeIndex(t.ravel(), tz='UTC')).fillna(False).to_numpy(bool).reshape(len(o), H)


def high_mse(err, y, mask=None, threshold=HIGH):
    """MSE over cells whose truth is >= threshold (optionally also inside mask)."""
    sel = y >= threshold
    if mask is not None:
        sel &= mask
    n = int(sel.sum())
    return {'mse': float(np.mean(err[sel] ** 2)) if n else np.nan,
            'bias': float(np.mean(err[sel])) if n else np.nan, 'n': n}


def internal(pred, Y, observed):
    """Validation metrics on the internal block (Y = filled truth, observed = not-filled mask)."""
    err = pred - Y
    by_h = np.array([np.mean(err[observed[:, h], h] ** 2) for h in range(H)])
    row = {'val_mse_obs': float(np.mean(err[observed] ** 2)), 'val_mse_all': float(np.mean(err ** 2)),
           'val_mse550_obs': high_mse(err, Y, observed)['mse']}
    for a, b in BANDS:
        row[f'val_h{a}-{b}'] = float(np.mean(err[:, a - 1:b][observed[:, a - 1:b]] ** 2))
    return row, by_h


WORKTREE = Path(__file__).resolve().parent / 'publication/worktree'  # leaderboard origin/main (F1+F2 rules)
STORE = 'tlabtlab/sunrun-lb-store'


def new_scoring():
    """The current upstream scorer (F1/F2 + >=550), loaded by path so the old local `scorer` stays untouched."""
    spec = importlib.util.spec_from_file_location('scoring_f1f2', WORKTREE / 'scorer/scoring.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Official:
    """Current leaderboard store + scorer loaded once; ``row(path)`` scores one prediction file.

    Files covering only F2 (reference runs) are scored on the F2 folds alone.
    """

    def __init__(self, revision=None):
        from huggingface_hub import HfApi, hf_hub_download
        self.revision = revision or HfApi().dataset_info(STORE).sha
        load = lambda n: pd.read_parquet(hf_hub_download(STORE, f'{n}.parquet', repo_type='dataset',
                                                         revision=self.revision))
        self.truth, self.folds, self.naive = load('truth'), load('folds'), load('naive')
        self.scoring = new_scoring()
        self.filled, self.observed = EV._ace()

    def _subset(self, pred):
        folds = self.folds[self.folds.origin_last_input_utc.isin(pred.origin_last_input_utc.unique())]
        keep = lambda d: d[d.origin_last_input_utc.isin(folds.origin_last_input_utc)]
        return keep(self.truth), folds, keep(self.naive)

    def row(self, path):
        path = Path(path)
        pred = pd.read_parquet(path)
        truth, folds, naive = self._subset(pred)
        periods = sorted(folds.period.unique())
        err_list = self.scoring.validate(pred, folds)
        if err_list:
            raise ValueError(f'{path.name}: {err_list[:3]}')
        sc = self.scoring.score(pred, truth, folds, naive)
        m = pred.merge(truth, on=self.scoring.KEY, validate='one_to_one').merge(
            folds[['origin_last_input_utc', 'period']], on='origin_last_input_utc')
        err, y = (m.pred_kms - m.target_kms).to_numpy(), m.target_kms.to_numpy()
        obs = m.was_missing.to_numpy() == 0
        row = {'periods': '+'.join(periods), 'mse': sc['mse'], 'mse_obs': sc['mse_observed'],
               'skill_vs_naive': sc.get('skill_vs_naive'), 'regime_balanced_mse': sc['regime_balanced_mse'],
               'csi': sc.get('csi'), 'timing_mae_h': sc.get('timing_mae_h')}  # upstream HSS-event diagnostics
        for tag, sel in [('', np.ones(len(m), bool))] + [(f'_{p.lower()}', (m.period == p).to_numpy()) for p in periods]:
            hi = high_mse(err[sel], y[sel])
            if not np.isclose(hi['mse'], sc[f'mse{tag}_ge550'], rtol=1e-9):
                raise ValueError('own >=550 MSE disagrees with the leaderboard scorer')
            row[f'mse{tag}'] = sc[f'mse{tag}']
            row.update({f'mse550{tag}': hi['mse'], f'bias550{tag}': hi['bias'], f'n550{tag}': hi['n'],
                        f'mse550_obs{tag}': high_mse(err[sel], y[sel], obs[sel])['mse']})
        if not np.isclose(float(np.mean(err ** 2)), sc['mse'], rtol=1e-9):
            raise ValueError('own MSE disagrees with the leaderboard scorer')
        for short, cell in CELLS.items():
            row[short] = sc['by_regime'][cell]['mse']
        frame = EV._load(path)
        for p in periods:
            o = pd.to_datetime(folds.loc[folds.period == p, 'origin_last_input_utc'], utc=True)
            ev = EV._events(frame[frame.origin.isin(o)], self.filled, self.observed)
            row.update({f'event_f1_{p}': ev['f1'], f'tp_{p}': ev['tp'], f'fp_{p}': ev['fp'], f'fn_{p}': ev['fn']})
        by_h = np.array([sc['mse_by_horizon'][h] for h in range(1, H + 1)])
        for a, b in BANDS:
            row[f'h{a}-{b}'] = float(by_h[a - 1:b].mean())
        if 'mse_f2' in row:
            for k, v in COMPARATORS.items():
                row[f'F2_vs_{k}_%'] = 100 * (row['mse_f2'] / v - 1)
        return row, by_h


def leaderboard(rows):
    """rows: {name: dict}; returns the leaderboard table sorted by official MSE."""
    order = ['family', 'periods', 'val_mse_obs', 'mse', 'mse_f1', 'mse_f2', 'mse550', 'mse550_f1', 'mse550_f2',
             'n550', 'bias550', 'bias550_f1', 'bias550_f2', 'mse550_obs', 'mse_obs', 'skill_vs_naive',
             'regime_balanced_mse', 'csi', 'timing_mae_h', *CELLS, 'event_f1_F1', 'event_f1_F2', 'tp_F2', 'fp_F2', 'fn_F2',
             *[f'h{a}-{b}' for a, b in BANDS], *[f'F2_vs_{k}_%' for k in COMPARATORS],
             'inference_seconds_per_fold', 'fit_seconds']
    t = pd.DataFrame.from_dict(rows, orient='index')
    t = t[[c for c in order if c in t.columns]].sort_values('mse')
    t.index.name = 'strategy'
    return t
