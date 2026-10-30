"""E2000 inputs re-timed to the origin hour (no ACE +2h lag), prefilled, and reshaped for block strategies.

Training code here never reads leaderboard truth/folds/naive; targets come from the hourly ACE
``filled_speed_kms`` series, which is the same series the official truth is built from.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path('/home/t-lab01/sunrun/leaderboard')
ROOT = LB.parent
if str(LB) not in sys.path:
    sys.path.insert(0, str(LB))

from experiments.e2000 import inputs as E2  # noqa: E402
from experiments.e2000 import prepare as P  # noqa: E402

HERE = Path(__file__).resolve().parent
RELOCATED = ROOT / 'research/e0472-multioutput-s42-20261026/relocated-v1'
TRAIN_FEATURES = RELOCATED / 'train_complete_causal.parquet'
OFFICIAL_FEATURES = RELOCATED / 'official_causal.parquet'
# E2000 F1 grid built by the E2000 publication session (read-only input; same 111 features, lag 2h).
F1_FEATURES = ROOT / 'research/e2000/publication/f1/inputs/features_f1.parquet'
FEATURES_JSON = LB / 'experiments/cumulative_lane12_20261024/materialized_v3/features_v3.json'

H = 72
LAG = 24  # recent speed window handed to every block model
CUTOFF, TRAIN_END = E2.CUTOFF, E2.TRAIN_END
VALID_START, VALID_END = E2.VALID_START, E2.VALID_END
VARYING = E2.VARYING
OFFICIAL_FIRST = pd.Timestamp('2026-05-31T23:00:00Z')
OFFICIAL_LAST = pd.Timestamp('2026-09-26T23:00:00Z')
F1_FIRST = pd.Timestamp('2026-02-28T23:00:00Z')
F1_LAST = pd.Timestamp('2026-05-28T23:00:00Z')
# Leaderboard rule (2026-10-26): F1 predictions come from models trained and tuned before F1_TRAIN_END.
F1_TRAIN_END = pd.Timestamp('2026-02-26T00:00:00Z')
# Selection stage (sigma / Stratify choice) lies wholly before F1_TRAIN_END and serves both periods.
SELECT_TRAIN_END = pd.Timestamp('2025-12-01T00:00:00Z')
SELECT_VALID_START = pd.Timestamp('2025-12-04T00:00:00Z')
SELECT_VALID_END = pd.Timestamp('2026-02-22T23:00:00Z')
GRIDS = (('train', TRAIN_FEATURES), ('f1', F1_FEATURES), ('official', OFFICIAL_FEATURES))


def ace_columns(features):
    return [c for c in features if c.startswith('ace_') or c in ('speed_d24', 'epam_p1_d6')]


def speed_columns(features):
    return [c for c in features if c.startswith('v_')]


def feature_names():
    features = json.loads(FEATURES_JSON.read_text())['features']
    constant = [c for c in features if c != 'h' and c not in VARYING]
    return features, constant, list(VARYING)


def retime(origins, p, ace, features, lag_hours=0.):
    """ACE/plasma/speed families evaluated at ``origin - lag_hours`` with the production functions."""
    o = pd.DatetimeIndex(origins)
    at = o - pd.Timedelta(hours=lag_hours)
    af, sf = p.HR.ace_features(at), p.ia.speed_features(at, ace)
    out = pd.DataFrame(index=o)
    for c in ace_columns(features):
        out[c] = af[c].to_numpy()
    for c in speed_columns(features):
        out[c] = sf[c].to_numpy()
    return out


class Bundle:
    """One origin grid: C (n, n_const), E (n, 72, n_vary), past speeds (n, LAG) ending at the origin."""

    def __init__(self, origins, C, E, past, constant, varying):
        self.origins, self.C, self.E, self.past = pd.DatetimeIndex(origins), C, E, past
        self.constant, self.varying = constant, varying

    def __len__(self):
        return len(self.origins)

    def take(self, index):
        index = np.asarray(index)
        return Bundle(self.origins[index], self.C[index], self.E[index], self.past[index],
                      self.constant, self.varying)


def past_window(series, origins, lag=LAG):
    """Speeds at origin-lag+1 .. origin (inclusive): never after the origin hour."""
    o = pd.DatetimeIndex(origins)
    t = o.values[:, None] + np.arange(-lag + 1, 1).astype('timedelta64[h]')[None, :]
    return series.reindex(pd.DatetimeIndex(t.ravel(), tz='UTC')).to_numpy(np.float32).reshape(len(o), lag)


def future_targets(series, origins):
    """Hourly targets o+1..o+72 (filled series, the official truth convention); NaN outside the series."""
    o = pd.DatetimeIndex(origins)
    t = o.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]
    return series.reindex(pd.DatetimeIndex(t.ravel(), tz='UTC')).to_numpy(float).reshape(len(o), H)


def target_times(origins):
    o = pd.DatetimeIndex(origins).tz_convert('UTC').tz_localize(None)
    return o.values[:, None] + np.arange(1, H + 1).astype('timedelta64[h]')[None, :]


def speed_series(p):
    """Hourly filled ACE speed, prefilled so every hour has a value from the past only."""
    s = p.A.filled_hourly()
    return s.ffill()


def _grid(path, features, constant):
    frame = E2.keyed(pd.read_parquet(path))
    origins = pd.DatetimeIndex(frame.origin.unique())
    if len(frame) != len(origins) * H:
        raise ValueError('incomplete feature grid: ' + str(path))
    C = frame.loc[frame.h == 1, constant].to_numpy(np.float32)
    E = np.stack([frame[c].to_numpy(np.float32).reshape(len(origins), H) for c in VARYING], axis=-1)
    return origins, frame, C, E


def prefill(C, E):
    """Forward-fill along the origin axis (C per column, E per horizon/column); leading NaN stay NaN."""
    C = pd.DataFrame(C).ffill().to_numpy(np.float32)
    n, h, k = E.shape
    E = pd.DataFrame(E.reshape(n, h * k)).ffill().to_numpy(np.float32).reshape(n, h, k)
    return C, E


def build(cache=None, parity_origins=256):
    """Return (train, f1, official Bundles, speed series, receipt). Cached as npz when ``cache`` is given."""
    features, constant, varying = feature_names()
    root = P.source_root()
    with P.relocated_builder(root) as p:
        ace = p.A.filled_hourly()
        series = speed_series(p)
        out, receipt = {}, {'lag_hours': 0, 'prefill': 'ffill along origin', 'source_root': str(root),
                            'inputs': {k: E2.reference(v) for k, v in
                                       GRIDS + (('features', FEATURES_JSON),)}}
        retimed = ace_columns(features) + speed_columns(features)
        for name, path in GRIDS:
            origins, frame, C, E = _grid(path, features, constant)
            # Parity: the production functions at lag 2h must reproduce the frozen parquet columns.
            pick = np.linspace(0, len(origins) - 1, min(parity_origins, len(origins))).astype(int)
            old = frame.loc[frame.h == 1, retimed].to_numpy(np.float64)[pick]
            new = retime(origins[pick], p, ace, features, lag_hours=2.).to_numpy(np.float64)
            ok = np.isclose(old, new, rtol=1e-5, atol=1e-4, equal_nan=True)
            receipt[f'{name}_lag2_parity'] = {'origins': len(pick), 'mismatch': int((~ok).sum())}
            if not ok.all():
                raise ValueError(f'{name}: lag-2h parity failed on {int((~ok).sum())} values')
            lag0 = retime(origins, p, ace, features, lag_hours=0.)
            for c in retimed:
                C[:, constant.index(c)] = lag0[c].to_numpy(np.float32)
            out[name] = (origins, C, E)
    # Prefill along the joined origin timeline (stable sort: train/F1 grids overlap in time).
    names = [name for name, _ in GRIDS]
    origins = pd.DatetimeIndex(np.concatenate([out[n][0].asi8 for n in names]), tz='UTC')
    order = np.argsort(origins.asi8, kind='stable')
    C_all, E_all = prefill(np.concatenate([out[n][1] for n in names])[order],
                           np.concatenate([out[n][2] for n in names])[order])
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    C_all, E_all = C_all[back], E_all[back]
    bundles, start = {}, 0
    for n in names:
        o = out[n][0]
        sl = slice(start, start + len(o))
        bundles[n] = Bundle(o, C_all[sl], E_all[sl], past_window(series, o), constant, varying)
        start += len(o)
    train, f1, official = bundles['train'], bundles['f1'], bundles['official']
    for b, n, first, last in ((official, 2833, OFFICIAL_FIRST, OFFICIAL_LAST), (f1, 2137, F1_FIRST, F1_LAST)):
        if len(b) != n or b.origins[0] != first or b.origins[-1] != last:
            raise ValueError('evaluation origin grid mismatch')
    receipt.update(train_origins=len(train), f1_origins=len(f1), official_origins=len(official),
                   nan_after_prefill={f'{n}_{k}': int(np.isnan(getattr(bundles[n], k)).sum())
                                      for n in names for k in ('C', 'E')})
    if cache is not None:
        save(cache, bundles, series, receipt)
    return train, f1, official, series, receipt


def save(path, bundles, series, receipt):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, b in bundles.items():
        np.savez(path / f'{name}.npz', origins=b.origins.asi8, C=b.C, E=b.E, past=b.past)
    series.to_frame('speed').to_parquet(path / 'speed.parquet')
    ref = bundles['train']
    (path / 'names.json').write_text(json.dumps({'constant': ref.constant, 'varying': ref.varying}))
    E2.atomic_json(path / 'receipt.json', receipt)


def load(path):
    path = Path(path)
    names = json.loads((path / 'names.json').read_text())
    bundles = []
    for name, _ in GRIDS:
        z = np.load(path / f'{name}.npz')
        bundles.append(Bundle(pd.DatetimeIndex(z['origins'], tz='UTC'), z['C'], z['E'], z['past'],
                              names['constant'], names['varying']))
    series = pd.read_parquet(path / 'speed.parquet')['speed']
    return (*bundles, series, json.loads((path / 'receipt.json').read_text()))


def internal_split(train, series):
    """Internal fit origins (all targets < TRAIN_END) and validation origins (72h purge before them)."""
    tt = target_times(train.origins)
    fit = tt[:, -1] < TRAIN_END.tz_localize(None).to_datetime64()
    val = (train.origins >= VALID_START) & (train.origins <= VALID_END)
    if (tt[val] >= CUTOFF.tz_localize(None).to_datetime64()).any():
        raise ValueError('validation targets must precede the final cutoff')
    return np.flatnonzero(fit), np.flatnonzero(val)


def _before(train, end):
    return np.flatnonzero(target_times(train.origins)[:, -1] < end.tz_localize(None).to_datetime64())


def final_split(train):
    """F2 models: every target before the official cutoff."""
    return _before(train, CUTOFF)


def f1_split(train):
    """F1 models: every target before 2026-02-26 (leaderboard F1 training cutoff)."""
    return _before(train, F1_TRAIN_END)


def select_split(train):
    """Selection stage: fit targets < 2025-12-01, validation origins 2025-12-04..2026-02-22 (targets < F1 cutoff)."""
    fit = _before(train, SELECT_TRAIN_END)
    val = np.flatnonzero((train.origins >= SELECT_VALID_START) & (train.origins <= SELECT_VALID_END))
    if (target_times(train.origins[val]) >= F1_TRAIN_END.tz_localize(None).to_datetime64()).any():
        raise ValueError('selection targets must precede the F1 training cutoff')
    return fit, val
