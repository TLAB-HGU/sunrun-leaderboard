"""Preregistered train-only adaptations; no evaluation-dependent tuning."""
import numpy as np
import pandas as pd

KEYS = ('mse', 'mse_lt550', 'mse_ge550')
LANES = ('hyper_gain', 'gaussian', 'mc_calibration')


def mature_indices(origins, cutoff):
    return np.flatnonzero(origins + pd.Timedelta(hours=72) < cutoff)


def decoder(bundle, names):
    anchor = np.asarray(bundle['C'][:, names['constant'].index('v_lag0')], dtype=float)
    if not np.isfinite(anchor).all():
        raise ValueError('Nonfinite causal anchor')
    delta = np.asarray(bundle['E'][:, :, names['varying'].index('rec_mean')], dtype=float) - anchor[:, None]
    gain = np.where(delta < 0, -1., 1.) * np.clip(np.abs(delta), 50., 300.)
    missing = ~np.isfinite(delta)
    gain[missing] = 100.
    return anchor, gain, int(missing.sum())


def gaussian_weights(target, pilot_prediction):
    loss = np.square(np.asarray(target) - np.asarray(pilot_prediction))
    if not np.isfinite(loss).all():
        raise ValueError('Nonfinite pilot loss')
    scale = loss.std()
    z = (loss - loss.mean()) / scale if scale > 0 else np.zeros_like(loss)
    raw = np.maximum(.1, np.exp(-.5 * np.square(z)))
    return raw / raw.mean(), dict(loss_mean=float(loss.mean()), loss_std=float(scale),
                                 floor_count=int((raw == .1).sum()))


def calibration_split(origins, train):
    boundary = int(np.floor(.8 * len(train)))
    calibration = train[boundary:]
    if not len(calibration):
        raise ValueError('Empty calibration')
    prefix = train[:boundary]
    prefix = prefix[origins[prefix] + pd.Timedelta(hours=72) < origins[calibration[0]]]
    if not len(prefix):
        raise ValueError('Empty purged prefix')
    return prefix, calibration


def gate(candidate, baseline, improvements=False):
    ratios = (.99, 1., .95) if improvements else (1., 1., 1.)
    return all(candidate.get(k) is not None and baseline.get(k) is not None
               and np.isfinite(candidate[k]) and np.isfinite(baseline[k])
               and candidate[k] <= baseline[k] * ratio for k, ratio in zip(KEYS, ratios))


def development_gate(candidate, baseline):
    return (all(gate(candidate[f'dev{i}'], baseline[f'dev{i}']) for i in range(3))
            and gate(candidate['pooled'], baseline['pooled'], True))
