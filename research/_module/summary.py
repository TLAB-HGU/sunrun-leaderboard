"""The summary features of the XGBoost rows as extra inputs of the sequence models (research/_module/seq.py).

Same 31 source features as vanila_mse, rebuilt from the hourly table data-v2/input/df_ts.parquet with the definitions of
data-v2/notebooks/raw_to_input_df.ipynb and the XGBoost table policy (forward fill along the origins, leading NaN kept):
  constant (23 per origin)    v_lag0/24/48/72, ACE last / 6 h / 24 h means, SUVI dark-area features of the frame in the
                              origin bin, img_age_h, ENLIL n_pending and kp
  per lead (8 x 72 per origin) rec26/27/28 = speed bin at target time - 26/27/28 d, their mean, and the longitude strips
                              a wind of 350/450/550/750 km/s arriving at the target time left from (15 deg cells, lat1)
Densities, temperatures and particle fluxes are passed as log10 (a monotone change of scale for the network only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import H

LAGS = (0, 24, 48, 72)
ACE_LAST = ['ace_speed_kms', 'ace_density_cm3', 'ace_temperature_k', 'ace_bt_nt', 'ace_epam_p_channel3', 'ace_sis_gt10_flux']
ACE_MEAN = {'ace_density_cm3_m6': ('ace_density_cm3', 6), 'ace_temperature_k_m24': ('ace_temperature_k', 24),
            'ace_bt_nt_m6': ('ace_bt_nt', 6), 'ace_by_gsm_nt_m24': ('ace_by_gsm_nt', 24)}
AREAS = ['suvi_t035_top2_area', 'suvi_t035_meridian15_equatorial_area', 'suvi_t035_meridian30_equatorial_area',
         'suvi_t055_meridian15_area', 'suvi_t055_meridian30_equatorial_area', 'suvi_t055_equatorial_area']
CONSTANT = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_temperature_k_last',
            'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last',
            'ace_sis_gt10_flux_last', *AREAS, 'img_age_h', 'enlil_n_pending', 'enlil_kp', 'v_lag24', 'v_lag48', 'v_lag72']
REC_DAYS = (26, 27, 28)
STRIPS = {'strip_v350_logmed': (350, 'log_median'), 'strip_v450_logmed': (450, 'log_median'),
          'strip_v550_dark': (550, 'dark_fraction'), 'strip_v750_logmed': (750, 'log_median')}
VARYING = ['rec26', 'rec27', 'rec28', 'rec_mean', *STRIPS]
LOG10 = ['ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'ace_temperature_k_m24',
         'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last']
OMEGA, D_KM, CELL_WIDTH, STRIP_HALF = 0.55, 147858620.7, 15.0, 5.0
CELLS = {stat: [f'suvi_lon{i:02d}_lat1_{stat}' for i in range(12)] for stat in ('log_median', 'dark_fraction')}
COLUMNS = sorted({*ACE_LAST, *(source for source, _ in ACE_MEAN.values()), *AREAS, 'img_age_h', 'enlil_n_pending',
                  'enlil_kp', *CELLS['log_median'], *CELLS['dark_fraction']})


def forward_fill(values, start):
    """Forward fill along time from the first origin on (values before it stay as they are)."""
    tail = values[start:]
    flat = tail.reshape(len(tail), -1)
    valid = ~np.isnan(flat)
    previous = np.maximum.accumulate(np.where(valid, np.arange(len(flat))[:, None], -1), axis=0)
    filled = np.where(previous >= 0, np.take_along_axis(flat, np.maximum(previous, 0), 0), np.nan)
    values[start:] = filled.reshape(tail.shape)
    return values


def strips(cells, age, speed):
    """(time x 72) overlap-weighted mean of the 12 lat1 cells over the 10 deg strip of one wind speed."""
    lead = np.arange(1, H + 1, dtype=np.float64)
    edges = -90.0 + CELL_WIDTH * np.arange(13)
    output = np.full((len(age), H), np.nan, np.float32)
    for begin in range(0, len(age), 4096):
        part = slice(begin, begin + 4096)
        center = OMEGA * (D_KM / speed / 3600.0 - (lead[None, :] + age[part, None]))
        weight = np.clip(np.minimum(center[..., None] + STRIP_HALF, edges[1:])
                         - np.maximum(center[..., None] - STRIP_HALF, edges[:-1]), 0, None) / (2 * STRIP_HALF)
        total = weight.sum(-1)
        value = (weight * cells[part, None, :]).sum(-1)
        output[part] = np.where(total > 0, value / np.where(total > 0, total, 1), np.nan)
    return output


def build(path, first_origin):
    """Unscaled features on the hourly axis of df_ts: constant (time x 23) and per lead (time x 72 x 8)."""
    frame = pd.read_parquet(path, columns=COLUMNS).astype(np.float64)
    speed = frame['ace_speed_kms']
    constant = {f'v_lag{lag}': speed.shift(lag) for lag in LAGS}
    constant.update({f'{name}_last': frame[name] for name in ACE_LAST})
    constant.update({name: frame[source].rolling(window, min_periods=1).mean() for name, (source, window) in ACE_MEAN.items()})
    constant.update({name: frame[name] for name in (*AREAS, 'img_age_h', 'enlil_n_pending', 'enlil_kp')})
    constant = np.stack([constant[name].to_numpy() for name in CONSTANT], 1)
    position = np.arange(len(frame))[:, None] + np.arange(1, H + 1)[None, :]
    series = speed.to_numpy()
    recurrence = []
    for days in REC_DAYS:
        source = position - days * 24          # speed bin at target time - days: always before the origin
        recurrence.append(np.where(source >= 0, series[np.clip(source, 0, len(series) - 1)], np.nan))
    counts = np.sum([~np.isnan(r) for r in recurrence], 0)
    mean = np.where(counts > 0, np.nansum(recurrence, 0) / np.maximum(counts, 1), np.nan)
    age = frame['img_age_h'].to_numpy()
    varying = [*recurrence, mean, *(strips(frame[CELLS[stat]].to_numpy(), age, speed_kms)
                                    for speed_kms, stat in STRIPS.values())]
    varying = np.stack(varying, 2)
    constant, varying = forward_fill(constant, first_origin), forward_fill(varying, first_origin)
    return constant.astype(np.float32), varying.astype(np.float32)


def scaled(constant):
    """log10 of the positive heavy-tailed constants (non-positive -> NaN); other columns unchanged."""
    constant = constant.astype(np.float64)
    for name in LOG10:
        column = CONSTANT.index(name)
        constant[:, column] = np.log10(np.where(constant[:, column] > 0, constant[:, column], np.nan))
    return constant
