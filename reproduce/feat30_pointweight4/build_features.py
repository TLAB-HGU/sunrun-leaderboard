"""Raw ACE parquet + frozen 111-feature grids -> block-model feature cache (lag 0, prefilled).

The three grid parquets (train / F1 / official) hold the E0472 111 causal features built with the ACE +2h
input delay. This step re-evaluates the ACE, plasma and speed families at the origin hour itself (no
delay, rule of 2026-10-29) from the raw ACE tables, checks that the same functions at lag 2h reproduce the
frozen parquet columns, forward-fills along the origin axis and writes ``train/f1/official.npz``,
``speed.parquet`` and ``names.json``. The SUVI / ENLIL / coronal-hole columns stay as frozen in the grids.

    python reproduce/feat30_pointweight4/build_features.py --out reproduce/outputs/pointweight4_cache
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402

H = 72
LAG = 24  # recent speed window handed to every block model
VARYING = ('rec26', 'rec27', 'rec28', 'rec_mean') + tuple(
    f'strip_v{v}_{s}' for v in (350, 450, 550, 650, 750) for s in ('logmed', 'dark', 'cover')) + ('enlil_dt_target',)
GRIDS = (('train', 'train_complete_causal.parquet', None),
         ('f1', 'features_f1.parquet', (2137, '2026-02-28T23:00:00Z', '2026-05-28T23:00:00Z')),
         ('official', 'official_causal.parquet', (2833, '2026-05-31T23:00:00Z', '2026-09-26T23:00:00Z')))


# --- raw ACE feature functions (headroom_audit.headroom / info_ablation, paths made relative) -----------

def hourly(df, col='timestamp_utc'):
    df = df.copy()
    df.index = pd.to_datetime(df.pop(col), utc=True).dt.floor('h')
    return df[~df.index.duplicated(keep='last')].sort_index()


def filled_hourly():
    ace = pd.read_parquet(paths.data('store/ch-v1/ace.parquet'))
    ace['timestamp_utc'] = pd.to_datetime(ace['timestamp_utc'], utc=True)
    s = ace.set_index('timestamp_utc')['filled_speed_kms'].sort_index()
    s = s[~s.index.duplicated(keep='last')].sort_index()
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='h', tz='UTC'))


def ace_features(origins):
    parts = sorted(glob.glob(str(paths.data('store/ch-breakthrough-v2/upload/raw-ace-snapshot') / 'year=*/part.parquet')))
    raw = hourly(pd.concat(pd.read_parquet(f) for f in parts))
    cols = ['ace_speed_kms', 'ace_density_cm3', 'ace_temperature_k', 'ace_bt_nt', 'ace_bz_gsm_nt', 'ace_by_gsm_nt']
    logs = ['ace_epam_e_channel1', 'ace_epam_e_channel2', 'ace_epam_p_channel1', 'ace_epam_p_channel3',
            'ace_epam_p_channel5', 'ace_sis_gt10_flux']
    x = raw[cols].astype(float)
    x[logs] = np.log10(raw[logs].astype(float).clip(lower=1e-3))
    x = x.reindex(pd.date_range(x.index.min(), x.index.max(), freq='h', tz='UTC'))
    f = {}
    for c in x.columns:  # rolling windows end at the origin hour: causal
        f[f'{c}_last'] = x[c].ffill(limit=6)
        f[f'{c}_m6'] = x[c].rolling(6, min_periods=1).mean()
        f[f'{c}_m24'] = x[c].rolling(24, min_periods=1).mean()
    f['speed_d24'] = f['ace_speed_kms_m6'] - f['ace_speed_kms_m6'].shift(24)
    f['epam_p1_d6'] = f['ace_epam_p_channel1_m6'] - f['ace_epam_p_channel1_m6'].shift(6)
    return pd.DataFrame(f).reindex(origins)


def speed_features(origins, truth_filled):
    s = truth_filled
    f = {f'v_lag{l}': s.shift(l) for l in (0, 1, 3, 6, 12, 24, 48, 72, 120)}
    for w in (6, 24, 72):
        f[f'v_m{w}'] = s.rolling(w, min_periods=1).mean()
        f[f'v_sd{w}'] = s.rolling(w, min_periods=2).std()
    return pd.DataFrame(f).reindex(origins)


# --- grid handling (analysis/forecast_strategies_e2000/data.py) ---------------------------------------

def ace_columns(features):
    return [c for c in features if c.startswith('ace_') or c in ('speed_d24', 'epam_p1_d6')]


def speed_columns(features):
    return [c for c in features if c.startswith('v_')]


def retime(origins, ace, features, lag_hours=0.):
    """ACE/plasma/speed families evaluated at ``origin - lag_hours``."""
    at = pd.DatetimeIndex(origins) - pd.Timedelta(hours=lag_hours)
    af, sf = ace_features(at), speed_features(at, ace)
    out = pd.DataFrame(index=pd.DatetimeIndex(origins))
    for c in ace_columns(features):
        out[c] = af[c].to_numpy()
    for c in speed_columns(features):
        out[c] = sf[c].to_numpy()
    return out


def keyed(frame):
    frame = frame.copy()
    frame['origin'] = pd.to_datetime(frame.origin, utc=True)
    if frame[['origin', 'h']].duplicated().any():
        raise ValueError('duplicate origin/horizon keys')
    if not frame.h.isin(range(1, H + 1)).all():
        raise ValueError('horizon outside 1..72')
    return frame.sort_values(['origin', 'h']).reset_index(drop=True)


def read_grid(path, constant):
    frame = keyed(pd.read_parquet(path))
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


def past_window(series, origins, lag=LAG):
    """Speeds at origin-lag+1 .. origin (inclusive): never after the origin hour."""
    o = pd.DatetimeIndex(origins)
    t = o.values[:, None] + np.arange(-lag + 1, 1).astype('timedelta64[h]')[None, :]
    return series.reindex(pd.DatetimeIndex(t.ravel(), tz='UTC')).to_numpy(np.float32).reshape(len(o), lag)


def build(out_dir, parity_origins=256):
    features = json.loads(paths.data('grids/features_v3.json').read_text())['features']
    constant = [c for c in features if c != 'h' and c not in VARYING]
    ace = filled_hourly()
    series = ace.ffill()
    retimed = ace_columns(features) + speed_columns(features)
    out, receipt = {}, {'lag_hours': 0, 'prefill': 'ffill along origin'}
    for name, filename, _ in GRIDS:
        origins, frame, C, E = read_grid(paths.data('grids', filename), constant)
        # Parity: the same functions at lag 2h must reproduce the frozen parquet columns.
        pick = np.linspace(0, len(origins) - 1, min(parity_origins, len(origins))).astype(int)
        old = frame.loc[frame.h == 1, retimed].to_numpy(np.float64)[pick]
        new = retime(origins[pick], ace, features, lag_hours=2.).to_numpy(np.float64)
        ok = np.isclose(old, new, rtol=1e-5, atol=1e-4, equal_nan=True)
        receipt[f'{name}_lag2_parity'] = {'origins': len(pick), 'mismatch': int((~ok).sum())}
        if not ok.all():
            raise ValueError(f'{name}: lag-2h parity failed on {int((~ok).sum())} values')
        lag0 = retime(origins, ace, features, lag_hours=0.)
        for c in retimed:
            C[:, constant.index(c)] = lag0[c].to_numpy(np.float32)
        out[name] = (origins, C, E)
    # Prefill along the joined origin timeline (stable sort: train/F1 grids overlap in time).
    names = [name for name, _, _ in GRIDS]
    origins = pd.DatetimeIndex(np.concatenate([out[n][0].asi8 for n in names]), tz='UTC')
    order = np.argsort(origins.asi8, kind='stable')
    C_all, E_all = prefill(np.concatenate([out[n][1] for n in names])[order],
                           np.concatenate([out[n][2] for n in names])[order])
    back = np.empty_like(order)
    back[order] = np.arange(len(order))
    C_all, E_all = C_all[back], E_all[back]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    for name, _, expect in GRIDS:
        o = out[name][0]
        sl = slice(start, start + len(o))
        start += len(o)
        if expect and (len(o) != expect[0] or o[0] != pd.Timestamp(expect[1]) or o[-1] != pd.Timestamp(expect[2])):
            raise ValueError(f'{name}: evaluation origin grid mismatch')
        np.savez(out_dir / f'{name}.npz', origins=o.asi8, C=C_all[sl], E=E_all[sl], past=past_window(series, o))
    series.to_frame('speed').to_parquet(out_dir / 'speed.parquet')
    (out_dir / 'names.json').write_text(json.dumps({'constant': constant, 'varying': list(VARYING)}))
    receipt.update({f'{n}_origins': len(out[n][0]) for n in names})
    (out_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(paths.OUTPUT_ROOT / 'pointweight4_cache'))
    parser.add_argument('--parity-origins', type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(build(args.out, args.parity_origins), indent=2))


if __name__ == '__main__':
    main()
