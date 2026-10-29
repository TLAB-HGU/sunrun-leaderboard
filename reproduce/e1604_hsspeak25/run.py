"""E1604 HSSPEAK25: raw ACE -> 29 causal features -> two HistGradientBoosting heads -> 12 h Gaussian bump -> predictions.

E1604 corrects the frozen E0472 forecasts. Two heads (peak speed, peak hour) are trained on the roll6 training matrix
and a 12 h Gaussian bump of height 0.15 * (peak - path minimum) is added from lead 25. The three E0472 seeds are
averaged. F2 (2026-06..09) uses every training row; F1 (2026-03..05) uses only rows whose 72 h label window ends before
2026-02-26 00:00 UTC (leaderboard F1 rule, 2026-10-26).

Inputs (``DATA_ROOT``, see reproduce/README.md):
    store/ch-v1/ace.parquet                       raw ACE hourly table
    e1604/prepared_inputs.npz                     frozen roll6 training matrix + F2 base forecasts (E0472)
    e1604/hsspeak_f1_inputs.npz                   frozen F1 base forecasts (E0472-F1), training row calendar
    e1604/huxt_traj_slim.npz                      HUXt Earth trajectory for the F1/F2 origins

Not regenerated: the E0472 base forecasts (D148 lean+ENLIL, SUVI strips, GONG/PFSS/WSA/HUXt) and the roll6 control
paths behind the training matrix. Everything downstream of them is rebuilt and checked against the frozen matrices.
E0472 and this experiment used the historical ACE +2h input timing; this script does not change that.

    python reproduce/e1604_hsspeak25/run.py --out reproduce/outputs/e1604 [--reference <published predictions.parquet>]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402

HORIZON = 72
CARR = 654                      # one Carrington rotation in hours
WIDTH, SHRINK, MIN_LEAD = 12.0, 0.15, 25
F1_CUTOFF = pd.Timestamp('2026-02-26T00:00:00Z')
HEAD = dict(max_iter=500, learning_rate=0.04, max_depth=4, min_samples_leaf=50, random_state=0)
INPUT_SHA256 = {'e1604/prepared_inputs.npz': 'f02d5e04095c28d71c34065b18699765ec4beff3018acc0b9efac736259f5ead',
                'e1604/hsspeak_f1_inputs.npz': '795df2238e423b8246f2c972a1360c2c2cda1721a59da6f662a733c6a1501210'}


def load_ace():
    """Hourly forward-filled ACE speed (expos.evaluate._ace)."""
    a = pd.read_parquet(paths.data('store/ch-v1/ace.parquet'))
    a.index = pd.to_datetime(a.pop('timestamp_utc'), utc=True).dt.floor('h')
    a = a[~a.index.duplicated(keep='last')].sort_index()
    a = a.reindex(pd.date_range(a.index.min(), a.index.max(), freq='h', tz='UTC'))
    return a['filled_speed_kms'].ffill().astype(float)


def features(o, path, hx, ace):
    """29 causal columns; each uses only data at or before the origin."""
    def at(lag):
        return ace.reindex(o - pd.Timedelta(hours=lag)).values.astype(float)

    def roll(win):
        return ace.rolling(win, min_periods=1).mean().reindex(o).values.astype(float)
    rec = np.empty_like(path)
    for h in range(1, HORIZON + 1):
        rec[:, h - 1] = ace.reindex(o + pd.Timedelta(hours=h - CARR)).values
    return np.column_stack([
        np.nanmax(rec, 1), np.nanargmax(rec, 1) + 1.0, np.nanmax(rec, 1) - np.nanmin(rec, 1),
        at(0), at(6), at(24), at(168), at(648), roll(6), roll(24), roll(168), roll(648),
        np.nanmax(path, 1), np.nanargmax(path, 1) + 1.0, np.nanmean(path, 1), np.nanmin(path, 1), np.nanstd(path, 1),
        path[:, 0], path[:, 23], path[:, 47], path[:, 71],
        np.nanmax(hx, 1), np.nanargmax(hx, 1) + 1.0, hx[:, 0], hx[:, 23], hx[:, 47], hx[:, 71],
        np.sin(2 * np.pi * o.dayofyear / 365.25), np.cos(2 * np.pi * o.dayofyear / 365.25)])


def bump(base, peak, hour, min_lead=MIN_LEAD):
    """``base`` (..., n, 72), ``peak``/``hour`` (..., n): Gaussian peak correction from lead ``min_lead``."""
    hh = np.arange(1, HORIZON + 1, dtype=float)
    b = np.exp(-((hh - np.clip(hour, 1, HORIZON)[..., None]) ** 2) / (2 * WIDTH ** 2))
    b /= np.nanmax(b, axis=-1, keepdims=True)
    inc = SHRINK * (peak[..., None] - np.nanmin(base, axis=-1, keepdims=True)) * b
    inc[..., :min_lead - 1] = 0.0
    return base + np.nan_to_num(inc)


def targets(ace, origins):
    """Peak speed and its hour over the 72 h after each origin (training labels)."""
    truth = np.stack([ace.reindex(origins + pd.Timedelta(hours=h)).values for h in range(1, HORIZON + 1)], axis=1)
    return np.nanmax(truth, 1), np.nanargmax(truth, 1) + 1.0


def fit_heads(x, peak, hour):
    return [HistGradientBoostingRegressor(**HEAD).fit(x, y) for y in (peak, hour)]


def predict(heads, x, base):
    """``x`` (3, n, 29), ``base`` (3, n, 72) -> per-seed (3, n, 72) and the seed mean."""
    flat = x.reshape(-1, x.shape[-1])
    peak = heads[0].predict(flat).reshape(x.shape[:2])
    hour = heads[1].predict(flat).reshape(x.shape[:2])
    per_seed = bump(base, peak, hour)
    return per_seed, per_seed.mean(axis=0)


def load_inputs(check_hashes=True):
    if check_hashes:
        for name, expected in INPUT_SHA256.items():
            if paths.sha256(paths.data(name)) != expected:
                raise ValueError('frozen input changed: ' + name)
    with np.load(paths.data('e1604/prepared_inputs.npz')) as z:
        f2 = dict(train_x=z['train_x'], train_peak=z['train_peak'], train_hour=z['train_hour'], x=z['official_x'],
                  base=z['base'], origins=pd.to_datetime(z['origins_ns'], utc=True))
    with np.load(paths.data('e1604/hsspeak_f1_inputs.npz')) as z:
        f1 = dict(x=z['f1_x'], base=z['f1_base'], origins=pd.to_datetime(z['f1_origins_ns'], utc=True),
                  train_origins=pd.to_datetime(z['train_origins_ns'], utc=True))
    with np.load(paths.data('e1604/huxt_traj_slim.npz')) as z:
        huxt = pd.DataFrame(z['traj'].astype(float), index=pd.to_datetime(z['origins_ns'], utc=True))
    return f2, f1, huxt


def verify(f2, f1, huxt, ace):
    """Rebuild the frozen matrices from raw ACE / base paths / HUXt and report any deviation."""
    report = {}
    # features of the evaluation origins
    for name, d in (('f2', f2), ('f1', f1)):
        hx = huxt.reindex(d['origins']).to_numpy()
        diffs = []
        for s in range(3):
            built = features(d['origins'], d['base'][s], hx, ace)
            diffs.append(float(np.nanmax(np.abs(built - d['x'][s]))) if not np.array_equal(built, d['x'][s], equal_nan=True) else 0.0)
        report[f'{name}_features_max_abs_diff'] = max(diffs)
    # training rows: ACE at the origin and the peak / hour labels
    origins = f1['train_origins']
    report['train_rows'] = len(origins)
    report['train_ace_at_origin_max_abs_diff'] = float(np.nanmax(np.abs(f2['train_x'][:, 3] - ace.reindex(origins).to_numpy())))
    peak, hour = targets(ace, origins)
    ok = np.isfinite(f2['train_peak'])
    report['train_peak_max_abs_diff'] = float(np.nanmax(np.abs(peak[ok] - f2['train_peak'][ok])))
    report['train_hour_max_abs_diff'] = float(np.nanmax(np.abs(hour[ok] - f2['train_hour'][ok])))
    report['train_labels_finite'] = int(ok.sum())
    return report


def run(out, reference=None, skip_verify=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ace = load_ace()
    f2, f1, huxt = load_inputs()
    assert f2['x'].shape == (3, 2833, 29) and f2['base'].shape == (3, 2833, 72)
    assert f1['x'].shape == (3, 2137, 29) and f1['base'].shape == (3, 2137, 72)
    assert len(f1['train_origins']) == len(f2['train_x'])
    receipt = {'verification': None if skip_verify else verify(f2, f1, huxt, ace)}
    f1_rows = np.asarray(f1['train_origins'] + pd.Timedelta(hours=HORIZON) < F1_CUTOFF)
    with threadpool_limits(limits=1):
        heads = fit_heads(f2['train_x'], f2['train_peak'], f2['train_hour'])
        heads_f1 = fit_heads(f2['train_x'][f1_rows], f2['train_peak'][f1_rows], f2['train_hour'][f1_rows])
        start = time.perf_counter()
        per_seed, pred = predict(heads, f2['x'], f2['base'])
        per_seed1, pred1 = predict(heads_f1, f1['x'], f1['base'])
        elapsed = time.perf_counter() - start
    for values in (per_seed, per_seed1):
        assert np.isfinite(values).all() and (values > 0).all() and (values <= 3000).all()
    origins = f1['origins'].append(f2['origins'])
    keys = {'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), HORIZON),
            'horizon_hours': np.tile(np.arange(1, HORIZON + 1), len(origins))}
    frame = pd.DataFrame({**keys, 'pred_kms': np.concatenate([pred1.ravel(), pred.ravel()])})
    frame.to_parquet(out / 'predictions.parquet', index=False)
    for seed in range(3):
        values = np.concatenate([per_seed1[seed].ravel(), per_seed[seed].ravel()])
        pd.DataFrame({**keys, 'pred_kms': values}).to_parquet(out / f'predictions_f1f2_s{seed}.parquet', index=False)
    receipt.update(rows=len(frame), f1_train_rows=int(f1_rows.sum()), train_rows=int(len(f1_rows)),
                   inference_seconds=elapsed, head=HEAD, width=WIDTH, shrink=SHRINK, min_lead=MIN_LEAD)
    if reference:
        ref = pd.read_parquet(reference)
        merged = frame.merge(ref, on=['origin_last_input_utc', 'horizon_hours'], suffixes=('', '_ref'), validate='one_to_one')
        assert len(merged) == len(frame) == len(ref)
        receipt['reference_max_abs_diff'] = float((merged.pred_kms - merged.pred_kms_ref).abs().max())
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(paths.OUTPUT_ROOT / 'e1604'))
    parser.add_argument('--reference', default=None, help='published predictions parquet to diff against')
    parser.add_argument('--skip-verify', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.reference, args.skip_verify), indent=2))


if __name__ == '__main__':
    main()
