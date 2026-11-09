"""Training-only descriptive audit of the existing target forward-fill policy.

No model fitting, evaluation labels, or policy changes. F1 training rows only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PREPARED = REPO / 'research/rerun_df_ffill_20261103/input/df_ffill_trainy.parquet'
RAW = REPO / 'data-v2/input/df.parquet'


def describe(values, speed=True):
    values = np.asarray(values, dtype=float)
    result = {'n': len(values), 'mean': float(values.mean()),
            'sd': float(values.std()),
            'quantiles_0_50_90_99_100': np.quantile(values, [0, .5, .9, .99, 1]).tolist()}
    if speed:
        result.update(tail_ge550_fraction=float((values >= 550).mean()),
                      tail_ge700_fraction=float((values >= 700).mean()))
    return result


def main():
    columns = ['origin', 'h', 'target_time', 'y']
    filters = [('fit_F1', '==', True)]
    raw = pd.read_parquet(RAW, columns=columns, filters=filters, use_threads=False)
    prepared = pd.read_parquet(PREPARED, columns=columns, filters=filters, use_threads=False)
    raw = raw.sort_values(['origin', 'h']).reset_index(drop=True)
    prepared = prepared.sort_values(['origin', 'h']).reset_index(drop=True)
    assert raw.drop(columns='y').equals(prepared.drop(columns='y'))
    assert (pd.to_datetime(raw.origin, utc=True) + pd.Timedelta(hours=144)
            <= pd.Timestamp('2025-12-29', tz='UTC')).all()
    assert (pd.to_datetime(raw.target_time, utc=True)
            < pd.Timestamp('2025-12-29', tz='UTC')).all()
    missing = raw.y.isna().to_numpy()
    assert np.isfinite(prepared.y).all()
    assert np.array_equal(raw.y[~missing], prepared.y[~missing])
    timeline = raw[['target_time', 'y']].drop_duplicates('target_time').sort_values('target_time')
    timeline = timeline.set_index('target_time')
    index = pd.to_datetime(timeline.index, utc=True)
    last = pd.Series(index, index=index).where(timeline.y.notna().to_numpy()).ffill()
    age = (pd.Series(index, index=index) - last).dt.total_seconds() / 3600
    row_age = age.reindex(pd.to_datetime(raw.target_time, utc=True)).to_numpy()
    ages = row_age[missing]
    values = prepared.y.to_numpy()
    weights = np.where(values >= 550, 4., 1.)
    result = {'scope': 'F1 training only; no official evaluation truth read',
              'policy': 'unchanged user-confirmed training-target forward fill',
              'cutoff': '2025-12-29T00:00:00Z', 'maturity_hours': 144,
              'row_count': len(raw), 'imputed_fraction': float(missing.mean()),
              'observed': describe(values[~missing]), 'imputed': describe(values[missing]),
              'imputed_source_age_hours': describe(ages, speed=False),
              'imputed_loss_weight_share_before_horizon_normalization': float(weights[missing].sum() / weights.sum()),
              'per_lead': [], 'source_files': {}}
    for lead in range(1, 73):
        selected = raw.h.eq(lead).to_numpy()
        w, m = weights[selected], missing[selected]
        result['per_lead'].append({'lead': lead, 'imputed_fraction': float(m.mean()),
              'imputed_weight_share': float(w[m].sum() / w.sum())})
    result['missing_fraction_by_target_utc_hour'] = {
        str(hour): float(missing[pd.to_datetime(raw.target_time, utc=True).dt.hour.eq(hour)].mean())
        for hour in range(24)}
    for name, path in [('raw', RAW), ('prepared', PREPARED)]:
        stat = path.stat()
        result['source_files'][name] = {'path': str(path), 'bytes': stat.st_size,
                                        'mtime_ns': stat.st_mtime_ns}
    result['limits'] = [
        'Descriptive distribution and loss-mass audit, not evidence of forecast improvement.',
        'Weight shares describe the diagonal tail loss before spectral coupling.',
        'Missing target ages condition on the available training timeline.',
        'Existing user-confirmed fill policy is retained; no experiment or change authorized here.',
        'Do not select thresholds or fit corrections using official evaluation truth.']
    (ROOT / 'training_imputation_diagnostic.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('row_count', 'imputed_fraction',
                    'imputed_loss_weight_share_before_horizon_normalization')}))


if __name__ == '__main__':
    main()
