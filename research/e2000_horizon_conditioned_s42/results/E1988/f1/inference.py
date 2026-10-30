"""Actual native XGBoost inference for frozen E2000 F2 models.

Requires the original hash-pinned causal feature parquet; no labels or forecasts
are inputs. This partial F2 release is not a complete F1+F2 leaderboard submission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from xgboost import Booster


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def predict(release, features):
    manifest = json.loads((release / 'manifest.json').read_text())
    if sha(features) != manifest['features_sha256']:
        raise ValueError('feature bytes differ from frozen causal features')
    frame = pd.read_parquet(features).sort_values(['origin', 'h'])
    frame['origin'] = pd.to_datetime(frame.origin, utc=True)
    origins = pd.DatetimeIndex(frame.origin.unique())
    expected = pd.MultiIndex.from_product([origins, range(1, 73)], names=['origin', 'h'])
    if not frame.set_index(['origin', 'h']).index.equals(expected):
        raise ValueError('complete unique origin/horizon grid required')
    columns = {}
    for name in manifest['base_features']:
        if name == 'h':
            continue
        a = frame[name].to_numpy(dtype=np.float32).reshape(-1, 72)
        if name in manifest['varying_features']:
            for h in range(1, 73):
                columns[f'{name}_h{h:02}'] = a[:, h-1]
        else:
            if not np.all((a == a[:, :1]) | (np.isnan(a) & np.isnan(a[:, :1]))):
                raise ValueError('origin invariant feature varies')
            columns[name] = a[:, 0]
    x = pd.DataFrame(columns)
    if list(x) != manifest['feature_names']:
        raise ValueError('feature order mismatch')
    models = []
    for item in manifest['models']:
        path = release / item['file']
        if sha(path) != item['sha256']:
            raise ValueError('model checksum mismatch')
        model = Booster()
        model.load_model(path)
        model.set_param({'nthread': 1})
        models.append(model)
    # Prepare all horizon inputs before measuring prediction time.
    if manifest['architecture'] == 'horizon_conditioned':
        inputs = [pd.DataFrame(np.column_stack((x.to_numpy(), np.full(len(x), h))),
                              columns=list(x) + ['forecast_horizon']) for h in range(1, 73)]
    else:
        inputs = [x] * 72
    start = time.perf_counter()
    result = np.column_stack([(models[h] if len(models) == 72 else models[0]).inplace_predict(inputs[h])
                              for h in range(72)]).astype(np.float64)
    elapsed = time.perf_counter() - start
    out = pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                        'horizon_hours': np.tile(np.arange(1, 73), len(origins)), 'pred_kms': result.ravel()})
    return out, {'inference_seconds': elapsed, 'folds': len(origins),
                 'inference_seconds_per_fold': elapsed / len(origins), 'n_jobs': 1,
                 'scope': 'actual loaded native model batch inference; excludes load and feature preparation'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timing', type=Path, required=True)
    a = p.parse_args()
    frame, timing = predict(a.release, a.features)
    frame.to_parquet(a.output, index=False)
    a.timing.write_text(json.dumps(timing, indent=2) + '\n')
    print(json.dumps(timing))

if __name__ == '__main__':
    main()
