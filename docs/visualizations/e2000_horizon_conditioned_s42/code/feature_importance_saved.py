"""Feature importance of the original E2000 Direct (E1987) and Direct-pooled (E1988) releases from their saved boosters.

The F1 and F2 boosters were saved by the original runs, so nothing is refit: the saved models are loaded, their
predictions on the hash-pinned causal feature files must equal the submitted predictions and reproduce the
leaderboard MSE, then split counts (booster JSON) and permutation importance (72 h MSE increase when one base
feature, all 72 horizon copies together, is shuffled across origins; 5 shuffles) are computed.

    python feature_importance_saved.py --experiment e2000_direct_s42 --out DIR
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import Booster

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_importance as F  # noqa: E402
import registry as R  # noqa: E402

PUB = Path('/home/t-lab01/sunrun/research/e2000/publication')
RUNS = {'e2000_direct_s42': 'E1987', 'e2000_horizon_conditioned_s42': 'E1988'}
FEATURES = {'f1': PUB / 'f1/inputs/features_f1.parquet',
            'final': Path('/home/t-lab01/sunrun/research/e0472-multioutput-s42-20261026/relocated-v1/official_causal.parquet')}
INPUTS = Path(__file__).resolve().parent / 'inputs'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(release, features):
    """Feature matrix and boosters exactly as the release's inference.py builds them."""
    manifest = json.loads((release / 'manifest.json').read_text())
    if sha(features) != manifest['features_sha256']:
        raise SystemExit('feature bytes differ from the frozen causal features')
    frame = pd.read_parquet(features).sort_values(['origin', 'h'])
    frame['origin'] = pd.to_datetime(frame.origin, utc=True)
    origins = pd.DatetimeIndex(frame.origin.unique())
    columns, groups = {}, []
    for name in manifest['base_features']:
        if name == 'h':
            continue
        a = frame[name].to_numpy(dtype=np.float32).reshape(-1, 72)
        if name in manifest['varying_features']:
            for h in range(1, 73):
                columns[f'{name}_h{h:02}'] = a[:, h - 1]
                groups.append(name)
        else:
            columns[name] = a[:, 0]
            groups.append(name)
    x = pd.DataFrame(columns)
    if list(x) != manifest['feature_names']:
        raise SystemExit('feature order mismatch')
    models = []
    for item in manifest['models']:
        if sha(release / item['file']) != item['sha256']:
            raise SystemExit('model checksum mismatch')
        m = Booster()
        m.load_model(release / item['file'])
        m.set_param({'nthread': 8})
        models.append(m)
    pooled = manifest['architecture'] == 'horizon_conditioned'
    return origins, x.to_numpy(np.float32), np.array(groups), models, pooled


def predict(X, models, pooled):
    if pooled:
        return np.column_stack([models[0].inplace_predict(np.column_stack([X, np.full(len(X), h, np.float32)]))
                                for h in range(1, 73)]).astype(np.float64)
    return np.column_stack([models[h].inplace_predict(X) for h in range(72)]).astype(np.float64)


def compute(experiment, out):
    rec = next(r for r in R.records() if r['experiment'] == experiment)
    submitted = pd.read_parquet(INPUTS / f'{experiment}.parquet')
    submitted['origin_last_input_utc'] = pd.to_datetime(submitted.origin_last_input_utc, utc=True)
    truth = pd.read_parquet(INPUTS / 'truth.parquet')
    truth['origin_last_input_utc'] = pd.to_datetime(truth.origin_last_input_utc, utc=True)
    rows, receipt = [], {'experiment': experiment, 'run': RUNS[experiment], 'source': 'saved',
                         'config_sha256': rec['config_sha256']}
    for stage, (_, _, metric) in F.STAGES.items():
        release = PUB / RUNS[experiment] / ('f1' if stage == 'f1' else '')
        origins, X, groups, models, pooled = load(release, FEATURES[stage])
        key = pd.MultiIndex.from_product([origins, range(1, 73)])
        sub = submitted.set_index(['origin_last_input_utc', 'horizon_hours']).reindex(key).pred_kms.to_numpy().reshape(-1, 72)
        y = truth.set_index(['origin_last_input_utc', 'horizon_hours']).reindex(key).target_kms.to_numpy().reshape(-1, 72)
        pred = predict(X, models, pooled)
        diff, base = float(np.abs(pred - sub).max()), float(np.mean((y - pred) ** 2))
        print(f'{experiment} {stage}: saved boosters max |diff| {diff:.2e} km/s, MSE {base:.6f} '
              f'(leaderboard {rec[metric]:.6f})', flush=True)
        if diff > 1e-3 or abs(base - rec[metric]) > 1e-6:
            raise SystemExit(f'{stage}: saved boosters do not reproduce the submitted predictions; stopping')
        cols = list(groups) + (['forecast_horizon'] if pooled else [])
        splits = pd.concat([F.split_counts(m, np.array(cols)) for m in models]).groupby(level=0, sort=False).sum()
        receipt[stage] = {'max_abs_diff_kms': diff, 'mse': base, 'leaderboard_mse': rec[metric], 'origins': len(origins),
                          'boosters': len(models), 'total_splits': int(splits.sum()),
                          'features_sha256': sha(FEATURES[stage])}
        rng = np.random.default_rng(F.SEED)
        unique = list(dict.fromkeys(groups))
        for i, g in enumerate(unique):
            idx = np.flatnonzero(groups == g)
            deltas = []
            for _ in range(F.REPEATS):
                perm = rng.permutation(len(X))
                Xp = X.copy()
                Xp[:, idx] = X[perm][:, idx]
                deltas.append(float(np.mean((y - predict(Xp, models, pooled)) ** 2)) - base)
            rows.append({'stage': stage, 'feature': g, 'kind': F.kind(g), 'split_count': int(splits.get(g, 0)),
                         'perm_delta_mse_mean': float(np.mean(deltas)), 'perm_delta_mse_sd': float(np.std(deltas, ddof=1))})
            if (i + 1) % 20 == 0:
                print(f'{experiment} {stage}: permuted {i + 1}/{len(unique)} features', flush=True)
        if pooled:
            rows.append({'stage': stage, 'feature': 'lead', 'kind': F.INTERNAL, 'split_count': int(splits['forecast_horizon']),
                         'perm_delta_mse_mean': np.nan, 'perm_delta_mse_sd': np.nan})
    pd.DataFrame(rows).to_csv(out / 'feature_importance.csv', index=False, float_format='%.6g')
    receipt.update({'repeats': F.REPEATS, 'seed': F.SEED, 'permutation_groups': len(unique)})
    (out / 'feature_importance_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--experiment', required=True, choices=RUNS)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--label', required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    compute(a.experiment, a.out)
    F.plot(None, a.out, a.label)
