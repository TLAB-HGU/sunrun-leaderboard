"""Read-only staged inference of frozen E1725 fold3 trees; never fit/select."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import gc
import hashlib
import json
import sys

ROOT = Path('/home/t-lab01/sunrun')
sys.path.insert(0, str(ROOT / 'leaderboard'))
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from experiments.solar_groupmean_anchor_20261023 import run as R

SOURCE = ROOT / 'research/expos/runs/E1725'
OUT = Path(__file__).resolve().parent / 'data'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stages(model, x):
    matrix = xgb.DMatrix(x, nthread=1)
    prediction = np.zeros(len(x), dtype=float)
    yield prediction
    for tree, ids, beta in model.trees:
        tree.set_param({'nthread': 1, 'device': 'cpu'})
        leaves = tree.predict(matrix, pred_leaf=True).astype(np.int64).reshape(-1)
        pos = np.searchsorted(ids, leaves)
        assert np.all(pos < len(ids)) and np.array_equal(ids[pos], leaves)
        prediction += model.rate * beta[pos]
        yield prediction


def run(seed):
    print(f'seed{seed}: loading immutable inputs', flush=True)
    manifest_path = SOURCE / 'source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    input_paths = [ROOT / 'leaderboard/store/ch-v1/ace.parquet', R.late.HUXT]
    input_paths += [R.clock.paths(k, seed)[0] for k in R.ORDER]
    input_paths += [Path(R.__file__), Path(R.late.__file__), Path(R.engine.__file__),
                    Path(R.late.front.__file__), Path(R.clock.__file__)]
    verified_inputs = {str(p): sha(p) for p in input_paths}
    assert all(value == manifest[path] for path, value in verified_inputs.items())
    receipt_path = SOURCE / f'receipt_s{seed}.json'
    archived = next(x for x in json.loads(receipt_path.read_text()) if x['fold'] == 3)
    model_path = SOURCE / f'model_f3_s{seed}.joblib'
    assert sha(model_path) == archived['model_sha256']
    model = joblib.load(model_path)
    assert R.model_digest(model) == archived['model_content_sha256']
    assert (model.seed, model.gamma, model.rounds, model.rate, len(model.trees)) == (seed, 2, 300, .03, 300)
    ace, channels, grids = R.late.load_inputs(seed)
    rows, x = R.materialize(grids, 3, ace, channels)
    train_path = SOURCE / f'training_f3_s{seed}.parquet'
    pd.testing.assert_frame_equal(rows, pd.read_parquet(train_path))
    training_digest = R.digest(rows, x)
    assert training_digest == archived['training_sha256']
    cutoff = R.clock.cutoff(3)
    assert (rows.target + 72 * R.HOUR <= cutoff).all()
    assert (rows.target < cutoff).all() and rows.lead.between(25, 72).all()
    y = (rows.y - rows.control).to_numpy()
    _, inverse, counts = R.engine.groups_index(rows[['source_fold', 'lead']].to_numpy())
    train_mse, objectives = [], []
    print(f'seed{seed}: {len(rows)} exact training rows, staged inference', flush=True)
    for i, pred in enumerate(stages(model, x)):
        residual = pred - y
        train_mse.append(float(np.mean(residual ** 2)))
        objective = R.engine.gradient_loss(residual, inverse, counts, model.gamma)[2]
        objectives.append(objective)
        if i:
            np.testing.assert_allclose(objective, model.trace[i-1]['after'], rtol=1e-12, atol=1e-8)
    assert np.array_equal(pred, model.training_prediction)
    del x, pred, residual
    gc.collect()
    org, control = grids[3]
    expected = pd.date_range('2026-02-01', '2026-05-28 23:00', freq='h', tz='UTC')
    assert org.equals(expected) and control.shape == (len(expected), 72)
    raw_path = SOURCE / 'raw/roll3' / f'f3_s{seed}.parquet'
    assert sha(raw_path) == archived['raw_sha256']
    raw = R.clock.load_grid(raw_path, org)
    target_times = pd.DatetimeIndex(np.repeat(org, 48)) + pd.to_timedelta(np.tile(np.arange(25, 73), len(org)), unit='h')
    truth = ace.reindex(target_times).to_numpy().reshape(len(org), 48)
    observed = np.isfinite(truth)
    assert target_times.min() > cutoff and observed.any()
    observed_targets = target_times[observed.ravel()]
    assert observed_targets.max() < R.CAP
    assert rows.target.max() < observed_targets.min()
    sums = np.zeros(301)
    max_delta = 0.
    for start in range(0, len(org), 256):
        stop = min(start + 256, len(org))
        features, names = R.design(org[start:stop], control[start:stop], channels, ace, 'A')
        assert len(names) == 125 and features.shape == ((stop-start)*72, 125)
        late = features.reshape(stop-start, 72, 125)[:, 24:, :].reshape(-1, 125)
        mask = observed[start:stop].ravel()
        targets = (truth[start:stop] - control[start:stop, 24:]).ravel()[mask]
        for i, prediction in enumerate(stages(model, late)):
            sums[i] += np.sum((prediction[mask] - targets) ** 2)
        reconstructed = control[start:stop, 24:] + prediction.reshape(stop-start, 48)
        delta = np.abs(reconstructed - raw[start:stop, 24:])
        max_delta = max(max_delta, float(delta.max()))
        np.testing.assert_allclose(reconstructed, raw[start:stop, 24:], rtol=0, atol=1e-10)
        print(f'seed{seed}: validation origins {stop}/{len(org)}', flush=True)
    frame = pd.DataFrame({'round': np.arange(301), 'train_mse': train_mse,
                          'validation_mse': sums / observed.sum(), 'train_anchor_objective': objectives,
                          'seed': seed, 'fold': 3})
    assert np.isfinite(frame.to_numpy()).all()
    out = OUT / f'e1725_fold3_seed{seed}_train_validation.csv'
    frame.to_csv(out.with_suffix('.tmp'), index=False)
    out.with_suffix('.tmp').replace(out)
    evidence = {
        'seed': seed, 'fold': 3, 'experiment': 'E1725', 'diagnostic_only': True,
        'method': 'Frozen stored tree leaf inference; no fitting, early stopping, model selection or official evaluation',
        'metric': 'Mean squared raw residual error: prediction - (observed ACE - archived control), h25..72',
        'round_zero': 'zero residual correction / archived control',
        'validation_status': 'Historically exposed preofficial fold3, not a new holdout',
        'training_rows': len(rows), 'training_target_min': str(rows.target.min()),
        'training_target_max': str(rows.target.max()), 'training_cutoff': str(cutoff),
        'training_maturity_hours': 72, 'training_digest': training_digest,
        'training_rows_match_archived_parquet': True, 'training_prediction_exact': True,
        'training_objective_matches_all_archived_rounds': True,
        'validation_origin_start': str(org.min()), 'validation_origin_end': str(org.max()),
        'validation_origins': len(org), 'validation_full_late_cells': observed.size,
        'validation_observed_cells': int(observed.sum()), 'validation_missing_cells': int((~observed).sum()),
        'validation_mask_sha256': hashlib.sha256(observed.tobytes()).hexdigest(),
        'validation_target_keys_sha256': hashlib.sha256(target_times.asi8.tobytes()).hexdigest(),
        'validation_target_min': str(target_times.min()), 'validation_target_max': str(target_times.max()),
        'validation_observed_target_min': str(observed_targets.min()),
        'validation_observed_target_max': str(observed_targets.max()),
        'validation_observed_targets_before_cap': str(R.CAP),
        'train_validation_targets_disjoint': True,
        'validation_max_absolute_archive_prediction_delta': max_delta,
        'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [receipt_path, model_path, train_path, raw_path]},
        'script_sha256': sha(Path(__file__)), 'csv_sha256': sha(out),
        'original_source_manifest_sha256': sha(manifest_path),
        'verified_original_source_hashes': verified_inputs,
        'original_model_file_unchanged': sha(model_path) == archived['model_sha256'],
    }
    (OUT / f'e1725_fold3_seed{seed}_receipt.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(f'seed{seed}: verified and saved {out}', flush=True)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in range(3):
        run(seed)
