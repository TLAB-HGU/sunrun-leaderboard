"""Finite, retrospective frozen V-JEPA ablation with explicit run paths."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from experiments.mse20_four_regimes import protocol, tree
from . import numeric

ARMS = ('baseline', 'image', 'video')
SELECTION_RULE = 'development ranks fixed arms, no tuning; selection lowest full-grid MSE among baseline and visual arms that do not regress selection precision or recall; freeze before audit; audit baseline and winner'


def score(predictions, frames, origins):
    result = protocol.score_development(predictions, frames, origins)
    p = protocol.canonical_predictions(predictions, origins)
    targets = pd.DatetimeIndex(pd.to_datetime(p.origin_last_input_utc, utc=True) + pd.to_timedelta(p.horizon_hours, unit='h'))
    observed = frames['observed'].reindex(targets).to_numpy(bool)
    truth = frames['speed'].reindex(targets).to_numpy(float)
    error = np.square(p.pred_kms.to_numpy() - truth)
    result.update(n_origins=len(origins), n_targets=len(p), n_observed=int(observed.sum()),
                  target_observation_mask_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),
                  prediction_keys_sha256=hashlib.sha256(pd.util.hash_pandas_object(p[protocol.KEY], index=False).values.tobytes()).hexdigest())
    result['horizon_buckets'] = {}
    for lo, hi in ((1, 24), (25, 48), (49, 72)):
        mask = p.horizon_hours.between(lo, hi).to_numpy()
        obs = mask & observed
        result['horizon_buckets'][f'{lo}-{hi}'] = dict(mse=float(error[mask].mean()), mse_observed=float(error[obs].mean()) if obs.any() else None,
                                                     n_targets=int(mask.sum()), n_observed=int(obs.sum()))
    result['forecastability_regimes'] = {'status': 'unavailable', 'reason': 'Frozen scorer requires period-specific preassigned T/S/F folds; official folds cannot be reused for these historical origins. No new regime thresholds fitted or wind-speed proxies substituted.'}
    return result


def eligible(candidate, baseline):
    for key in ('precision', 'recall'):
        a, b = candidate.get(key), baseline.get(key)
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            return False
        if a < b:
            return False
    return True


def save_plan(run_dir, frames, config):
    plans = numeric.plan_origins(frames, config['max_train_origins'])
    payload = {f'{phase}_{kind}': times.asi8 for phase, plan in plans.items() for kind, times in plan.items()}
    payload['origins'] = np.unique(np.concatenate(list(payload.values())))
    path = run_dir / 'planned_origins.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(payload) or any(not np.array_equal(old[k], v) for k, v in payload.items()):
                raise ValueError('existing planned origins differ')
    else:
        np.savez(path, **payload)
    return plans


def run(run_dir, embeddings=None, device='cpu', plan_only=False, source_metadata=None):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = tree.candidate_grid()[0]
    frames = numeric.load_frames()
    plans = save_plan(run_dir, frames, config)
    prereg = dict(config=config, splits=protocol.SPLITS, arms=ARMS, selection_rule=SELECTION_RULE,
                  visual_transform='PCA16 randomized seed42, train-mean imputation; fit only matching train origins, no per-arm HPO',
                  retrospective_exploratory=True, official_truth_access=False,
                  identities=frames['identities'], source_metadata=source_metadata,
                  schedule={k:{a:len(t) for a,t in v.items()} for k,v in plans.items()})
    prereg_path = run_dir / 'comparison_protocol.json'
    if prereg_path.exists():
        previous = json.loads(prereg_path.read_text())
        if previous['config'] != config or previous['selection_rule'] != SELECTION_RULE or previous['identities'] != frames['identities']:
            raise ValueError('comparison protocol drift')
    else:
        protocol.write_json(prereg_path, prereg)
    if plan_only:
        return prereg
    if embeddings is None:
        raise ValueError('--embeddings required unless --plan-only')
    code_ref = {str(path): protocol.digest(path) for path in [Path(__file__), Path(numeric.__file__), Path(tree.__file__)]}
    code_path = run_dir / 'comparison_code_identity.json'
    if code_path.exists() and json.loads(code_path.read_text()) != code_ref:
        raise ValueError('comparison code changed across restart')
    if not code_path.exists():
        protocol.write_json(code_path, code_ref)
    embedding_ref = protocol.file_ref(embeddings)
    binding_path = run_dir / 'comparison_embedding_identity.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != embedding_ref:
        raise ValueError('embedding file changed across comparison restart')
    protocol.write_json(binding_path, embedding_ref)
    results = {}
    winner_path = run_dir / 'winner.json'
    for phase, spec in protocol.SPLITS.items():
        if phase == 'audit':
            if winner_path.exists():
                winner = json.loads(winner_path.read_text())['arm']
            else:
                allowed = ['baseline'] + [a for a in ARMS[1:] if eligible(results['selection'][a]['metrics'], results['selection']['baseline']['metrics'])]
                winner = min(allowed, key=lambda a: (results['selection'][a]['metrics']['mse'], ARMS.index(a)))
                protocol.write_json(winner_path, dict(arm=winner, allowed=allowed, rule=SELECTION_RULE,
                                                      frozen_before_audit_utc=protocol.utcnow(), embedding_identity=embedding_ref))
            arms = list(dict.fromkeys(['baseline', winner]))
        else:
            arms = ARMS
        split_frames = protocol.slice_frames(frames, spec['end'])
        origins = plans[phase]['eval']
        results[phase] = {}
        for arm in arms:
            output = run_dir / 'comparison' / phase / arm
            complete = output / 'result.json'
            protocol.write_json(run_dir / 'comparison_progress.json', dict(stage='fitting', phase=phase, arm=arm, updated_utc=protocol.utcnow()))
            start = time.perf_counter()
            if (output / 'metadata.json').exists():
                metadata = json.loads((output / 'metadata.json').read_text())
                if metadata['config'] != config or pd.Timestamp(metadata['cutoff']) != pd.Timestamp(spec['cutoff']) or metadata['input_identities'] != frames['identities']:
                    raise ValueError('saved model does not match current comparison protocol')
                pred = numeric.predict_saved(split_frames, origins, output, device, embeddings)
            else:
                pred, metadata = numeric.fit_predict(split_frames, spec['cutoff'], origins, config, device, output, arm, embeddings)
            replay_started = time.perf_counter()
            replay = numeric.predict_saved(split_frames, origins, output, device, embeddings)
            replay_seconds = time.perf_counter() - replay_started
            maxdiff = float(np.max(np.abs(pred.pred_kms.to_numpy() - replay.pred_kms.to_numpy())))
            if maxdiff > 1e-3:
                raise ValueError(f'saved inference replay differs: {maxdiff}')
            p = protocol.canonical_predictions(pred, origins)
            p.to_parquet(output / 'predictions.parquet', index=False)
            record = dict(metrics=score(pred, split_frames, origins), metadata=metadata,
                          saved_replay_max_abs_diff=maxdiff, saved_replay_seconds=replay_seconds, saved_replay_seconds_per_origin=replay_seconds / len(origins), comparison_wall_seconds=time.perf_counter() - start,
                          predictions=protocol.file_ref(output / 'predictions.parquet'))
            protocol.write_json(complete, record)
            results[phase][arm] = record
            baseline = results[phase]['baseline']
            for key in ('prediction_keys_sha256', 'target_observation_mask_sha256'):
                if record['metrics'][key] != baseline['metrics'][key]:
                    raise ValueError('arms differ in scoring support')
            if metadata['train_origin_sha256'] != baseline['metadata']['train_origin_sha256']:
                raise ValueError('arms differ in training origins')
            protocol.write_json(run_dir / 'comparison_results.json', results)
    protocol.write_json(run_dir / 'comparison_progress.json', dict(stage='complete', winner=winner, updated_utc=protocol.utcnow()))
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--source-metadata', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    metadata = json.loads(args.source_metadata.read_text()) if args.source_metadata else None
    if args.device.startswith('cuda') and not args.plan_only:
        from experiments.ch_breakthrough_v2.common import acquire_gpu_lease
        with acquire_gpu_lease(protocol.ROOT / 'store/ch-breakthrough-v2/gpu-leases') as gpu:
            device = f'cuda:{gpu}'
            protocol.write_json(args.run_dir / 'comparison_gpu_lease.json', dict(gpu=gpu, device=device, acquired_utc=protocol.utcnow()))
            run(args.run_dir, args.embeddings, device, args.plan_only, metadata)
    else:
        run(args.run_dir, args.embeddings, args.device, args.plan_only, metadata)


if __name__ == '__main__':
    main()
