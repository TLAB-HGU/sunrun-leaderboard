import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SOURCE = Path(__file__).with_name('evaluate_goal.py')


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location('test_gate', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def synthetic(gate, tmp_path, monkeypatch):
    root = tmp_path / 'campaign'
    parent = tmp_path / 'parent'
    obs = tmp_path / 'synthetic-observations.parquet'
    monkeypatch.setattr(gate, 'PARENT', parent)
    monkeypatch.setattr(gate, 'OBSERVATIONS', obs)
    index = pd.date_range('2025-12-31', '2026-10-01', freq='h', tz='UTC')
    speed = pd.Series(400 + 40 * np.sin(np.arange(len(index)) / 17), index=index)
    pd.DataFrame({'timestamp_utc': index, 'speed_kms': speed.values}).to_parquet(obs)
    boundaries = {'F1': ('2026-01-01', '2026-04-01'),
                  'F2': ('2026-04-01', '2026-07-01'),
                  'F3': ('2026-07-01', '2026-09-30')}
    origin_sets = {}
    truth_sets = {}
    for period, (start, end) in boundaries.items():
        origins = pd.date_range(pd.Timestamp(start, tz='UTC') - pd.Timedelta(hours=1),
                                pd.Timestamp(end, tz='UTC') - pd.Timedelta(hours=73), freq='h')
        origin_sets[period] = origins
        truth_sets[period] = np.column_stack([speed.reindex(origins + pd.Timedelta(hours=h))
                                              for h in range(1, 73)])
    def make_run(run, seed, error, full=True):
        frames = []
        for period in gate.PERIODS:
            origins = origin_sets[period]
            pred = truth_sets[period] + error
            run.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(run / f'{period}_predictions.npz', origins=origins.asi8, pred=pred)
            frames.append(pd.DataFrame({'origin_last_input_utc': np.repeat(origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 72),
                                        'horizon_hours': np.tile(np.arange(1, 73), len(origins)),
                                        'pred_kms': pred.ravel()}))
            if full:
                meta = {'seed': seed, 'period': period, 'smoke': False}
                write(run / f'{period}_metadata.json', meta)
        pd.concat(frames, ignore_index=True).to_parquet(run / 'predictions.parquet', index=False)
        if full:
            (run / 'train.exit').write_text('0\n')
            write(run / 'complete.json', {'status': 'complete', 'exit_code': 0, 'seed': seed})
    for seed in gate.SEEDS:
        make_run(root / 'runs' / f'roundselect_s{seed}', seed, 40)
        make_run(parent / 'runs' / f'drop035_v650_s{seed}', seed,
                 gate.BEST_SINGLE_S + seed - 42, full=False)
    return gate, root, make_run


def test_incomplete_never_scores(gate, tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'fresh_scores', lambda _: pytest.fail('truth opened'))
    report = gate.evaluate(tmp_path)
    assert report['exit_code'] == 2 and not report['fresh_official_scoring']


def test_one_bad_seed_fail_despite_missing_audit(synthetic):
    gate, root, make_run = synthetic
    make_run(root / 'runs/roundselect_s44', 44, 99)
    report = gate.evaluate(root)
    assert report['exit_code'] == 1, report
    assert report['verdict'] == 'FAIL'
    assert report['all_seed_scores'] == pytest.approx([40, 40, 99])
    assert len(report['failures']) == 1 and '44' in report['failures'][0]


def test_good_scores_without_evidence_partial(synthetic):
    gate, root, _ = synthetic
    report = gate.evaluate(root)
    assert report['exit_code'] == 2 and report['fresh_official_scoring'], report
    assert report['all_seed_scores'] == pytest.approx([40, 40, 40])


def add_integrity(gate, root):
    runs = gate.completed_runs(root)
    treatment, scorer, context = gate.fresh_scores([gate.prediction_frame(run) for run in runs])
    controls = [gate.score_frame(scorer, context, pd.read_parquet(gate.PARENT / 'runs' /
                 f'drop035_v650_s{seed}' / 'predictions.parquet')) for seed in gate.SEEDS]
    summaries = {arm: {scope: {metric: {'per_seed': [s[scope][metric] for s in scores],
                  'mean': float(np.mean([s[scope][metric] for s in scores]))}
                  for metric in ('s', 'rmse', 'trmse', 'timse', 'pooled_mae')}
                  for scope in gate.SCOPES} for arm, scores in [('control', controls), ('treatment', treatment)]}
    draws = np.linspace(.5, .6, 2000)
    np.savez_compressed(root / 'bootstrap.npz', reduction=draws)
    inputs = [run / 'predictions.parquet' for run in runs] + [gate.PARENT / 'runs' /
               f'drop035_v650_s{seed}' / 'predictions.parquet' for seed in gate.SEEDS]
    result = {'status': 'complete', 'seeds': list(gate.SEEDS), 'best_reference_s': gate.BEST_SINGLE_S,
              'half_s_target': gate.HALF_S_TARGET, 'bootstrap': {'replicates': 2000, 'block_hours': 168,
              'seed': 20261104, 'period_stratified': True, 'empirical_timse_quantiles_recomputed': True},
              'paired_reduction_ci95': np.quantile(draws, [.025, .975]).tolist(), 'summaries': summaries,
              'candidate_gate_passed': True, 'auxiliary_gate_passed': True,
              'inputs': [gate.identity(p) for p in inputs], 'observations': gate.identity(gate.OBSERVATIONS),
              'scorer': gate.identity(gate.SCORER)}
    write(root / 'results.json', result)
    for seed, run in zip(gate.SEEDS, runs):
        for period in gate.PERIODS:
            model_dir = run / f'{period}_models'
            frozen_path = root / 'frozen_controls' / f'drop035_v650_s{seed}' / f'{period}_models/prior_predictions.npz'
            frozen_path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {'origins': np.arange(2), 'training_truth': np.ones((2,72)), 'raw': np.ones((2,72))}
            np.savez_compressed(frozen_path, **arrays)
            model_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(model_dir / 'prior900_predictions.npz', **arrays)
            control = {'passed': True, 'tolerance': 1e-6, 'max_absolute_difference': 0.0}
            heads = [{'block': block, 'native_rmse': list(range(900,0,-1)), 'selected_rounds': 900,
                      'full_rounds': 900, 'prior_rounds': 900, 'inference_iteration_range': [0,900]}
                     for block in range(3)]
            meta = {'seed': seed, 'period': period, 'smoke': False, 'fresh_training': True,
                    'reused_model_or_predictions': False, 'prior_rounds': 900, 'selected_rounds': [900]*3,
                    'tree_selection': heads, 'control': control, 'config': {'seed': seed}}
            for key in ('features', 'expanded_features', 'feature_count', 'expanded_feature_count',
                        'training_cutoff', 'maturity_hours', 'train_origins', 'train_start', 'train_end'):
                meta[key] = 'synthetic identical fixture'
            write(run / f'{period}_metadata.json', meta)
            write(gate.PARENT / 'runs' / f'drop035_v650_s{seed}' / f'{period}_metadata.json', meta)
            write(model_dir / 'control.json', control)
            write(model_dir / 'tree_selection.json', heads)
    for path in gate.integrity_files(root, runs):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic fixture only; no real fitted model')
    audit = {'status': 'pass', 'integrity_pass': True, 'reviewer': 'synthetic test fixture',
             'unresolved_required_gaps': [], 'checks': {name: {'passed': True, 'evidence': ['strategy.md']}
                 for name in gate.REQUIRED_CHECKS},
             'inputs': [gate.identity(p) for p in gate.integrity_files(root, runs)]}
    write(root / 'verification-independent.json', audit)


def test_complete_synthetic_evidence_pass(synthetic):
    gate, root, _ = synthetic
    add_integrity(gate, root)
    report = gate.evaluate(root)
    assert report['exit_code'] == 0 and report['goal_achieved'], report


@pytest.mark.parametrize('damage', ['threshold', 'ci', 'control', 'selected', 'audit_gap', 'stale'])
def test_integrity_damage_partial(synthetic, damage):
    gate, root, _ = synthetic
    add_integrity(gate, root)
    if damage in ('threshold', 'ci'):
        path = root / 'results.json'; data = json.loads(path.read_text())
        data['half_s_target' if damage == 'threshold' else 'paired_reduction_ci95'] = 90 if damage == 'threshold' else [-.1,.2]
        write(path, data)
    elif damage == 'audit_gap':
        path = root / 'verification-independent.json'; data = json.loads(path.read_text())
        data['unresolved_required_gaps'] = ['unverified provenance']; write(path,data)
    elif damage == 'selected':
        path = root / 'runs/roundselect_s42/F1_metadata.json'; data=json.loads(path.read_text())
        data['selected_rounds']=[1,1,1]; write(path,data)
    elif damage == 'control':
        path = root / 'runs/roundselect_s42/F1_models/prior900_predictions.npz'
        np.savez_compressed(path, origins=np.arange(2), training_truth=np.ones((2,72)), raw=np.full((2,72),2))
    else:
        (root / 'strategy.md').write_text('changed since audit')
    report = gate.evaluate(root)
    assert report['exit_code'] == 2 and not report['goal_achieved'], report
