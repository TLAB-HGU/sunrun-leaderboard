"""Focused guards for the durable ensemble publication path; no network or training."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ensemble
import publish
import publish_visualization as visualization


def frame():
    return pd.DataFrame({'origin_last_input_utc': ['2026-01-01T00:00:00Z'] * 2,
                         'horizon_hours': [1, 2], 'pred_kms': [100.0, 200.0]})


def test_mean_aligns_keys_and_preserves_fixed_weights():
    first = frame()
    output, seconds = ensemble.average_predictions([first, first.assign(pred_kms=[103.0, 203.0]).iloc[::-1],
                                                   first.assign(pred_kms=[106.0, 206.0])])
    np.testing.assert_array_equal(output.pred_kms, [103.0, 203.0])
    assert seconds >= 0 and output.horizon_hours.tolist() == [1, 2]


@pytest.mark.parametrize('bad', [lambda f: pd.concat([f, f.iloc[:1]]), lambda f: f.iloc[:1],
                                lambda f: f.assign(pred_kms=np.nan), lambda f: f.assign(pred_kms=-1),
                                lambda f: f.assign(horizon_hours=[1, 73])])
def test_mean_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError):
        ensemble.average_predictions([frame(), frame(), bad(frame())])


def test_resume_preserves_prediction_and_timing(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, 'WORK', tmp_path)
    runs = {}
    for seed in publish.SEEDS:
        run = tmp_path / str(seed)
        run.mkdir()
        frame().assign(pred_kms=[100 + seed, 200 + seed]).to_parquet(run / 'predictions.parquet', index=False)
        runs[seed] = run
    publish.prepare_predictions(runs)
    timing = (tmp_path / 'ensemble.json').read_bytes()
    publish.prepare_predictions(runs)
    assert (tmp_path / 'ensemble.json').read_bytes() == timing
    frame().assign(pred_kms=[1.0, 2.0]).to_parquet(runs[42] / 'predictions.parquet', index=False)
    with pytest.raises(AssertionError):
        publish.prepare_predictions(runs)


def test_existing_result_resume_and_collision():
    config = {'data': {'fixed': True}}
    row = {'team_member': publish.MEMBER, 'experiment': publish.EXPERIMENT, 'config': config}
    assert publish.existing_result([], config) is None
    assert publish.existing_result([row], config) is row
    with pytest.raises(RuntimeError):
        publish.existing_result([row, row], config)
    with pytest.raises(RuntimeError):
        publish.existing_result([row], {'data': {'fixed': False}})


def test_unfinished_training_cannot_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, 'ROOT', tmp_path)
    with pytest.raises(RuntimeError, match='successful training exit'):
        publish.validate_runs()
    assert not (tmp_path / 'campaign-complete.json').exists()


def test_scoring_uses_frozen_assignment_observations(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(publish, 'ROOT', tmp_path)
    monkeypatch.setattr(publish, 'WORK', tmp_path/'publication')
    (tmp_path/'publication').mkdir()
    (tmp_path/'ensemble.py').write_text('# fixed mean\n')
    (tmp_path/'strategy.md').write_text('frozen\n')
    frozen = tmp_path/'external_observations.parquet'
    pd.DataFrame({'timestamp_utc':['2026-01-01T00:00:00Z'],'speed_kms':[400.]}).to_parquet(frozen)
    publish.write(tmp_path/'assignment.global.json', {'observations':str(frozen)})
    run=tmp_path/'run';run.mkdir()
    for name in publish.MODULES:
        (run/name).write_text('# source\n')
    monkeypatch.setitem(sys.modules, 'architecture_code', SimpleNamespace())
    class ReachedContext(Exception):
        pass
    def prepare(observations):
        assert observations.speed_kms.tolist()==[400.]
        raise ReachedContext
    with pytest.raises(ReachedContext):
        publish.publish_result(SimpleNamespace(prepare_context=prepare), {42:run,43:run,44:run}, tmp_path/'unused.parquet')


def test_replay_compiles_and_accepts_help(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, 'ROOT', tmp_path)
    (tmp_path / 'ensemble.py').write_text('print("mean")\n')
    runs = {}
    for seed in publish.SEEDS:
        run = tmp_path / str(seed)
        run.mkdir()
        for module in publish.MODULES:
            (run / module).write_text('print("frozen module")\n')
        runs[seed] = run
    replay = tmp_path / 'replay.py'
    replay.write_bytes(publish.replay_source(runs))
    result = subprocess.run([sys.executable, str(replay), '--help'], capture_output=True, text=True, check=False)
    assert result.returncode == 0 and '--data' in result.stdout
    assert 'frozen module' not in result.stdout


def test_visualization_resume_requires_complete_matching_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(visualization, 'OUT', tmp_path)
    assert not visualization.valid_manifest('fixed-time')
    outputs = []
    for month in range(1, 10):
        name = f'forecast_{month}.gif'
        (tmp_path / name).write_bytes(b'GIF')
        outputs.append({'file': name, 'bytes': 3})
    for name in ('README.md', 'leaderboard-result.json', 'lead_curves.png'):
        (tmp_path / name).write_text('fixture')
    manifest = {'experiment': publish.EXPERIMENT, 'submitted_utc': 'fixed-time', 'outputs': outputs,
                    'verification': {'complete': True, 'all_equal_to_record': True, 'rendered_frames': 1052}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    assert visualization.valid_manifest('fixed-time')
    assert not visualization.valid_manifest('different-time')
    (tmp_path / outputs[0]['file']).write_bytes(b'incomplete')
    assert not visualization.valid_manifest('fixed-time')


def complete_runs(tmp_path):
    runs = {}
    for seed in publish.SEEDS:
        run = tmp_path / 'runs' / f'drop035_v650_s{seed}'
        run.mkdir(parents=True)
        resolved = {'exclude_features': publish.EXCLUDED, 'tail_weight': 4, 'spectral': True, 'offset': True,
                        'offset_scope': 'horizon', 'offset_window': 336, 'alpha': .25, 'block_size': 24,
                        'clip_component': True, 'clip_final': True, 'seed': seed,
                        'spectral_hessian_only': False, 'experiment': run.name}
        publish.write(run / 'config.json', resolved)
        publish.write(run / 'resolved_config.json', resolved)
        (run / 'train.exit').write_text('0\n')
        publish.write(run / 'training-validation.json', {'rows': 454680})
        for module in (*publish.MODULES, 'runner_at_launch.py'):
            (run / module).write_text('# frozen source\n')
        for period, cutoff in publish.CUTOFFS.items():
            publish.write(run / f'{period}_metadata.json', {
                'period': period, 'smoke': False, 'training_cutoff': cutoff, 'maturity_hours': 144,
                'seed': seed, 'feature_count': 28, 'rounds': 900, 'config': resolved,
                'features': {'constant': publish.RETAINED_FEATURES},
                'parameters': {'device': 'cpu'}, 'inference_seconds_per_fold': .1, 'query_origins': 100})
        runs[seed] = run
    return runs


def test_exact_two_removal_config_and_four_publication_items(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, 'ROOT', tmp_path)
    monkeypatch.setattr(publish, 'WORK', tmp_path)
    runs = complete_runs(tmp_path)
    assert publish.validate_runs() == runs
    (tmp_path / 'ensemble.py').write_text('# fixed mean\n')
    publish.write(tmp_path / 'ensemble.json', {'prediction_mean_seconds_per_fold': .001})
    items = publish.publication_items(runs, tmp_path / 'predictions.parquet', 'pinned-revision')
    assert len(items) == 4
    assert [item[0]['experiment'] for item in items] == [
        f'e2000fs_calib_offset_roll_drop_t035_v650_s{s}' for s in (42, 43, 44)
    ] + [publish.EXPERIMENT]
    assert items[-1][0]['config']['hyperparameters']['weights'] == [1 / 3] * 3
    for _, _, replay, _ in items:
        compile(replay, 'replay.py', 'exec')


@pytest.mark.parametrize('mutation', ['alpha', 'hessian', 'rounds', 'snapshot', 'extra_config'])
def test_rejects_unrequested_training_changes(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(publish, 'ROOT', tmp_path)
    runs = complete_runs(tmp_path)
    run = runs[42]
    if mutation == 'snapshot':
        (run / 'runner_at_launch.py').write_text('# changed\n')
    elif mutation == 'extra_config':
        path = run / 'config.json'
        publish.write(path, dict(publish.read(path), offset=False))
    else:
        path = run / 'F1_metadata.json'
        report = publish.read(path)
        if mutation == 'rounds':
            report['rounds'] = 2
        elif mutation == 'alpha':
            report['config']['alpha'] = 1
        else:
            report['config']['spectral_hessian_only'] = True
        publish.write(path, report)
    with pytest.raises(AssertionError):
        publish.validate_runs()


def test_recipe_only_config_is_not_a_valid_launch_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, 'ROOT', tmp_path)
    runs = complete_runs(tmp_path)
    publish.write(runs[42] / 'config.json', {'seed': 42, 'exclude_features': publish.EXCLUDED})
    with pytest.raises(AssertionError):
        publish.validate_runs()
