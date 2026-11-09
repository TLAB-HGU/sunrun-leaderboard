"""Publication invariants, tested without training or external writes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('lag_publication_under_test', ROOT / 'publish.py')
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


class FakeCore:
    baseline = SimpleNamespace(PARENT_NAMES={'constant': [f'c{i}' for i in range(20)],
                                            'varying': [f'e{i}' for i in range(8)]})

    @staticmethod
    def normalize_config(config):
        return dict(config)

    @classmethod
    def feature_names(cls, config):
        return {'constant': cls.baseline.PARENT_NAMES['constant'] + config['add_features'],
                'varying': cls.baseline.PARENT_NAMES['varying']}

    @classmethod
    def expanded_names(cls, names, start, size):
        return (names['constant'][:20] + [f'{n}@h{h + 1}' for h in range(start, start + size)
                                        for n in names['varying']] + names['constant'][20:])


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(publish, 'ROOT', tmp_path)
    monkeypatch.setattr(publish, 'WORK', tmp_path / 'publication')
    monkeypatch.setattr(publish, 'load_core', lambda: FakeCore)
    recipes = [dict(experiment=f'lag24_48_72_s{seed}', leaderboard_experiment=f'published_lags_s{seed}',
                    seed=seed, config={'seed': seed, 'add_features': publish.ADDITIONS}) for seed in publish.SEEDS]
    publish.write(tmp_path / 'assignment.global.json', dict(experiments=recipes, data='/frozen/input.parquet',
                                                         observation_revision='frozen-revision'))
    publish.write(tmp_path / 'provenance.json', {'parent': 'audited_leader_s42'})
    for name in publish.MODULES[1:]:
        (tmp_path / name).write_text('{}' if name.endswith('.json') else '# frozen source\n')
    runs = {}
    for recipe in recipes:
        run = tmp_path / 'runs' / recipe['experiment']
        run.mkdir(parents=True)
        for name in publish.MODULES[1:]:
            (run / name).write_bytes((tmp_path / name).read_bytes())
        for name in ('experiment.py', 'runner_at_launch.py'):
            (run / name).write_text('# frozen entry\n')
        (run / 'train.exit').write_text('0\n')
        expected, names, expanded = publish.expected_contract(recipe, FakeCore)
        publish.write(run / 'config.json', expected)
        publish.write(run / 'resolved_config.json', expected)
        publish.write(run / 'training-validation.json', dict(rows=454680, saved_model_replay={
            p: dict(rounds=900, boosters=3, max_abs_error=0.0) for p in publish.CUTOFFS}))
        for period, cutoff in publish.CUTOFFS.items():
            publish.write(run / f'{period}_metadata.json', dict(smoke=False, training_cutoff=cutoff,
                maturity_hours=144, seed=recipe['seed'], rounds=900, config=expected, features=names,
                expanded_features=expanded, feature_count=31, expanded_feature_count=215,
                parameters={'device': 'cpu'}, inference_seconds_per_fold=.1, query_origins=2000))
        runs[recipe['seed']] = run
    return tmp_path, runs


def test_all_three_fresh_full_runs_and_exact_union_are_required(campaign):
    _, runs = campaign
    assert publish.validate_runs() == runs
    for run in runs.values():
        meta = publish.seed_metadata(run, 'pinned-revision')
        assert meta['config']['hyperparameters']['feature_count'] == 31
        assert meta['config']['hyperparameters']['expanded_feature_count'] == 215
        assert meta['config']['hyperparameters']['add_features'] == publish.ADDITIONS
        assert meta['config']['data']['parent'] == 'audited_leader_s42'
        assert 'Exploratory' in meta['description'] and 'seed42' in meta['description']
        assert 'aggregation' not in meta['config']['hyperparameters']
        assert '/published/core.py' in meta['architecture_url']
    assert len({publish.seed_metadata(r, 'revision')['experiment'] for r in runs.values()}) == 3


@pytest.mark.parametrize('mutation', ['unfinished', 'reference', 'replay', 'snapshot', 'rounds',
                                    'count', 'ordering', 'resolved_config', 'extra_feature'])
def test_publication_rejects_incomplete_or_changed_training(campaign, mutation):
    root, runs = campaign
    run = runs[42]
    if mutation == 'unfinished':
        (run / 'train.exit').unlink()
    elif mutation in ('reference', 'replay'):
        path = run / 'training-validation.json'
        value = publish.read(path)
        if mutation == 'reference':
            value['reference_only'] = True
        else:
            value['saved_model_replay']['F3']['max_abs_error'] = .01
        publish.write(path, value)
    elif mutation == 'snapshot':
        (run / 'runner_at_launch.py').write_text('# changed\n')
    elif mutation == 'resolved_config':
        publish.write(run / 'resolved_config.json', {})
    elif mutation == 'extra_feature':
        path = root / 'assignment.global.json'
        value = publish.read(path)
        value['experiments'][0]['config']['add_features'] = ['v_lag24']
        publish.write(path, value)
    else:
        path = run / 'F1_metadata.json'
        value = publish.read(path)
        if mutation == 'rounds':
            value['rounds'] = 2
        elif mutation == 'count':
            value['feature_count'] = 28
        else:
            value['expanded_features']['0'] = value['expanded_features']['0'][::-1]
        publish.write(path, value)
    with pytest.raises((ValueError, RuntimeError)):
        publish.validate_runs()
    assert not (root / 'campaign-complete.json').exists()


def test_existing_result_is_idempotent_and_rejects_collision():
    config = {'fixed': True}
    row = {'team_member': publish.MEMBER, 'experiment': 'one', 'config': config}
    assert publish.existing_result([], config, 'one') is None
    assert publish.existing_result([row], config, 'one') is row
    with pytest.raises(RuntimeError):
        publish.existing_result([row, row], config, 'one')
    with pytest.raises(RuntimeError):
        publish.existing_result([row], {'fixed': False}, 'one')


def test_score_parity_checks_each_period_and_all_metrics():
    scores = {scope: {metric: 10.0 for metric in publish.METRICS} for scope in publish.SCOPES}
    publish.require_score_parity(scores, scores)
    wrong = json.loads(json.dumps(scores))
    wrong['f3']['timse'] += .01
    with pytest.raises(ValueError, match='f3/timse'):
        publish.require_score_parity(wrong, scores)
    wrong = json.loads(json.dumps(scores))
    wrong['f1']['n_replaced'] = 1
    with pytest.raises(ValueError, match='replaced'):
        publish.require_score_parity(wrong, scores)


def test_replay_contains_and_restores_frozen_inventory(campaign):
    root, runs = campaign
    run = runs[42]
    (run / 'inventory.json').write_text('{"frozen": true}')
    (run / 'experiment.py').write_text('import json,os\nfrom pathlib import Path\n'
        'assert json.loads(Path(__file__).with_name("inventory.json").read_text())["frozen"]\n'
        'assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\nprint("inventory restored")\n')
    replay = root / 'replay.py'
    replay.write_bytes(publish.seed_replay(run))
    result = subprocess.run([sys.executable, str(replay)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'inventory restored' in result.stdout


def test_registration_posts_three_explicit_seed_names(campaign, monkeypatch):
    root, _ = campaign
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='registered', stderr='')
    monkeypatch.setattr(publish.subprocess, 'run', run)
    rows = publish.register('대기')
    assert len(rows) == len(calls) == 3
    assert {row['experiment'] for row in rows} == {f'published_lags_s{s}' for s in publish.SEEDS}
    assert all(command[command.index('--status') + 1] == '대기' for command in calls)
    assert (root / 'publication/registration-대기.json').is_file()


def load_controller():
    spec = importlib.util.spec_from_file_location('lag_publication_controller_under_test', ROOT / 'publication_controller.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analysis_receipt():
    return dict(status='complete', seeds=[42, 43, 44], add_features=publish.ADDITIONS,
                bootstrap={'replicates': 2000}, gate={'passed': False})


def test_controller_requires_resource_admission_before_analysis(tmp_path, monkeypatch):
    from contextlib import nullcontext
    controller = load_controller()
    monkeypatch.setattr(controller, 'analysis_slot', lambda: nullcontext(False))
    calls = []
    monkeypatch.setattr(controller.subprocess, 'run', lambda *a, **k: calls.append(a))
    assert controller.attempt_publication(tmp_path) == 'waiting_for_analysis_resources'
    assert calls == []


def test_controller_analyzes_before_publication_even_if_performance_gate_fails(tmp_path, monkeypatch):
    from contextlib import nullcontext
    controller = load_controller()
    monkeypatch.setattr(controller, 'analysis_slot', lambda: nullcontext(True))
    calls = []
    def run(command, **kwargs):
        calls.append(Path(command[2]).name)
        if 'analyze_results.py' in command[2]:
            assert command[-2:] == ['--bootstrap', '2000']
            publish.write(tmp_path / 'analysis/results.json', analysis_receipt())
        else:
            assert controller.analysis_ready(tmp_path)
            publish.write(tmp_path / 'campaign-complete.json', {'complete': True})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(controller.subprocess, 'run', run)
    assert controller.attempt_publication(tmp_path) == 'published'
    assert calls == ['analyze_results.py', 'publish.py']


def test_controller_cannot_publish_after_analysis_failure(tmp_path, monkeypatch):
    from contextlib import nullcontext
    controller = load_controller()
    monkeypatch.setattr(controller, 'analysis_slot', lambda: nullcontext(True))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(controller.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='analysis failed'):
        controller.attempt_publication(tmp_path)
    assert len(calls) == 1 and 'analyze_results.py' in calls[0][2]


def remote_receipt_fixture(campaign):
    root, runs = campaign
    assignment = publish.read(root / 'assignment.global.json')
    assignment['host'] = 'golab2'
    publish.write(root / 'assignment.json', assignment)
    payload = []
    scores = {scope: dict(s=10.0, rmse=11.0, trmse=12.0, timse=144.0, n_replaced=0)
              for scope in publish.SCOPES}
    for recipe in assignment['experiments']:
        run = runs[recipe['seed']]
        config = publish.read(run / 'config.json')
        publish.write(run / 'launch.json', dict(campaign_root=str(root.resolve()), experiment=run.name,
                                               training_host='golab2'))
        publish.write(run / 'local_score.json', {'competition': scores})
        publish.write(run / 'submission.json', {'competition': scores, 'local_only': True, 'published': False})
        publish.write(run / 'complete.json', {'publish': 'pending', 'training_evidence': 'retain'})
        record = dict(experiment=recipe['leaderboard_experiment'], team_member=publish.MEMBER,
                      config={'hyperparameters': {'run_config': config}}, competition=scores,
                      submitted_utc=f'time-{recipe["seed"]}', architecture_url='https://example.test/pinned/core.py',
                      local_only=False, published=True)
        payload.append(dict(run=run.name, config=config, record=record))
    return root, runs, payload


def test_remote_receipt_sync_is_exact_scoped_and_idempotent(campaign):
    root, runs, payload = remote_receipt_fixture(campaign)
    originals = {seed: (run / 'submission.json').read_bytes() for seed, run in runs.items()}
    first = publish.apply_remote_receipts(root, payload)
    second = publish.apply_remote_receipts(root, payload)
    assert first == second and first['verified'] and len(first['records']) == 3
    for seed, run in runs.items():
        assert (run / 'submission.local.json').read_bytes() == originals[seed]
        assert publish.read(run / 'submission.json')['published'] is True
        complete = publish.read(run / 'complete.json')
        assert complete['publish'] == 'complete' and complete['training_evidence'] == 'retain'


@pytest.mark.parametrize('mutation', ['wrong_owner', 'changed_config', 'changed_score', 'missing_seed', 'existing_other_result'])
def test_remote_sync_rejects_all_before_any_receipt_mutation(campaign, mutation):
    root, runs, payload = remote_receipt_fixture(campaign)
    if mutation == 'wrong_owner':
        path = runs[44] / 'launch.json'
        publish.write(path, dict(publish.read(path), campaign_root='/different/campaign'))
    elif mutation == 'changed_config':
        path = runs[44] / 'config.json'
        publish.write(path, dict(publish.read(path), seed=42))
    elif mutation == 'changed_score':
        payload[2]['record'] = json.loads(json.dumps(payload[2]['record']))
        payload[2]['record']['competition']['f2']['rmse'] += 1
    elif mutation == 'missing_seed':
        payload = payload[:2]
    else:
        other = dict(payload[2]['record'], submitted_utc='different-post')
        publish.write(runs[44] / 'submission.json', other)
    before = {seed: (run / 'submission.json').read_bytes() for seed, run in runs.items()}
    with pytest.raises(ValueError):
        publish.apply_remote_receipts(root, payload)
    for seed, run in runs.items():
        assert (run / 'submission.json').read_bytes() == before[seed]
        assert not (run / 'submission.local.json').exists()


def test_remote_sync_function_runs_as_standalone_transported_code(campaign):
    import inspect
    root, _, payload = remote_receipt_fixture(campaign)
    source = inspect.getsource(publish.apply_remote_receipts)
    source += '\nimport json\nprint(json.dumps(apply_remote_receipts(' + repr(str(root)) + ', ' + repr(payload) + ')))\n'
    result = subprocess.run([sys.executable, '-'], input=source, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    proof = json.loads(result.stdout)
    assert proof['verified'] and len(proof['records']) == 3
