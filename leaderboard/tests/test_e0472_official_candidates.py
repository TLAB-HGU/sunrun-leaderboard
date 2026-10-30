import copy
import json
from pathlib import Path

import pytest

from experiments.e0472_ablation_hpo import campaign as C
from experiments.e0472_ablation_hpo import official_candidates as O


def frozen_fixture(tmp_path, monkeypatch):
    source_root = tmp_path / 'source'
    source_root.mkdir()
    configs = {}
    source_configs = {}
    for name in O.NAMES:
        config = copy.deepcopy(C.DEFAULT_CONFIG)
        config['name'] = name
        path = source_root / f'{name}.json'
        C.write_json(path, config)
        configs[name] = dict(config=config, hash=C.digest(config), source_path=str(path))
        source_configs[name] = dict(config=config, kind='ablation')
    source = dict(started_at=1000, max_hours=72, configs=source_configs, units={})
    C.write_json(source_root / 'state.json', source)
    request = dict(candidates=list(O.CANDIDATES), baseline='baseline', configs=configs,
                   seeds=[0, 1, 2], source_code_hashes={'helper.py': 'unchanged'},
                   deadline_epoch=1000+72*3600, policy=dict(official_only=True,
                   official_fit_target_before='2026-05-29', n_jobs=5, max_parallel=2,
                   submit=False, lb_eval_once_per_experiment=True))
    request_path = tmp_path / 'request.json'
    C.write_json(request_path, request)
    monkeypatch.setattr(C, 'code_hashes', lambda: {'helper.py': 'unchanged'})
    return request_path, source_root, request


def test_exact_candidates_configs_and_original_deadline(tmp_path, monkeypatch):
    path, source, request = frozen_fixture(tmp_path, monkeypatch)
    parsed, _ = O.frozen_request(path, source)
    assert parsed['candidates'] == list(O.CANDIDATES)
    request['candidates'][-1] = 'xgb_004'
    C.write_json(path, request)
    with pytest.raises(ValueError, match='exactly'):
        O.frozen_request(path, source)


@pytest.mark.parametrize('mutation', ['deadline', 'config', 'source', 'seeds'])
def test_frozen_request_rejects_mutation(tmp_path, monkeypatch, mutation):
    path, source, request = frozen_fixture(tmp_path, monkeypatch)
    if mutation == 'deadline':
        request['deadline_epoch'] += 1
    elif mutation == 'config':
        request['configs']['xgb_003']['config']['k'] = 60
    elif mutation == 'source':
        request['source_code_hashes']['helper.py'] = 'changed'
    else:
        request['seeds'] = [0]
    C.write_json(path, request)
    with pytest.raises(ValueError):
        O.frozen_request(path, source)


def test_official_authorization_has_no_retired_confirmation_gate(tmp_path):
    promotion = O.OfficialPromotion(O.parser().parse_args(['--status', '--root', str(tmp_path)]))
    promotion.state = {'locked': list(O.CANDIDATES), 'units': {}}
    assert promotion.full_ready('drop_strip', 'official', ['official'])
    assert not promotion.full_ready('xgb_004', 'official', ['official'])
    assert not promotion.full_ready('drop_strip', 'confirm', ['W0', 'W1'])
    with pytest.raises(RuntimeError, match='fixed three-seed'):
        promotion.prepare_unit('drop_strip', 'official', ['official'], [0])


def test_observation_preserves_fit_arguments_return_and_restores_method():
    calls = []
    class Booster:
        def save_raw(self):
            return b'actual-model'
    class Model:
        def fit(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self
        def get_params(self, deep=False):
            return dict(random_state=4, n_jobs=5, device='cpu')
        def get_booster(self):
            return Booster()
    original = Model.fit
    model, x, y, traces = Model(), object(), object(), []
    with O.record_actual_fits(Model, traces):
        assert model.fit(x, y, verbose=False) is model
    assert Model.fit is original
    assert calls == [((x, y), {'verbose': False})]
    assert traces[0]['parameters']['random_state'] == 4
    assert traces[0]['booster_sha256'] == O.hashlib.sha256(b'actual-model').hexdigest()


def test_fit_observation_restores_after_error():
    class Model:
        def fit(self):
            raise ValueError('fit failed')
    original = Model.fit
    with pytest.raises(ValueError):
        with O.record_actual_fits(Model, []):
            Model().fit()
    assert Model.fit is original


def summary(mse=1000, f1=.3, cell=2000):
    return dict(summary=dict(n_seeds=3, mse_by_seed=[mse-1, mse, mse+1],
        mse_seedmean=mse, f1_seedmean=f1, cells_seedmean={c: cell for c in O.HIGH_CELLS}))


def test_adoption_uses_all_four_cells_f1_and_noise_floor():
    baseline = summary()
    assert O.comparison(summary(950), baseline)['verdict'] == 'small_change'
    assert O.comparison(summary(800), baseline)['established_improvement']
    worse_cell = summary(800)
    worse_cell['summary']['cells_seedmean']['low_high_high'] = 2001
    assert not O.comparison(worse_cell, baseline)['metrics_gate_pass']
    assert not O.comparison(summary(800, .29), baseline)['metrics_gate_pass']


def evaluation_fixture(tmp_path, monkeypatch):
    from expos import core
    promotion = O.OfficialPromotion(O.parser().parse_args(['--status', '--root', str(tmp_path)]))
    promotion.state = dict(started_at=C.time.time(), max_hours=72, units={'drop_strip:official:official': {'status': 'complete', 'experiment': 'TEST'}},
                           official_evaluations={})
    graph = {'experiments': {'TEST': {'lbeval': []}}}
    monkeypatch.setattr(core, 'state', lambda: graph)
    monkeypatch.setattr(promotion, 'verified_result', lambda record: {})
    monkeypatch.setattr(promotion, 'event', lambda *args: None)
    return promotion, graph


def test_official_request_is_persisted_before_single_side_effect(tmp_path, monkeypatch):
    promotion, graph = evaluation_fixture(tmp_path, monkeypatch)
    calls = []
    def cli(*args, **kwargs):
        assert C.read_json(promotion.path)['official_evaluations']['drop_strip']['status'] == 'requested'
        calls.append(args)
        graph['experiments']['TEST']['lbeval'].append(summary())
    monkeypatch.setattr(promotion, 'cli', cli)
    promotion.evaluate_once('drop_strip')
    promotion.evaluate_once('drop_strip')
    assert calls == [('lb-eval', 'TEST')]
    assert promotion.state['official_evaluations']['drop_strip']['status'] == 'complete'


def test_uncertain_official_request_is_not_repeated(tmp_path, monkeypatch):
    promotion, graph = evaluation_fixture(tmp_path, monkeypatch)
    promotion.state['official_evaluations']['drop_strip'] = {'status': 'requested'}
    monkeypatch.setattr(promotion, 'cli', lambda *a, **k: pytest.fail('must not score twice'))
    promotion.evaluate_once('drop_strip')
    assert promotion.state['official_evaluations']['drop_strip']['status'] == 'requested'


def test_duplicate_evaluation_history_rejected(tmp_path, monkeypatch):
    promotion, graph = evaluation_fixture(tmp_path, monkeypatch)
    graph['experiments']['TEST']['lbeval'] = [summary(), summary()]
    with pytest.raises(RuntimeError, match='ambiguous'):
        promotion.evaluate_once('drop_strip')


def test_dry_run_does_not_write_promotion_or_source(tmp_path, monkeypatch, capsys):
    path, source, request = frozen_fixture(tmp_path, monkeypatch)
    before = C.file_hash(source / 'state.json')
    root = tmp_path / 'promotion'
    O.main(['--dry-run', '--root', str(root), '--request', str(path), '--source-root', str(source)])
    assert not root.exists()
    assert C.file_hash(source / 'state.json') == before
    result = json.loads(capsys.readouterr().out)
    assert result['official_only'] and result['seeds'] == [0, 1, 2]
    assert result['max_parallel'] == 2 and result['n_jobs'] == 5


def test_register_routes_to_audited_producer_and_binds_frozen_files(tmp_path, monkeypatch):
    promotion = O.OfficialPromotion(O.parser().parse_args(['--status', '--root', str(tmp_path)]))
    promotion.state = {'request_path': str(tmp_path / 'request.json')}
    seen = []
    monkeypatch.setattr(C.Campaign, 'cli', lambda self, *args, **kw: seen.append(args))
    command = 'python -m experiments.e0472_ablation_hpo.campaign --execute-unit /tmp/unit.json'
    promotion.cli('register', '--cmd', command, '--code', 'source.py')
    assert '-m ' + O.MODULE in seen[0][2]
    assert seen[0][-2:] == (str(tmp_path / 'request.json'), str(tmp_path / 'frozen_manifest.json'))


def test_official_score_does_not_start_after_original_deadline(tmp_path, monkeypatch):
    promotion, graph = evaluation_fixture(tmp_path, monkeypatch)
    promotion.state.update(started_at=0, max_hours=72)
    monkeypatch.setattr(promotion, 'cli', lambda *a, **k: pytest.fail('budget exhausted'))
    promotion.evaluate_once('drop_strip')
    assert promotion.state['official_evaluations']['drop_strip']['status'] == 'budget_exhausted'
