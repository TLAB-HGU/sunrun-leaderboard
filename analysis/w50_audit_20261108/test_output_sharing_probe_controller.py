import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def controller():
    spec = importlib.util.spec_from_file_location('sharing_controller_test', Path(__file__).with_name('output_sharing_probe_controller.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_statistics_match_brute_force_masks_and_permutations(controller):
    rng = np.random.default_rng(391)
    x = rng.normal(size=(96, 4)).astype(np.float32)
    x[[0, 7, 22], 0] = np.nan
    x[[2, 8, 14], 2] = np.nan
    g = rng.normal(size=(96, 3)).astype(np.float32)
    h = rng.uniform(.1, 3., (96, 3)).astype(np.float32)
    ids = np.arange(96, dtype=np.int64) + 900
    result = controller.candidate_statistics(x, g, h, row_ids=ids, derivative_row_ids=ids)
    previous = (-1, -np.inf, -1)
    for i, (feature, cut, missing_left) in enumerate(zip(result['feature_indices'], result['thresholds'], result['missing_left'], strict=True)):
        key = (feature, cut, 0 if missing_left else 1)
        assert key > previous
        previous = key
        mask = (x[:, feature] < cut) | (np.isnan(x[:, feature]) & missing_left)
        assert result['left_counts'][i] == mask.sum()
        assert result['right_counts'][i] == (~mask).sum()
        np.testing.assert_allclose(result['left_gradient'][i], g[mask].sum(axis=0, dtype=np.float64), rtol=0, atol=1e-12)
        np.testing.assert_allclose(result['left_hessian'][i], h[mask].sum(axis=0, dtype=np.float64), rtol=0, atol=1e-12)
        np.testing.assert_allclose(result['right_gradient'][i], g[~mask].sum(axis=0, dtype=np.float64), rtol=0, atol=1e-12)
        np.testing.assert_allclose(result['right_hessian'][i], h[~mask].sum(axis=0, dtype=np.float64), rtol=0, atol=1e-12)
    permutation = rng.permutation(len(x))
    permuted = controller.candidate_statistics(x[permutation], g[permutation], h[permutation],
                                                row_ids=ids[permutation], derivative_row_ids=ids[permutation])
    for key in result:
        if key == 'feature_counts':
            assert result[key] == permuted[key]
        else:
            np.testing.assert_allclose(result[key], permuted[key], rtol=0, atol=1e-12)


def test_float32_ceiling_preserves_float64_quantile_partitions(controller):
    low = np.float32(1.)
    high = np.nextafter(low, np.float32(np.inf))
    x = np.r_[np.full(32, low), np.full(32, high)].reshape(-1, 1)
    g, h = np.ones((64, 2), np.float32), np.ones((64, 2), np.float32)
    result = controller.candidate_statistics(x, g, h, row_ids=np.arange(64), derivative_row_ids=np.arange(64))
    assert result['thresholds'].tolist() == [high]
    assert result['left_counts'].tolist() == [32]
    q = np.quantile(x[:, 0].astype(float), .5, method='linear')
    assert low < q < high
    np.testing.assert_array_equal(x[:, 0] < result['thresholds'][0], x[:, 0].astype(float) < q)
    assert result['missing_left'].tolist() == [True]


def test_duplicates_constant_all_missing_and_no_missing_only_cuts(controller):
    x = np.array([[1., 7., np.nan], [1., 7., np.nan], [2., 7., np.nan], [2., np.nan, np.nan]], np.float32)
    g = np.ones((4, 2), np.float32)
    result = controller.candidate_statistics(x, g, g, row_ids=np.arange(4), derivative_row_ids=np.arange(4))
    assert result['feature_indices'].tolist() == [0]
    assert result['feature_counts'][0]['duplicate_partitions_dropped'] > 0
    assert result['feature_counts'][1]['retained_candidates'] == 0
    assert result['feature_counts'][2]['retained_candidates'] == 0
    empty = controller.candidate_statistics(x[:, 1:], g, g, row_ids=np.arange(4), derivative_row_ids=np.arange(4))
    assert empty['left_gradient'].shape == (0, 2)
    assert empty['thresholds'].shape == (0,)


@pytest.mark.parametrize('kind', ['rows', 'identity', 'duplicate_identity', 'nan_gradient', 'infinite_feature', 'negative_hessian', 'dtype'])
def test_invalid_candidate_alignment_or_statistics_rejected(controller, kind):
    x = np.ones((4, 2), np.float32)
    g, h = x.copy(), x.copy()
    ids, derivative_ids = np.arange(4), np.arange(4)
    if kind == 'rows':
        g = g[:3]
    elif kind == 'identity':
        derivative_ids = derivative_ids[::-1]
    elif kind == 'duplicate_identity':
        ids = derivative_ids = np.array([0, 0, 2, 3])
    elif kind == 'nan_gradient':
        g[0, 0] = np.nan
    elif kind == 'infinite_feature':
        x[0, 0] = np.inf
    elif kind == 'negative_hessian':
        h[0, 0] = -1
    else:
        x = x.astype(float)
    with pytest.raises(ValueError):
        controller.candidate_statistics(x, g, h, row_ids=ids, derivative_row_ids=derivative_ids)


@pytest.fixture
def contracts(controller, tmp_path):
    runner = controller.module(controller.REPO / 'research/w50_roundselect_20261108/experiment.py', 'synthetic_sharing_loader')
    baseline, core = runner.parent_modules()
    config = runner.fixed_config(core, 42, 'drop035_v650_s42')
    names = core.feature_names(config)
    origins = pd.date_range('2020-01-01', periods=4, freq='h', tz='UTC')
    train = {'origins': origins, 'C': np.ones((4, 20), np.float32), 'E': np.ones((4, 72, 8), np.float32)}
    target = np.full((4, 72), 400.)
    data = tmp_path / 'synthetic-data.parquet'
    data.write_bytes(b'no actual parquet is opened')
    receipt = controller.identity(data)
    reconstructed = {'xgboost': '2.1.4', 'trees_created': 0, 'parameters': dict(baseline.PARAMS, seed=42, nthread=4),
                     'effective_train_parameters': dict.fromkeys(('alpha', 'reg_alpha', 'gamma', 'min_split_loss', 'max_delta_step'), '0')}
    reconstructed['effective_train_parameters'].update({'lambda': '5', 'reg_lambda': '5', 'min_child_weight': '20'})
    metadata = {'parameters': reconstructed['parameters'].copy(), 'config': config, 'rounds': 900, 'smoke': False,
                'seed': 42, 'period': 'F1', 'features': names, 'expanded_features': {str(s): core.expanded_names(names, s, 24) for s in (0, 24, 48)},
                'feature_count': 28, 'expanded_feature_count': 212, 'identity': {'data': receipt},
                'training_cutoff': baseline.CUTOFFS['F1'], 'maturity_hours': 144,
                'train_origins': 4, 'train_start': str(origins.min()), 'train_end': str(origins.max())}
    return runner, baseline, core, config, names, train, target, data, reconstructed, metadata


def test_valid_metadata_contract(controller, contracts):
    _, _, core, config, names, train, target, data, reconstructed, metadata = contracts
    controller.validate_period(core, config, names, metadata, train, target, 'F1', controller.identity(data), reconstructed)


@pytest.mark.parametrize('kind', ['parameters', 'maturity', 'features', 'identity'])
def test_metadata_and_maturity_guards_precede_saved_model_access(controller, contracts, tmp_path, monkeypatch, kind):
    runner, baseline, core, config, names, train, target, data, reconstructed, metadata = contracts
    if kind == 'parameters':
        metadata['parameters']['max_depth'] = 9
    elif kind == 'maturity':
        train['origins'] = pd.date_range('2025-12-28', periods=4, freq='h', tz='UTC')
    elif kind == 'features':
        metadata['feature_count'] = 29
    else:
        metadata['identity']['data'] = dict(metadata['identity']['data'], bytes=999)
    run = tmp_path / 'frozen-parent'
    run.mkdir()
    for source in (Path(baseline.__file__), Path(core.__file__)):
        (run / source.name).write_bytes(source.read_bytes())
    (run / 'F1_metadata.json').write_text(json.dumps(metadata))
    reconstructed_path = tmp_path / 'reconstructed-config.json'
    reconstructed_path.write_text(json.dumps(reconstructed))
    seal = tmp_path / 'training-fit-probe.json'
    seal.write_text(json.dumps({'seed': 42, 'features': names, 'config': config,
                                'input_receipts': {'prepared': controller.identity(data)}}))
    monkeypatch.setattr(controller, 'RUN', run)
    monkeypatch.setattr(controller, 'CONFIG_RECEIPT', reconstructed_path)
    monkeypatch.setattr(controller, 'FIT_RECEIPT', seal)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(controller.xgb, '__version__', '2.1.4')
    monkeypatch.setattr(controller.xgb, 'Booster', lambda *a, **k: pytest.fail('Model accessed before validation'))
    monkeypatch.setattr(baseline, 'load_grid', lambda data, period, names, training: (train, target) if training is True else pytest.fail('Query load'))
    loader = SimpleNamespace(parent_modules=lambda: (baseline, core), fixed_config=runner.fixed_config,
                             assert_parent_contract=lambda *a: names)
    monkeypatch.setattr(controller, 'module', lambda path, name: loader if path.name == 'experiment.py' else SimpleNamespace())
    with pytest.raises(ValueError):
        controller.main(['--data', str(data), '--out', str(tmp_path / 'out'), '--device', 'cpu', '--threads', '1'])
    assert not (tmp_path / 'out/report.json').exists()
    assert not list((tmp_path / 'out').rglob('*.npz'))


def test_completed_output_is_not_overwritten(controller, tmp_path, monkeypatch):
    out = tmp_path / 'out'
    out.mkdir()
    report = out / 'report.json'
    report.write_text('existing completion')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(controller.xgb, '__version__', '2.1.4')
    monkeypatch.setattr(controller, 'module', lambda *a: pytest.fail('Loader accessed for completed output'))
    with pytest.raises(ValueError, match='must not be overwritten'):
        controller.main(['--out', str(out), '--device', 'cpu', '--threads', '1'])
    assert report.read_text() == 'existing completion'


@pytest.mark.parametrize('change_input_during_inference', [False, True])
def test_synthetic_controller_writes_all18_contexts_without_fitting(controller, contracts, tmp_path, monkeypatch, change_input_during_inference):
    runner, baseline, core, config, names, train, target, data, reconstructed, metadata = contracts
    run = tmp_path / 'parent'
    run.mkdir()
    for source in (Path(baseline.__file__), Path(core.__file__)):
        (run / source.name).write_bytes(source.read_bytes())
    sealed = {'seed': 42, 'features': names, 'config': config,
              'input_receipts': {'prepared': controller.identity(data)}, 'periods': {}}
    for period in ('F1', 'F2', 'F3'):
        meta = dict(metadata, period=period, training_cutoff=baseline.CUTOFFS[period])
        (run / f'{period}_metadata.json').write_text(json.dumps(meta))
        heads = []
        for block in range(3):
            path = run / f'{period}_models/full/b{block}.ubj'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'synthetic model placeholder; fake loader only')
            heads.append({'block': block, 'model': controller.identity(path)})
        sealed['periods'][period] = {'heads': heads}
    config_path, seal_path = tmp_path / 'config.json', tmp_path / 'seal.json'
    config_path.write_text(json.dumps(reconstructed))
    seal_path.write_text(json.dumps(sealed))
    monkeypatch.setattr(controller, 'CONFIG_RECEIPT', config_path)
    monkeypatch.setattr(controller, 'FIT_RECEIPT', seal_path)
    monkeypatch.setattr(controller, 'RUN', run)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(controller.xgb, '__version__', '2.1.4')
    monkeypatch.setattr(baseline, 'load_grid', lambda data, period, names, training:
                        (train, target) if training is True else pytest.fail('Query read'))
    actual_module = controller.module
    loader = SimpleNamespace(parent_modules=lambda: (baseline, core), fixed_config=runner.fixed_config,
                             assert_parent_contract=lambda *a: names)
    monkeypatch.setattr(controller, 'module', lambda path, name:
                        loader if path.name == 'experiment.py' else actual_module(path, name))
    calls = []

    class FakeBooster:
        def __init__(self, params):
            assert params == {'device': 'cpu', 'nthread': 1}

        def load_model(self, path):
            calls.append(('load', path))

        def set_param(self, params):
            assert params == {'device': 'cpu', 'nthread': 1}

        def save_config(self):
            return json.dumps({'learner': {'learner_model_param': {'num_target': '24'}}})

        def num_boosted_rounds(self):
            return 900

        def num_features(self):
            return 212

        def inplace_predict(self, x, iteration_range):
            assert x.shape == (4, 212)
            assert iteration_range in ((0, 300), (0, 900))
            calls.append(('predict', iteration_range))
            if change_input_during_inference and sum(kind == 'predict' for kind, _ in calls) == 18:
                data.write_bytes(b'input changed during last synthetic inference')
            return np.full((4, 24), 390., np.float32)

    monkeypatch.setattr(controller.xgb, 'Booster', FakeBooster)
    monkeypatch.setattr(controller.xgb, 'train', lambda *a, **k: pytest.fail('No training permitted'))
    out = tmp_path / 'output'
    argv = ['--data', str(data), '--out', str(out), '--device', 'cpu', '--threads', '1']
    if change_input_during_inference:
        with pytest.raises(ValueError, match='Postflight identity changed'):
            controller.main(argv)
        assert not (out / 'report.json').exists()
        assert not (out / 'complete.json').exists()
        assert len(list((out / 'contexts').glob('*.npz'))) == 18
        return
    controller.main(argv)
    assert sum(kind == 'load' for kind, _ in calls) == 9
    assert sum(kind == 'predict' for kind, _ in calls) == 18
    report = json.loads((out / 'report.json').read_text())
    assert len(report['contexts']) == 18
    assert report['postflight_unique_receipts'] > 30
    completion = json.loads((out / 'complete.json').read_text())
    assert completion['publish'] == 'not_required'
    assert completion['postflight_unique_receipts'] == report['postflight_unique_receipts']
    assert len(list((out / 'contexts').glob('*.npz'))) == 18
    assert all(not row['diagnostic']['regret']['informative'] for row in report['contexts'])
    assert config_path.read_text() == json.dumps(reconstructed)
    with np.load(out / 'contexts/F1_b0_prefix300.npz', allow_pickle=False) as arrays:
        assert arrays['left_gradient'].shape == (0, 24)
        assert arrays['gain__common_feasible'].dtype == bool
        assert arrays['diagnostic__shared_candidate'] == -1
        assert arrays['diagnostic__selected_regret_sum'] == 0


def test_sealed_model_identity_rejects_changed_bytes_before_loading(controller, tmp_path):
    model = tmp_path / 'b0.ubj'
    model.write_bytes(b'synthetic saved model')
    receipt = controller.identity(model)
    sealed = {'periods': {'F1': {'heads': [{'block': i, 'model': receipt} for i in range(3)]}}}
    assert controller.model_identity(model, sealed, 'F1', 0) == receipt
    model.write_bytes(b'changed synthetic saved model')
    with pytest.raises(ValueError, match='sealed training probe'):
        controller.model_identity(model, sealed, 'F1', 0)


def test_no_split_can_still_be_informative_conflict(controller):
    math_core = controller.module(controller.ROOT / 'output_sharing_probe_math.py', 'no_split_informative_test')
    result = math_core.sharing_diagnostics([[3., -4.], [-4., 3.]], [True, True])
    assert result['shared_candidate'] == -1
    assert result['regret']['informative']
    assert result['regret']['individual_maximum_sum'] == 6
    assert result['regret']['raw'] == 6


def test_recursive_postflight_detects_changes_and_conflicting_duplicates(controller, tmp_path):
    source = tmp_path / 'input.bin'
    source.write_bytes(b'original synthetic input')
    receipt = controller.identity(source)
    records = {'data': receipt, 'contexts': [{'model': dict(receipt)}], 'nested': ({'source': receipt},)}
    assert controller.validate_receipts(records) == 1
    conflicting = {'data': receipt, 'contexts': [{'model': dict(receipt, bytes=receipt['bytes'] + 1)}]}
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        controller.validate_receipts(conflicting)
    source.write_bytes(b'changed synthetic input contents')
    with pytest.raises(ValueError, match='Postflight identity changed'):
        controller.validate_receipts(records)
    assert records['data'] == receipt  # Stale receipts are never recomputed/replaced.
