import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def experiment():
    spec = importlib.util.spec_from_file_location('roundselect_test', Path(__file__).with_name('experiment.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('seed', [42, 43, 44])
def test_frozen_parent_contract_and_private_import(experiment, seed):
    previous = sys.modules.get('baseline')
    baseline, core = experiment.parent_modules()
    config = experiment.fixed_config(core, seed, f'roundselect_s{seed}')
    names = experiment.assert_parent_contract(core, config, seed)
    assert core.baseline is baseline
    assert sys.modules.get('baseline') is previous
    assert len(names['constant']) == 20
    assert len(names['varying']) == 8
    assert 'img_age_h' in names['constant']
    assert len(core.expanded_names(names, 0, 24)) == 212
    assert 'reuse_run' not in config


@pytest.mark.parametrize('period,count', [('F1', 1706), ('F2', 1682), ('F3', 1730)])
def test_temporal_subsets_keep_original_offset_and_full_rows(experiment, period, count):
    baseline, _ = experiment.parent_modules()
    # Official hourly fit grid ends exactly 144 hours before the original cutoff.
    end = pd.Timestamp(baseline.CUTOFFS[period], tz='UTC') - pd.Timedelta(hours=144)
    origins = pd.date_range(end=end, periods=10000, freq='h')
    reference, calibration, stopping, offset, boundary = experiment.fitting_subsets(baseline, origins, period)
    assert len(stopping) == count
    np.testing.assert_array_equal(offset, calibration[-336:])
    assert len(offset) == 336
    assert origins[reference].max() + pd.Timedelta(hours=144) <= boundary
    assert origins[stopping].max() + pd.Timedelta(hours=144) <= origins[offset].min()
    assert not np.intersect1d(stopping, offset).size
    with pytest.raises(ValueError, match='stopping count'):
        experiment.fitting_subsets(baseline, origins.delete(stopping[0]), period)


def test_first_minimum_and_incomplete_curve_rejection(experiment):
    assert experiment.first_minimum([8., 2., 2., 4.], 4) == 2
    assert experiment.first_minimum([3., 2., 1.], 3) == 3
    for curve in ([1.], [1., float('nan')], [1., float('inf')]):
        with pytest.raises(ValueError, match='Incomplete or nonfinite'):
            experiment.first_minimum(curve, 2)


def test_frozen_control_checks_raw_predictions_origins_and_targets(experiment, tmp_path):
    raw = np.arange(48.).reshape(2, 24)
    truth = raw + 1
    origins = np.array([1, 2])
    path = tmp_path / 'prior_predictions.npz'
    np.savez(path, raw=raw, training_truth=truth, origins=origins)
    assert experiment.check_control(raw + 1e-7, truth, origins, path)['passed']
    for changed_raw, changed_truth, changed_origins, message in [
        (raw + 1e-4, truth, origins, 'mismatch'),
        (raw, truth + 1, origins, 'targets'),
        (raw, truth, origins + 1, 'origins'),
    ]:
        with pytest.raises(ValueError, match=message):
            experiment.check_control(changed_raw, changed_truth, changed_origins, path)


def test_cpu_spectral_24target_history_does_not_change_fit_and_prefix_replays(experiment, tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    _, core = experiment.parent_modules()
    config = experiment.fixed_config(core, 42, 'roundselect_s42')
    rng = np.random.default_rng(42)
    x = rng.normal(size=(96, 8)).astype(np.float32)
    y = (400 + 80 * x[:, :1] + rng.normal(size=(96, 24)) * 15).astype(np.float64)
    xs = rng.normal(size=(32, 8)).astype(np.float32)
    ys = (400 + 80 * xs[:, :1] + rng.normal(size=(32, 24)) * 15).astype(np.float64)
    budget = 9
    plain, empty, _ = experiment.train_head(core, x, y, config, 1, budget)
    evaluated, curve, _ = experiment.train_head(core, x, y, config, 1, budget, stopping=(xs, ys))
    assert empty == []
    assert len(curve) == budget
    assert evaluated.num_boosted_rounds() == budget
    np.testing.assert_array_equal(plain.inplace_predict(xs), evaluated.inplace_predict(xs))
    selected = experiment.first_minimum(curve, budget)
    assert selected == np.flatnonzero(np.asarray(curve) == min(curve))[0] + 1
    # Native curves use float32 DMatrix labels; verify every prefix directly.
    for rounds in range(1, budget + 1):
        pred = evaluated.inplace_predict(xs, iteration_range=(0, rounds))
        rmse = np.sqrt(np.mean((pred.astype(np.float64) - ys.astype(np.float32)) ** 2))
        assert curve[rounds - 1] == pytest.approx(rmse, abs=1e-5)
    path = tmp_path / 'prior.ubj'
    evaluated.save_model(path)
    replay = experiment.xgb.Booster(params={'device': 'cpu', 'nthread': 1})
    replay.load_model(path)
    assert replay.num_boosted_rounds() == budget
    np.testing.assert_array_equal(evaluated.inplace_predict(xs, iteration_range=(0, selected)),
                                  replay.inplace_predict(xs, iteration_range=(0, selected)))
    # An explicit interior prefix catches accidental default-endpoint inference.
    short, _, _ = experiment.train_head(core, x, y, config, 1, 3)
    np.testing.assert_array_equal(evaluated.inplace_predict(xs, iteration_range=(0, 3)),
                                  short.inplace_predict(xs))
    full, _, _ = experiment.train_head(core, x, y, config, 1, selected)
    full.save_model(tmp_path / 'full.ubj')
    replay.load_model(tmp_path / 'full.ubj')
    assert replay.num_boosted_rounds() == selected
    np.testing.assert_array_equal(full.inplace_predict(xs), replay.inplace_predict(xs))


@pytest.mark.parametrize('control_matches', [True, False])
def test_period_control_precedes_selection_and_full_fit_keeps_all_rows(
        experiment, tmp_path, monkeypatch, control_matches):
    baseline, core = experiment.parent_modules()
    config = experiment.fixed_config(core, 42, 'roundselect_s42')
    end = pd.Timestamp(baseline.CUTOFFS['F1'], tz='UTC') - pd.Timedelta(hours=144)
    origins = pd.date_range(end=end, periods=10000, freq='h')
    train = {'origins': origins, 'C': np.zeros((len(origins), 20), np.float32),
             'E': np.zeros((len(origins), 72, 8), np.float32)}
    query = baseline.subset(train, np.arange(3))
    y = np.full((len(origins), 72), 400.)
    monkeypatch.setattr(baseline, 'load_grid', lambda data, period, names, training:
                        (train, y) if training else (query, None))
    monkeypatch.setitem(baseline.QUERY_COUNTS, 'F1', 3)
    _, calibration, _stopping, _, _ = experiment.fitting_subsets(baseline, origins, 'F1')
    parent = tmp_path / 'parent'
    frozen = parent / 'runs/drop035_v650_s42/F1_models/prior_predictions.npz'
    frozen.parent.mkdir(parents=True)
    np.savez(frozen, raw=np.full((len(calibration), 72), 400. if control_matches else 401.),
             training_truth=y[calibration], origins=origins[calibration].asi8)
    monkeypatch.setattr(experiment, 'PARENT', parent)
    fits, prefixes = [], []

    class Model:
        def __init__(self, rounds=900, **kwargs):
            self.rounds = rounds

        def save_model(self, path):
            Path(path).write_text(str(self.rounds))

        def load_model(self, path):
            self.rounds = int(Path(path).read_text())

        def num_boosted_rounds(self):
            return self.rounds

        def inplace_predict(self, x, iteration_range=None):
            if iteration_range is not None:
                assert len(fits) >= 3
                prefixes.append(iteration_range)
            value = 390. if iteration_range else (400. if self.rounds == 900 else 380.)
            return np.full((len(x), 24), value)

    def fake_train(core, x, target, config, threads, rounds, stopping=None):
        fits.append((len(x), rounds, stopping))
        if stopping is not None:
            assert len(stopping[0]) == 1706
        else:
            assert len(x) == len(y)
        curve = [3.] * 900
        curve[6] = curve[7] = 1.
        return Model(rounds), curve, {}

    monkeypatch.setattr(experiment, 'train_head', fake_train)
    monkeypatch.setattr(experiment.xgb, 'Booster', Model)
    out = tmp_path / 'out'
    out.mkdir()
    if not control_matches:
        with pytest.raises(ValueError, match='Control prior900 mismatch'):
            experiment.run_period(core, config, 'unused', out, 'F1', 4, {})
        assert len(fits) == 3
        assert prefixes == []
        assert not (out / 'F1_predictions.npz').exists()
        return
    report = experiment.run_period(core, config, 'unused', out, 'F1', 4, {})
    assert [item[1] for item in fits] == [900, 900, 900, 7, 7, 7]
    assert prefixes == [(0, 7)] * 3
    assert report['selected_rounds'] == [7] * 3
    assert report['frozen_state']['offset_rows'] == 336
    np.testing.assert_array_equal(report['frozen_state']['offset'], np.full(72, 2.5))
    with np.load(out / 'F1_predictions.npz') as result:
        np.testing.assert_array_equal(result['pred'], np.full((3, 72), 382.5))
