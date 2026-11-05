import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


E = module(HERE/'experiment.py', 'noise_runner')
C = module(ROOT/'leaderboard/scorer/competition.py', 'official_competition')
B = module(ROOT/'research/rerun_df_ffill/calib11.py', 'baseline_calib11')


def test_feature_and_sera_control_parity():
    for block in range(3):
        assert E.feature_names(block) == B.feature_names('e2000fs_calib11g_serarounds2', block)
        assert len(E.feature_names(block)) == 237
    y = np.array([[300., 500.], [400., np.nan], [700., 800.]])
    np.testing.assert_allclose(E.sera_weights(y), B.normalized_weights(B.sera_weights(y), np.isfinite(y), 'median'), rtol=1e-6)
    for key in B.PARAMS:
        if key not in ('seed',):
            assert E.PARAMS[key] == B.PARAMS[key]


def test_temporal_guards():
    train = pd.DatetimeIndex(['2025-06-23T00:00Z'])
    valid = E.validation_origins()
    E.check_split(train, valid)
    with pytest.raises(ValueError, match='maturity'):
        E.check_split(train+pd.Timedelta(hours=1), valid)
    with pytest.raises(ValueError, match='window'):
        E.check_split(train, pd.DatetimeIndex(['2025-09-29T00:00Z']))
    assert valid.min() > train.max()+pd.Timedelta(hours=72)


def test_scores_match_official_with_missing_truth():
    periods = {'F1': ('2025-07-01', '2025-07-11'), 'F2': ('2025-07-11', '2025-07-21'),
               'F3': ('2025-07-21', '2025-07-31')}
    dates = pd.date_range('2025-06-30', '2025-08-01', freq='h', tz='UTC')
    values = 450+100*np.sin(np.arange(len(dates))*.03)
    values[::17] = np.nan
    context = C.prepare_context(pd.DataFrame({'timestamp_utc': dates, 'speed_kms': values}), periods)
    truth = context['truth']
    pred = np.nan_to_num(truth, nan=430)+np.arange(72)[None, :]*.5-8
    origins = context['origins']
    frame = pd.DataFrame({'origin_last_input_utc': np.repeat(origins, 72),
                          'horizon_hours': np.tile(np.arange(1, 73), len(origins)), 'pred_kms': pred.ravel()})
    official = C.score_competition(frame, context)['total']
    actual = E.score_arrays(pred, truth)
    for a, b in [('W', 's'), ('rmse', 'rmse'), ('timse', 'timse')]:
        assert actual[a] == pytest.approx(official[b], rel=1e-12)
    assert actual['n_pairs'] == official['n_pairs']


def test_loader_uses_unfilled_truth_and_preserves_matrix(tmp_path, monkeypatch):
    train = pd.DatetimeIndex(['2025-06-22T00:00Z', '2025-06-23T00:00Z'])
    valid = E.validation_origins()[:3]
    monkeypatch.setattr(E, "validation_origins", lambda: valid)
    origins = train.append(valid)
    frame = pd.DataFrame({'origin': np.repeat(origins, 72), 'h': np.tile(np.arange(1, 73), len(origins)),
                          'set': 'train', 'y': 999.})
    for j, name in enumerate(E.CONSTANT):
        frame[name] = j+1.
    for j, name in enumerate(E.VARYING):
        frame[name] = frame.h*100+j
    source = tmp_path/'filled.parquet'
    frame.to_parquet(source)
    times = pd.date_range(E.START, E.START+pd.Timedelta(hours=100), freq='h')
    raw = pd.DataFrame({'target_time': times, 'set': 'train', 'h': 1, 'y': np.arange(len(times))+300.})
    raw.loc[4, 'y'] = np.nan
    truth = tmp_path/'raw.parquet'
    raw.to_parquet(truth)
    x, y, xq, tq, ot, ov = E.load_block(source, truth, 0)
    assert x.shape == (2, 237) and xq.shape == (3, 237)
    assert (y == 999).all()
    assert tq[0, 0] == 301 and np.isnan(tq[0, 3])
    np.testing.assert_array_equal(x[0, :21], np.arange(1, 22))
    np.testing.assert_array_equal(x[0, 21:30], np.arange(100, 109))
    E.check_split(ot, ov)


@pytest.mark.parametrize('family', ['gaussian', 'lognormal', 'mixture'])
def test_transformed_coordinate_gradient_and_units(family):
    rng = np.random.default_rng(42)
    y = np.exp(rng.normal(6.0, .15, size=(40, 2)))
    offset, scale = E.initialise(y, family)
    p = E.PARAMETER_COUNTS[family]
    q = np.zeros((1, 2*p))
    theta = E.theta_from_margin(q, offset, scale)
    target = y[:1]/500
    _, grad, _ = E.nll_and_grad(theta.reshape(-1, p), target.ravel(), family, sigma_floor=E.SIGMA_FLOOR)
    analytic = (grad.reshape(1, 2, p)*scale).ravel()
    numeric = []
    for j in range(q.size):
        delta = np.zeros_like(q)
        delta[0, j] = 1e-6
        def loss(v):
            t = E.theta_from_margin(v, offset, scale)
            return E.nll_and_grad(t.reshape(-1, p), target.ravel(), family, sigma_floor=E.SIGMA_FLOOR)[0].sum()
        numeric.append((loss(q+delta)-loss(q-delta))/2e-6)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)
    metrics = E.distribution_metrics(theta.reshape(-1, p), y[0], family)
    raw_loss = E.nll_and_grad(theta.reshape(-1, p), target.ravel(), family, sigma_floor=E.SIGMA_FLOOR)[0]
    assert metrics['nll'] == pytest.approx(raw_loss.mean()+np.log(500))
    assert sum(metrics['pit_hist']) == 2
