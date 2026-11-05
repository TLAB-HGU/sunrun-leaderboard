from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import top30


def original(name):
    spec = importlib.util.spec_from_file_location('source_' + name, HERE / 'source_archive' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_recipe_preserves_source_feature_order():
    names = json.loads((HERE.parents[1] / 'analysis/A0014/outputs/cache/names.json').read_text())
    for name in top30.RECIPES:
        source = original(name)
        if hasattr(source, 'CONSTANT'):
            const, varying = source.CONSTANT, source.VARYING
        else:
            const = [n for n in names['constant'] if n in source.FEATURES]
            varying = [n for n in names['varying'] if n in source.FEATURES]
        assert [n.replace('ch_ch_', 'suvi_') for n in const] == top30.CONSTANT, name
        assert varying == top30.VARYING, name
        engine = top30.RECIPES[name][0]
        if engine.startswith('lgb'):
            params = dict(source.PARAMS); params.pop('n_jobs')
            expected = top30.LGB_PARAMS if engine == 'lgb' else top30.LGB_TUNED
            assert params == expected, name


def test_objective_complete_labels_matches_original_point_and_shape():
    rng = np.random.default_rng(1)
    y = rng.uniform(300, 800, (8, 24)).astype(np.float32)
    prediction = rng.uniform(300, 800, (8, 24)).astype(np.float32)
    matrix = xgb.DMatrix(rng.normal(size=(8, 5)), label=y)
    point = original('e2000fs_dirmo_24_feat30_pointweight4')
    actual = top30.masked_objective(y, 'point')(prediction, matrix)
    wanted = point.weighted_l2(prediction, matrix, 0)
    np.testing.assert_allclose(actual[0], wanted[0], rtol=2e-6)
    np.testing.assert_allclose(actual[1], wanted[1], rtol=2e-6)
    shape = original('e2000fs_feat30_shape')
    actual = top30.masked_objective(y, 'shape')(prediction, matrix)
    wanted = shape.shape_loss(y, prediction)
    np.testing.assert_allclose(actual[0], wanted[0], rtol=2e-6, atol=3e-5)
    np.testing.assert_allclose(actual[1], wanted[1])


@pytest.mark.parametrize('kind', ['plain', 'point', 'event', 'shape'])
def test_missing_targets_have_zero_gradient_hessian(kind):
    y = np.array([[400., np.nan, 600.], [500., 700., np.nan]])
    pred = np.full(y.shape, 450.)
    # The event rule needs a full block, as in production.
    if kind == 'event':
        y = np.tile(y, (1, 8)); pred = np.full(y.shape, 450.)
    gradient, hessian = top30.masked_objective(y, kind)(pred, None)
    assert (gradient[~np.isfinite(y)] == 0).all()
    assert (hessian[~np.isfinite(y)] == 0).all()
    assert np.isfinite(gradient).all() and np.isfinite(hessian).all()
    if kind == 'point':
        observed = np.isfinite(y)
        np.testing.assert_allclose(hessian.sum(axis=0) / observed.sum(axis=0), 1.)


def test_shape_loss_does_not_bridge_missing_target():
    y = np.array([[400., np.nan, 700.], [450., 500., 550.]])
    pred = np.array([[410., 9999., 690.], [460., 490., 580.]])
    grad, hess = top30.masked_objective(y, 'shape')(pred, None)
    np.testing.assert_array_equal(grad[0], [10., 0., -10.])
    np.testing.assert_array_equal(hess[0], [1., 0., 1.])


def test_wavelet_design_matches_original():
    rng = np.random.default_rng(12)
    c, e = rng.normal(size=(5, 21)), rng.normal(size=(5, 24, 9))
    old = original('e2000fs_feat30_wavelet')
    names = {'constant': [n.replace('suvi_', 'ch_ch_') for n in top30.CONSTANT], 'varying': top30.VARYING}
    expected = old.design({'C': c, 'E': e}, names, 0)
    np.testing.assert_array_equal(top30.design(c, e, True), expected)
    assert len(top30.feature_names(0, True)) == expected.shape[1] == 660


def test_histogram_primitives_match_original_and_ignore_missing_representatives():
    old = original('e2000fs_feat30_extreme_histprior')
    y = np.array([220., 400., 550., 580., 880., 1200.])
    np.testing.assert_array_equal(top30.soft_labels(y), old.soft_labels(y))
    np.testing.assert_array_equal(top30.representatives(np.r_[y, np.nan]), old.representatives(y))


def synthetic_table(path):
    rng = np.random.default_rng(3)
    tr = pd.date_range('2025-10-01', periods=100, freq='h', tz='UTC')
    qu = pd.date_range('2025-12-31T23:00:00Z', periods=3, freq='h')
    origins = tr.append(qu)
    frame = pd.DataFrame({'set': np.repeat(['train'] * len(tr) + ['query'] * len(qu), 72),
                          'origin': np.repeat(origins, 72), 'h': np.tile(np.arange(1, 73), len(origins)),
                          'period': np.repeat([None] * len(tr) + ['F1'] * len(qu), 72),
                          'fit_F1': np.repeat([True] * len(tr) + [False] * len(qu), 72),
                          'y': rng.uniform(350, 700, len(origins) * 72)})
    for name in top30.CONSTANT:
        frame[name] = np.repeat(rng.normal(size=len(origins)), 72).astype('float32')
    for name in top30.VARYING:
        frame[name] = rng.normal(size=len(frame)).astype('float32')
    frame.loc[frame.index % 5 == 0, 'y'] = np.nan
    frame.to_parquet(path, index=False)
    return frame


@pytest.mark.parametrize('experiment', list(top30.RECIPES))
def test_smoke_trains_saves_models_and_prediction_parts(tmp_path, experiment):
    data = tmp_path / 'data.parquet'; synthetic_table(data)
    engine = top30.RECIPES[experiment][0]
    out = tmp_path / 'out'
    report = top30.fit_block(experiment, data, out, 'F1', 0, smoke=True, lead=None if engine == 'xgb' else 1)
    assert report['smoke'] and report['status'] == 'complete'
    assert report['training_cutoff'] == '2025-12-29'
    assert report['maturity_hours'] == 144
    assert report['inference_seconds_per_fold'] > 0
    assert all(Path(p).stat().st_size > 0 for p in report['model_files'])
    z = np.load(next(out.glob('*.npz')))
    assert np.isfinite(z['pred']).all() and (z['pred'] > 0).all()
    assert z['pred'].shape == (3, 24 if engine == 'xgb' else 1)
    assert report['fits'][0]['rounds'] == 2
    with pytest.raises(ValueError, match='Smoke outputs'):
        top30.merge(out, ['F1'])


def test_query_target_values_do_not_affect_model_predictions(tmp_path):
    data = tmp_path / 'data.parquet'; frame = synthetic_table(data)
    name = 'e2000fs_calib_point4_144h'
    top30.fit_block(name, data, tmp_path / 'a', 'F1', 0, smoke=True)
    frame.loc[frame['set'] == 'query', 'y'] = 1e9
    frame.to_parquet(data, index=False)
    top30.fit_block(name, data, tmp_path / 'b', 'F1', 0, smoke=True)
    np.testing.assert_array_equal(np.load(tmp_path / 'a/F1_b0.npz')['pred'], np.load(tmp_path / 'b/F1_b0.npz')['pred'])


def test_masked_plain_matches_original_xgb_with_finite_targets():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 20)).astype('float32'); y = rng.uniform(300, 800, (100, 24)).astype('float32')
    old = xgb.XGBRegressor(n_estimators=2, **dict(top30.XGB_PARAMS, n_jobs=1))
    old.fit(x, y)
    matrix = xgb.DMatrix(x, label=y, nthread=1)
    new = xgb.train(dict(top30.XGB_PARAMS, nthread=1, base_score=float(np.mean(y))), matrix,
                    num_boost_round=2, obj=top30.masked_objective(y, 'plain'))
    np.testing.assert_allclose(new.inplace_predict(x), old.predict(x), atol=.001, rtol=1e-6)


@pytest.mark.parametrize('corruption', [None, 'identity', 'origins', 'leads'])
def test_merge_requires_matching_complete_parts(tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(top30, 'PERIODS', {'F1': 2})
    origin = pd.date_range('2025-12-31T23:00:00Z', periods=2, freq='h').asi8
    for block in range(3):
        leads = np.arange(block * 24 + 1, (block + 1) * 24 + 1)
        ns = origin.copy()
        report = {'smoke': False, 'experiment': 'e2000fs_calib_point4_144h', 'source': 'fixed.parquet',
                  'maturity_hours': 144, 'period': 'F1', 'block': block, 'leads': leads.tolist(),
                  'query_origins': 2, 'inference_seconds_per_fold': .01}
        if block == 1:
            if corruption == 'identity': report['source'] = 'other.parquet'
            if corruption == 'origins': ns += 3600 * 10**9
            if corruption == 'leads': leads += 1
        np.savez_compressed(tmp_path / f'F1_b{block}.npz', origins=ns, leads=leads, pred=np.full((2, 24), 450.))
        (tmp_path / f'F1_b{block}.json').write_text(json.dumps(report))
    if corruption:
        with pytest.raises(ValueError):
            top30.merge(tmp_path, ['F1'])
    else:
        result = top30.merge(tmp_path, ['F1'])
        assert len(result) == 144
        assert not result.duplicated(['origin_last_input_utc', 'horizon_hours']).any()
