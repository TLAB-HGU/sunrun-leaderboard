import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('calibration', HERE / 'calibration.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def original_function(filename, name, context):
    tree = ast.parse((HERE / 'source_archive' / filename).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), filename, 'exec'), context)  # noqa: S102 — isolated pure pinned function for parity
    return context[name]


def test_tweedie_finite_parity_and_missing_mask():
    y = np.array([[300., 600.], [500., 700.]])
    pred = np.array([[1., 400.], [3200., 200.]], dtype=np.float32)
    matrix = xgb.DMatrix(np.ones((2, 2)), label=y)
    original = original_function('e2000fs_calib_tweedie15.py', 'tweedie_objective', {'np': np, 'SCALE': 500.})
    for actual, expected in zip(c.objective_for(y, 'tweedie')(pred, matrix), original(pred, matrix, 1.5)):
        np.testing.assert_allclose(actual, expected)
    y[0, 1] = np.nan
    grad, hess = c.objective_for(y, 'tweedie')(pred, matrix)
    assert grad[0, 1] == hess[0, 1] == 0


def test_point4_finite_parity_and_missing_normalizer():
    y = np.array([[300., 600.], [500., 700.]])
    original = original_function('e2000fs_calib_tweedie15.py', 'point_weights', {'np': np, 'HPO': 550.})
    g, h = c.objective_for(y)(y+2, None)
    np.testing.assert_allclose(h, original(y))
    np.testing.assert_allclose(g, 2*original(y))
    y[0, 1] = np.nan
    g, h = c.objective_for(y)(np.zeros_like(y), None)
    assert h[0, 1] == g[0, 1] == 0
    assert h[1, 1] == 1


def test_era_unnormalized_point4():
    y = np.array([[300., 600.], [500., np.nan]])
    _, h = c.objective_for(y, row_weights=np.array([.5, 1.5]))(np.zeros_like(y), None)
    np.testing.assert_array_equal(h, [[.5, 2.], [1.5, 0.]])


@pytest.mark.parametrize('experiment', c.EXPERIMENTS)
def test_small_training_all_recipes(tmp_path, experiment):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(96, 8)).astype(np.float32)
    x[::7, 2] = np.nan
    xq = x[:4].copy()
    y = rng.uniform(250, 800, size=(96, 24))
    y[::5, ::3] = np.nan
    y[0] = np.nan
    origins = pd.date_range('2024-01-01', periods=96, freq='7D', tz='UTC')
    pred, _, report = c.fit_recipe(experiment, x, xq, y, origins, 0, 2, 1, tmp_path, 'smoke')
    assert pred.shape == (4, 24)
    assert np.isfinite(pred).all()
    assert ((pred >= 1) & (pred <= 3000)).all()
    assert report['dropped_no_target'] == 1
    assert list(tmp_path.glob('*.ubj'))
    assert list(tmp_path.glob('*_loss.json'))


def test_unsupported_recipe_fails(tmp_path):
    with pytest.raises(ValueError, match='Unsupported'):
        c.fit_recipe('pool', np.ones((3, 2)), np.ones((1, 2)), np.ones((3, 24)),
                     pd.date_range('2024-01-01', periods=3), 0, 2, 1, tmp_path, 'bad')


def test_query_target_poison_does_not_change_training(tmp_path, monkeypatch):
    rng = np.random.default_rng(19)
    origins = pd.date_range('2024-01-01', periods=48, freq='D', tz='UTC')
    con = rng.normal(size=(48, len(c.CONSTANT))).astype(np.float32)
    ext = rng.normal(size=(48, 72, len(c.VARYING))).astype(np.float32)
    y = rng.uniform(250, 800, size=(48, 72))
    data = tmp_path / 'input'; data.write_text('fixed input identity')
    predictions = []
    for poison in (float('nan'), 1e10):
        monkeypatch.setattr(c, 'load', lambda *_, poison=poison: ((origins, con, ext, y),
                            (origins[:4], con[:4], ext[:4], np.full((4, 72), poison))))
        out = tmp_path / str(poison); out.mkdir()
        c.train_period(data, 'e2000fs_calib2_detroute', 'F1', [0], 2, 1, out)
        with np.load(out / 'F1_b0.npz') as z:
            predictions.append(z['pred'])
    np.testing.assert_array_equal(*predictions)
