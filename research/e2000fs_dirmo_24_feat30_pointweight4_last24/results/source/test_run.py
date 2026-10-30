import importlib.util
from pathlib import Path
import numpy as np
import xgboost as xgb

spec = importlib.util.spec_from_file_location('dirmo_point4', Path(__file__).with_name('run.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def test_point_weights_do_not_spill_across_outputs():
    y = np.array([[550., 400.], [400., 550.], [400., 400.]])
    w = r.weights(y)
    np.testing.assert_allclose(w.mean(axis=0), 1.)
    assert w[0, 0] / w[0, 1] == 4
    assert w[1, 1] / w[1, 0] == 4
    np.testing.assert_array_equal(w[2], [.5, .5])


def test_objective_is_weighted_l2_derivative():
    y = np.array([[550., 400.], [400., 550.]], dtype=np.float32)
    pred = y + np.array([[2., -3.], [4., -5.]])
    matrix = xgb.DMatrix(np.ones((2, 3)), label=y)
    grad, hess = r.objective(pred, matrix)
    w = r.weights(y)
    eps = 1e-3
    numerical = (0.5*w*(pred+eps-y)**2 - 0.5*w*(pred-eps-y)**2)/(2*eps)
    np.testing.assert_allclose(grad, numerical, rtol=1e-6)
    np.testing.assert_array_equal(hess, w)
    assert grad.shape == hess.shape == (2, 2)


def test_true_vector_output_tree_and_reload(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 4)).astype(np.float32)
    y = np.tile(np.linspace(400, 650, 24), (64, 1)).astype(np.float32) + x[:, :1] * 50
    model = xgb.train(dict(tree_method='hist', multi_strategy='multi_output_tree', num_target=24,
                           max_depth=2, nthread=2, base_score=500),
                      xgb.DMatrix(x, label=y), num_boost_round=2, obj=r.objective)
    path = tmp_path / 'vector.json'
    model.save_model(path)
    doc = r.read(path)
    trees = doc['learner']['gradient_booster']['model']['trees']
    assert len(trees) == 2
    assert all(int(t['tree_param']['size_leaf_vector']) == 24 for t in trees)
    np.testing.assert_array_equal(model.inplace_predict(x), xgb.Booster(model_file=path).inplace_predict(x))


def test_first_two_blocks_are_ordinary_l2_even_for_high_targets():
    y = np.array([[400., 550., 800.], [600., 300., 1000.]], dtype=np.float32)
    pred = y + 10
    matrix = xgb.DMatrix(np.ones((2, 3)), label=y)
    for block in (0, 1):
        grad, hess = r.objective(pred, matrix, block=block)
        np.testing.assert_array_equal(grad, pred-y)
        np.testing.assert_array_equal(hess, np.ones_like(y))
        np.testing.assert_array_equal(r.block_weights(y, block, np.array([2., 3., 4.])), np.ones_like(y))


def test_last_block_matches_frozen_parent_objective():
    parent_path = Path(__file__).resolve().parents[1] / 'e2000fs_dirmo_24_feat30_pointweight4/run.py'
    parent_spec = importlib.util.spec_from_file_location('allblock_parent', parent_path)
    parent = importlib.util.module_from_spec(parent_spec)
    parent_spec.loader.exec_module(parent)
    y = np.array([[549., 550., 700.], [600., 300., 650.]], dtype=np.float32)
    pred = y + 20
    matrix = xgb.DMatrix(np.ones((2, 4)), label=y)
    actual = r.objective(pred, matrix, block=2)
    expected = parent.objective(pred, matrix)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    normalizer = np.array([1.5, 2., 3.])
    np.testing.assert_array_equal(r.block_weights(y, 2, normalizer), parent.weights(y, normalizer))


def test_unweighted_vector_model_matches_native_squared_error():
    from functools import partial
    rng = np.random.default_rng(42)
    x = rng.normal(size=(96, 4)).astype(np.float32)
    y = np.tile(np.linspace(350, 750, 24), (96, 1)).astype(np.float32) + x[:, :1]*80
    matrix = xgb.DMatrix(x, label=y)
    params = dict(tree_method='hist', multi_strategy='multi_output_tree', num_target=24,
                  max_depth=2, nthread=2, base_score=float(y.mean()), objective='reg:squarederror')
    native = xgb.train(params, matrix, num_boost_round=3)
    for block in (0, 1):
        custom = xgb.train(params, matrix, num_boost_round=3, obj=partial(r.objective, block=block))
        np.testing.assert_array_equal(native.inplace_predict(x), custom.inplace_predict(x))
