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
