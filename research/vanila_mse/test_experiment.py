"""Preflight invariants for the pure-MSE replacement experiment."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('vanila_mse', HERE / 'experiment.py')
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


class ExperimentTest(unittest.TestCase):
    def test_unweighted_derivatives_and_invalid_targets(self):
        y = np.array([[300., 550., 850.], [900., 400., 600.]], dtype=np.float32)
        pred = y + np.array([[10., 10., 10.], [-20., -20., -20.]], dtype=np.float32)
        gradient, hessian = model.make_objective(y)(pred, None)
        np.testing.assert_array_equal(gradient, pred - y)
        np.testing.assert_array_equal(hessian, np.ones_like(y))
        self.assertEqual(gradient.dtype, np.float32)
        self.assertEqual(hessian.dtype, np.float32)
        with self.assertRaises(ValueError):
            model.make_objective(np.array([[np.nan]]))
        with self.assertRaises(ValueError):
            model.make_objective(np.empty((0, 24)))

    def test_native_squared_error_training_parity(self):
        rng = np.random.default_rng(20261112)
        x = rng.normal(size=(192, 215)).astype(np.float32)
        y = (450 + 60 * x[:, :24] + rng.normal(size=(192, 24))).astype(np.float32)
        query = rng.normal(size=(17, 215)).astype(np.float32)
        params = dict(model.PARAMS, nthread=1, num_target=24, base_score=float(y.mean()))
        matrix = xgb.DMatrix(x, label=y, nthread=1)
        native = xgb.train(params, matrix, num_boost_round=2)
        custom = xgb.train(params, matrix, num_boost_round=2, obj=model.make_objective(y))
        np.testing.assert_array_equal(native.inplace_predict(query), custom.inplace_predict(query))
        self.assertEqual(bytes(native.save_raw('ubj')), bytes(custom.save_raw('ubj')))

    def test_data_and_offset_paths_match_parent(self):
        parent = ROOT / 'research/current_offset_roll_direct_s/experiment.py'
        def definitions(path):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
        old, new = definitions(parent), definitions(HERE / 'experiment.py')
        for name in ('timeline', 'fill_feature', 'fill_training_labels', 'load_table', 'load_grid',
                     'prior_split', 'subset', 'design', 'component_values', 'estimate_offset',
                     'apply_offset', 'prediction_frame', 'mean3', 'score'):
            with self.subTest(name=name):
                self.assertEqual(old[name], new[name])
        parent_spec = importlib.util.spec_from_file_location('direct_s_parent', parent)
        original = importlib.util.module_from_spec(parent_spec)
        parent_spec.loader.exec_module(original)
        for name in ('CONSTANT', 'VARYING', 'PARENT_CONSTANT', 'ADDED_CONSTANT', 'PARAMS',
                     'CUTOFFS', 'QUERY_COUNTS', 'SEEDS', 'ROUNDS', 'BLOCK', 'ALPHA',
                     'OFFSET_WINDOW', 'PURGE_HOURS'):
            self.assertEqual(getattr(original, name), getattr(model, name))
        self.assertEqual(len(model.CONSTANT) + len(model.VARYING), 31)
        self.assertEqual(len(model.CONSTANT) + 24 * len(model.VARYING), 215)

    def test_progress_log_writes_monitor_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            logs = model.Logs(Path(folder) / 'logs', 'test')
            logs('s42', 'fit', period='F1', horizon=24, unit=1, units=6)
            records = (Path(folder) / 'train.log').read_text().splitlines()
            self.assertEqual(len(records), 1)
            row = json.loads(records[0])
            self.assertEqual((row['period'], row['horizon'], row['unit'], row['units']),
                             ('F1', 24, 1, 6))


if __name__ == '__main__':
    unittest.main()
