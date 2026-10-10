"""Regression checks for CPU execution of CUDA-fitted regression trees."""
import json
import pickle
import unittest

import numpy as np
import pandas as pd
import xgboost as xgb

from model_release.export_regular import cpu_models
from model_release.regular_runtime import predict_bundle, predict_estimator


class RegularRuntimeTest(unittest.TestCase):
    def test_offset_export_keeps_trees_and_survives_pickle(self):
        frame = pd.DataFrame({"a": np.linspace(-2, 2, 64, dtype=np.float32)})
        model = xgb.XGBRegressor(n_estimators=8, max_depth=2, n_jobs=1, device="cpu").fit(
            frame, 450 + 30 * frame.a.to_numpy())
        trees = model.get_booster().get_dump(dump_format="json")
        base = float(json.loads(model.get_booster().save_config())["learner"]["learner_model_param"]["base_score"])
        cpu_models(model, preserve_cuda_arithmetic=True)
        memory = predict_estimator(model, frame)
        restored = pickle.loads(pickle.dumps(model))
        self.assertEqual(trees, restored.get_booster().get_dump(dump_format="json"))
        self.assertEqual(float(restored.get_booster().attr("release_gpu_base_score")), base)
        self.assertEqual(restored.get_params()["base_score"], 0)
        self.assertEqual(memory.dtype, np.float32)
        np.testing.assert_array_equal(predict_estimator(restored, frame), memory)
        # Export is idempotent: a second pass cannot discard the original intercept.
        cpu_models(restored, preserve_cuda_arithmetic=True)
        np.testing.assert_array_equal(predict_estimator(restored, frame), memory)

    def test_plain_cpu_model_has_no_offset(self):
        frame = pd.DataFrame({"a": [0., 1., 2., 3.]})
        model = xgb.XGBRegressor(n_estimators=2, n_jobs=1, device="cpu").fit(frame, [450., 460., 430., 470.])
        np.testing.assert_array_equal(predict_estimator(model, frame), model.predict(frame))

    def test_invalid_features_and_horizons_fail_before_prediction(self):
        bundle = {"columns": ["a"], "models": None, "kind": "direct"}
        with self.assertRaisesRegex(ValueError, "Missing model features"):
            predict_bundle(bundle, pd.DataFrame({"h": [1]}))
        with self.assertRaisesRegex(ValueError, "Horizons"):
            predict_bundle(bundle, pd.DataFrame({"h": [0], "a": [1]}))


if __name__ == "__main__":
    unittest.main()
