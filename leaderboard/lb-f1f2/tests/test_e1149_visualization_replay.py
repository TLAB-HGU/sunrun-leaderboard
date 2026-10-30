"""Small regression checks for replay plumbing; full reproduction is a runtime gate."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/predict_e1149_visualization.py'
spec = importlib.util.spec_from_file_location('e1149_visualization', SCRIPT)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class ReplayTests(unittest.TestCase):
    def test_checked_replacement_rejects_recipe_drift(self):
        self.assertEqual(replay.replace_once('abc', 'b', 'd'), 'adc')
        for value in ('abcabc', 'xyz'):
            with self.assertRaises(ValueError):
                replay.replace_once(value, 'abc', 'd')

    def test_prediction_shape_and_finiteness(self):
        origins = pd.date_range('2026-06-01', periods=2, freq='h', tz='UTC')
        result = replay.prediction_frame(origins, np.ones((2, 72)))
        self.assertEqual(len(result), 144)
        self.assertEqual(result.horizon_hours.tolist(), list(range(1, 73)) * 2)
        for values in (np.ones((2, 71)), np.full((2, 72), np.nan)):
            with self.assertRaises(ValueError):
                replay.prediction_frame(origins, values)

    def test_comparison_uses_keys_and_tolerance(self):
        origins = pd.date_range('2026-06-01', periods=2, freq='h', tz='UTC')
        actual = replay.prediction_frame(origins, np.ones((2, 72)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.parquet'
            actual.iloc[::-1].to_parquet(path)
            self.assertTrue(replay.compare_original(actual, path)['passed'])
            changed = actual.copy()
            changed.loc[0, 'pred_kms'] += 1e-3
            with self.assertRaises(ValueError):
                replay.compare_original(changed, path)
            duplicate = pd.concat([actual, actual.iloc[:1]])
            with self.assertRaises(ValueError):
                replay.compare_original(duplicate, path)

    def test_checkpoint_fit_restores_without_features_pickle_or_refit(self):
        class Fit:
            calls = 0
            def __init__(self, features, config, cutoff, seed, threads):
                Fit.calls += 1
                self.features, self.models = features, np.arange(4)
                self.config = config
                features.coverage, features.excluded_rows = 0.9, 4
        features = types.SimpleNamespace(unpickleable=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fit.joblib'
            original = replay.cached_fit(types.SimpleNamespace(Fit=Fit), features, {}, 'cutoff', 2, path, False)
            fresh = types.SimpleNamespace(unpickleable=lambda: None)
            restored = replay.cached_fit(types.SimpleNamespace(Fit=Fit), fresh, {}, 'cutoff', 2, path, True)
            self.assertEqual(Fit.calls, 1)
            np.testing.assert_array_equal(original.models, restored.models)
            self.assertIs(restored.features, fresh)
            self.assertEqual(fresh.excluded_rows, 4)

    def test_source_verification_detects_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source'
            path.write_text('first')
            sources = {str(path): {'sha256': replay.sha256(path)}}
            replay.verify_sources(sources)
            path.write_text('second')
            with self.assertRaises(ValueError):
                replay.verify_sources(sources)

    def test_october_preserves_unavailable_images_without_paths(self):
        times = pd.date_range('2026-09-01', '2026-10-04', freq='h', tz='UTC')
        slots = times[-2:]
        metadata = pd.DataFrame({'slot': slots, 'obs_end': slots,
                                 's3_modified': slots, 'available': [True, False]})
        raw_meta = pd.DataFrame({'slot': slots, 'nc_path': ['present.nc', None]})
        selection = pd.DataFrame({'missing': [True, True], 'slot_idx': [-1, -1], 'img_stale': [1., 1.]})
        source = types.SimpleNamespace(speed=pd.Series(400., index=times), meta=metadata,
                                       raw=pd.DataFrame(), ch=pd.DataFrame(),
                                       block=pd.DataFrame(), events=pd.DataFrame())
        labels = pd.Series([300.])
        data = types.SimpleNamespace(labels=labels)
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / 'suvi-meta/year=2026').mkdir(parents=True)
            (snapshot / 'suvi-meta/year=2026/part.parquet').touch()
            with patch('model_release.features.FeatureInputs', return_value=source), \
                    patch('model_release.strips.select_slots', return_value=selection), \
                    patch('model_release.numeric.ace_features', side_effect=lambda grid, _: pd.DataFrame(index=grid)), \
                    patch('model_release.numeric.ch_features', side_effect=lambda grid, _: pd.DataFrame(index=grid)), \
                    patch.object(pd, 'read_parquet', return_value=raw_meta):
                origins, _ = replay.switch_to_reference(data, snapshot, 2)
                self.assertEqual(len(origins), 2)
                self.assertEqual(data.meta.nc_path.tolist(), ['present.nc', ''])
                self.assertIs(data.labels, labels)
                self.assertEqual(data.ace.index[0], times[0] + 2 * replay.HOUR)
                source.meta.loc[1, 'available'] = True
                with self.assertRaisesRegex(ValueError, 'Available reference SUVI'):
                    replay.switch_to_reference(data, snapshot, 2)


if __name__ == '__main__':
    unittest.main()
