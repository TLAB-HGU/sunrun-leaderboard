"""Portable API/input causality tests, runnable with unittest."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from model_release import load_model
from model_release.features import FeatureInputs, RAW_COLUMNS, normalize_origins


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.times = pd.date_range("2026-04-01", periods=800, freq="h", tz="UTC")
        rawdir = self.root / "raw" / "year=2026"
        rawdir.mkdir(parents=True)
        raw = pd.DataFrame({"timestamp_utc": self.times})
        for i, name in enumerate(RAW_COLUMNS):
            raw[name] = np.arange(len(raw), dtype=float) / 100 + i + 1
        raw.to_parquet(rawdir / "part.parquet")
        pd.DataFrame({"timestamp_utc": self.times, "filled_speed_kms": 400 + np.sin(np.arange(800))}).to_parquet(self.root / "speed.parquet")
        pd.DataFrame({"ch_t035_area": np.arange(800, dtype=float)}, index=self.times).to_parquet(self.root / "ch.parquet")
        pd.DataFrame({f"lon{i:02d}_lat1_{suffix}": np.ones(800) for i in range(12)
                      for suffix in ("log_median", "log_q10", "dark_fraction", "valid_fraction")}, index=self.times).to_parquet(self.root / "suvi.parquet")
        pd.DataFrame({"slot": self.times, "obs_end": self.times, "s3_modified": self.times,
                      "status": "ok", "available": True}).to_parquet(self.root / "manifest.parquet")
        self.inputs = dict(raw_ace_dir=self.root / "raw", speed_parquet=self.root / "speed.parquet",
                           ch_parquet=self.root / "ch.parquet", suvi_parquet=self.root / "suvi.parquet",
                           suvi_manifest=self.root / "manifest.parquet")
        self.origin = self.times[750]

    def test_future_poison_and_positive_control(self):
        before = FeatureInputs(self.inputs).build([self.origin, self.times[775]])
        for key in ("speed_parquet", "ch_parquet", "suvi_parquet"):
            frame = pd.read_parquet(self.inputs[key])
            times = frame["timestamp_utc"] if "timestamp_utc" in frame else frame.index
            mask = times > self.origin
            columns = frame.select_dtypes("number").columns
            frame.loc[mask, columns] += 100
            frame.to_parquet(self.inputs[key])
        rawpath = self.inputs["raw_ace_dir"] / "year=2026" / "part.parquet"
        raw = pd.read_parquet(rawpath)
        raw.loc[raw.timestamp_utc > self.origin, list(RAW_COLUMNS)] += 100
        raw.to_parquet(rawpath)
        after = FeatureInputs(self.inputs).build([self.origin, self.times[775]])
        pd.testing.assert_frame_equal(before.iloc[:72], after.iloc[:72], check_exact=True)
        self.assertFalse(before.iloc[72:].equals(after.iloc[72:]))

    def test_missing_inputs_and_metadata_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Missing inputs"):
            FeatureInputs({})
        meta = pd.read_parquet(self.inputs["suvi_manifest"]).drop(columns="available")
        meta.to_parquet(self.inputs["suvi_manifest"])
        with self.assertRaisesRegex(ValueError, "available"):
            FeatureInputs(self.inputs)

    def test_suvi_requires_exact_source_statistics(self):
        original = pd.read_parquet(self.inputs["suvi_parquet"])
        invalid = (original.drop(columns="lon00_lat1_log_q10"),
                   original.rename(columns={"lon00_lat1_log_median": "lon12_lat1_log_median"}))
        for frame in invalid:
            with self.subTest(columns=tuple(frame.columns)):
                frame.to_parquet(self.inputs["suvi_parquet"])
                with self.assertRaisesRegex(ValueError, "exactly 48"):
                    FeatureInputs(self.inputs)

    def test_missing_images_produce_stale_features(self):
        meta = pd.read_parquet(self.inputs["suvi_manifest"])
        meta["available"] = False
        meta.to_parquet(self.inputs["suvi_manifest"])
        result = FeatureInputs(self.inputs).build([self.origin])
        self.assertTrue(result.img_stale.eq(1).all())
        self.assertTrue(result.strip_v350_logmed.isna().all())

    def test_origins_validation(self):
        for origins in ([], [self.origin, self.origin], ["2026-05-01 12:30Z"]):
            with self.assertRaises(ValueError):
                normalize_origins(origins)

    def test_enlil_required_and_future_completion_ignored(self):
        with self.assertRaisesRegex(ValueError, "donki_dir"):
            FeatureInputs(self.inputs, include_enlil=True)
        directory = self.root / "donki"
        directory.mkdir()
        event = {"modelCompletionTime": (self.origin + pd.Timedelta("1h")).isoformat(),
                 "estimatedShockArrivalTime": (self.origin + pd.Timedelta("24h")).isoformat(), "kp_90": 8}
        (directory / "WSAEnlilSimulations_test.json").write_text(json.dumps([event]))
        source = FeatureInputs(self.inputs | {"donki_dir": directory}, include_enlil=True)
        result = source.build([self.origin, self.origin + pd.Timedelta("2h")])
        self.assertTrue(result.iloc[:72].enlil_next_arr_h.isna().all())
        self.assertTrue(result.iloc[72:].enlil_next_arr_h.eq(22).all())

    def test_training_cutoff_enforced(self):
        from model_release.api import ReleasedModel
        model = ReleasedModel(dict(format_version=1, experiment="test", kind="direct", models=None,
                                   state={}, columns=[], config={"hyperparameters": {"train_target_cutoff_utc": "2026-05-29T00:00:00Z"}}))
        for section in ("hyperparameters", "data"):
            model.bundle["config"] = {section: {"train_target_cutoff_utc": "2026-05-29T00:00:00Z"}}
            with self.assertRaisesRegex(ValueError, "training cutoff"):
                model.predict([self.origin], self.inputs)

    def test_pickle_api_and_feature_order(self):
        bundle = dict(format_version=1, experiment="test", kind="direct", models=None,
                      state={}, columns=["v_lag0", "h"])
        path = self.root / "test.pkl"
        path.write_bytes(pickle.dumps(bundle))
        def prediction(loaded, frame):
            self.assertEqual(loaded["columns"], ["v_lag0", "h"])
            return frame[loaded["columns"]].iloc[:, 0].to_numpy()
        with patch("model_release.regular_runtime.predict_bundle", side_effect=prediction):
            result = load_model(path).predict([self.origin], self.inputs)
        self.assertEqual(list(result.columns), ["origin_last_input_utc", "horizon_hours", "pred_kms"])
        self.assertEqual(result.horizon_hours.tolist(), list(range(1, 73)))
        self.assertTrue(np.isfinite(result.pred_kms).all())


if __name__ == "__main__":
    unittest.main()
