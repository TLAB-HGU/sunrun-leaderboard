import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("fill_train_y", Path(__file__).with_name("fill_train_y.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    return pd.DataFrame({"set": ["train", "query", "train", "query", "train", "train"],
                         "target_time": pd.to_datetime(["2025-01-01T02Z", "2025-01-01T03Z", "2025-01-01T01Z",
                                                         "2025-01-01T02Z", "2025-01-01T03Z", "2025-01-01T00Z"]),
                         "y": [np.nan, 900., 400., np.nan, 900., np.nan]})


def test_physical_target_axis_and_query_truth_unchanged():
    frame = fixture()
    np.testing.assert_allclose(module.fill_training_labels(frame), [400., 900., 400., np.nan, 900., np.nan], equal_nan=True)


def test_future_label_does_not_fill_earlier_target():
    frame = fixture()
    before = module.fill_training_labels(frame)
    frame.loc[frame.target_time.eq(frame.target_time.max()), "y"] = 9999.
    after = module.fill_training_labels(frame)
    np.testing.assert_allclose(before[[0, 2, 3, 5]], after[[0, 2, 3, 5]], equal_nan=True)


def test_duplicate_target_disagreement_rejected():
    frame = fixture()
    frame.loc[3, "y"] = 700.
    with pytest.raises(ValueError, match="inconsistent"):
        module.fill_training_labels(frame)
