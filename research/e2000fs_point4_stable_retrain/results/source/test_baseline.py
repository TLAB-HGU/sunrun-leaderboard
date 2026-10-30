import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('seasonal_baseline_test', Path(__file__).with_name('baseline.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def test_weight_threshold_and_normalization():
    weights = b.point_weights(np.array([549.999, 550., 700.]))
    assert weights.mean() == 1.
    assert weights[1] / weights[0] == 4.
    assert weights[2] == weights[1]


def test_strict_target_maturity():
    cutoff = pd.Timestamp('2025-03-04', tz='UTC')
    origins = pd.DatetimeIndex([cutoff-pd.Timedelta(hours=h) for h in (75, 72, 69)])
    np.testing.assert_array_equal(b.mature_indices(origins, cutoff), [0])


def test_model_roundtrip_and_tamper(tmp_path):
    x = np.arange(120, dtype=float).reshape(40, 3)
    y = np.linspace(400., 650., 40)
    model = b.d.controls_fit(x, y, x[:8], y[:8],
                            dict(n_estimators=3, n_jobs=2, min_child_samples=3, verbosity=-1),
                            b.point_weights(y), tmp_path)
    context = dict(stage='dev0', horizon=1, manifest_sha256='fixed', train_indices_sha256='fixed')
    b.d.write(tmp_path/'receipt.json', dict(context=context,
              sha256={name:b.d.digest(tmp_path/name) for name in ('model.txt', 'loss.csv')}))
    restored = b.checkpoint(tmp_path, context)
    np.testing.assert_array_equal(model.predict(x, num_threads=2), restored.predict(x, num_threads=2))
    with pytest.raises(ValueError, match='identity'):
        b.checkpoint(tmp_path, dict(context, horizon=2))
    with (tmp_path/'model.txt').open('a') as stream:
        stream.write('\nchanged\n')
    with pytest.raises(ValueError, match='Frozen file changed'):
        b.checkpoint(tmp_path, context)
