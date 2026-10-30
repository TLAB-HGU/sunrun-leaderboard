"""Predicted-onset rule of expos/arrival.py and the label builder's onset rule on synthetic ramps."""
import numpy as np
import pandas as pd

from experiments.arrival_labels.build_labels import first_sustained
from expos.arrival import predicted_onset


def ramp(onset_h, base=380.0, peak=600.0, width=6, n=72):
    h = np.arange(1, n + 1, dtype=float)
    return base + (peak - base) * np.clip((h - onset_h) / width, 0, 1)


def test_predicted_onset_matches_shift():
    for onset in (10, 24, 40):
        got = predicted_onset(ramp(onset), 220.0)
        # 20% of a linear 6h ramp is crossed ~1.2h after its start
        assert got is not None and abs(got - (onset + 2)) <= 1, (onset, got)


def test_flat_or_damped_trajectory_is_a_miss():
    assert predicted_onset(np.full(72, 420.0), 200.0) is None
    assert predicted_onset(ramp(20, peak=410.0), 220.0) is None   # 30 km/s rise < max(50, 0.25*220)


def test_label_onset_needs_sustained_crossing():
    idx = pd.date_range("2025-01-01", periods=12, freq="h", tz="UTC")
    v = pd.Series([380, 390, 430, 410, 430, 455, 490, 550, 600, 610, 615, 620], index=idx, dtype=float)
    t = first_sustained(v, 424.0, idx[0], 3)
    assert t == idx[4]
