"""Independent acceptance checks for the delegated experimental foundation."""

import numpy as np
import pandas as pd

from experiments.ch_breakthrough_v2 import common


def test_observation_age_and_missing_count_survive_forward_fill():
    index = pd.date_range("2024-01-01", periods=800, freq="h", tz="UTC")
    missing = np.zeros(len(index), dtype=int)
    missing[750:753] = 1
    ace = pd.DataFrame({
        "timestamp_utc": index,
        "filled_speed_kms": np.full(len(index), 400.0),
        "was_missing": missing,
    })
    ch = pd.DataFrame({"ch_t035_area": np.full(len(index), 0.01)}, index=index)

    features = common.build_features(ace, ch)

    assert features.loc[index[750], "ace_age_hours"] == 1
    assert features.loc[index[751], "ace_age_hours"] == 2
    assert features.loc[index[752], "ace_age_hours"] == 3
    assert features.loc[index[753], "ace_age_hours"] == 0
    assert features.loc[index[756], "ace_missing_72"] == 3


def test_capped_origin_sampling_preserves_utc_and_is_reproducible():
    origins = pd.date_range("2024-01-01", periods=30 * 24, freq="h", tz="UTC")

    sampled, policy = common.subsample_origins(
        origins, step="daily", max_origins=12, seed=42,
    )
    repeated, _ = common.subsample_origins(
        origins, step="daily", max_origins=12, seed=42,
    )

    assert len(sampled) == 12
    assert str(sampled.tz) == "UTC"
    assert sampled.equals(repeated)
    assert sampled.is_unique and sampled.is_monotonic_increasing
    assert (sampled.hour == 0).all()
    assert policy["kept"] == 12


def test_observed_mask_preserves_original_missingness_after_speed_loading():
    index = pd.date_range("2024-01-01", periods=8, freq="h", tz="UTC")
    ace = pd.DataFrame({
        "timestamp_utc": index,
        "filled_speed_kms": np.full(len(index), 400.0),
        "was_missing": [0, 0, 1, 1, 0, 0, 1, 0],
    })

    speed = common.as_speed_series(ace)
    observed = common.observed_mask(speed, index)

    assert observed.tolist() == [True, True, False, False, True, True, False, True]
