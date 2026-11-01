"""Regression checks on the hourly e2000fs-lag0 extension bundles (data-v2/builders/e2000fs_lag0_extend.py)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path("/home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store")
EXT, FROZEN = ROOT / "e2000fs-lag0-ext", ROOT / "e2000fs-lag0"
pytestmark = pytest.mark.skipif(not (EXT / "receipt.json").exists(), reason="extension bundles not built")

EXPECTED = {"head": ("2025-12-31T23:00:00Z", "2026-02-28T22:00:00Z", 1416),
            "gap": ("2026-05-29T00:00:00Z", "2026-05-31T22:00:00Z", 71)}


@pytest.mark.parametrize("name", EXPECTED)
def test_bundle_is_the_exact_hourly_origin_range_with_finite_features(name):
    first, last, n = EXPECTED[name]
    z = np.load(EXT / f"{name}.npz")
    o = pd.DatetimeIndex(z["origins"], tz="UTC")
    assert (o[0], o[-1], len(o)) == (pd.Timestamp(first), pd.Timestamp(last), n)
    assert (np.diff(o.asi8) == 3_600_000_000_000).all()
    names = json.loads((FROZEN / "names.json").read_text())
    assert z["C"].shape == (n, len(names["constant"])) and z["E"].shape == (n, 72, len(names["varying"]))
    assert z["past"].shape == (n, 24)
    for k in ("C", "E", "past"):
        assert np.isfinite(z[k]).all()


def test_together_with_the_frozen_queries_they_cover_every_new_period_origin_once():
    new_f1 = pd.date_range("2025-12-31T23:00Z", "2026-03-28T23:00Z", freq="h")
    new_f2 = pd.date_range("2026-03-31T23:00Z", "2026-06-27T23:00Z", freq="h")
    have = pd.DatetimeIndex(np.concatenate([np.load(EXT / "head.npz")["origins"], np.load(EXT / "gap.npz")["origins"],
                                            np.load(FROZEN / "f1.npz")["origins"], np.load(FROZEN / "official.npz")["origins"]]), tz="UTC")
    assert have.is_unique
    assert new_f1.isin(have).all() and new_f2.isin(have).all()


def test_rebuild_reproduces_every_column_of_the_frozen_hourly_bundles_on_shared_origins():
    receipt = json.loads((EXT / "receipt.json").read_text())
    checked = 0
    for seg in receipt["segments"].values():
        for c in seg["checks"]:
            if "f1.npz" in c["label"] or "official.npz" in c["label"]:
                if c["n_origins"]:
                    checked += c["n_origins"]
                    assert c["columns_with_any_mismatch"] == 0, c["label"]
    assert checked >= 300  # 121 + 216 + 1 shared origins
