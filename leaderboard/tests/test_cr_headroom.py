"""Tests for the D155 cosmic-ray headroom helpers (pure math, no data)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/d155_cr_headroom"))
import cr_lib as RL


def test_oracle_matches_demeaned_truth():
    rng = np.random.default_rng(0)
    b = rng.normal(450, 40, size=(5, 72))
    y = rng.normal(450, 60, size=(5, 72))
    o = RL.demeaned_oracle(b, y)
    resid = y - b
    m = resid.mean(axis=1, keepdims=True)
    assert np.allclose(o, y - m)


def test_gap_groups():
    gaps = np.array([3.0, 12.0, 12.5, 70.0, np.inf])
    assert list(RL.gap_group(gaps)) == ["le12", "le12", "gt12", "gt12", "none"]


def test_next_gap():
    o = np.array([0, 10, 20]) * 3_600_000_000_000
    e = np.array([5, 15]) * 3_600_000_000_000
    g = RL.next_gap_hours(o, e)
    assert np.allclose(g, [5.0, 5.0, np.inf], equal_nan=False)


def test_lead_bins_cover_72():
    h = np.arange(1, 73)
    idx = RL.lead_bin_index(h)
    assert (idx >= 0).all()
    assert [int((idx == i).sum()) for i in range(5)] == [6, 6, 12, 24, 24]


def test_bootstrap_oracle_gain_negative():
    rng = np.random.default_rng(1)
    n = 60
    # oracle strictly better by a fixed margin on every row
    sums = np.full(n, -72 * 100.0)
    cnts = np.full(n, 72.0)
    tot = np.full(n, 72.0)
    r = RL.boot_estimators(sums, cnts, tot, 500, rng)
    assert r["mean"] == r["contrib"] == -100.0
    assert r["mean_hi"] < 0 and r["contrib_hi"] < 0
