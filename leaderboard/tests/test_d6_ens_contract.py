"""Contract for D6 cross-source ensemble: dev-only weights, reuse, causality.

Ensemble blends independent sources (strips + physics + recurrence, X0033),
never same-input heads. Weights come from dev-OOF only; members are reused
frozen predictions (no refit) plus a no-fit causal recurrence lookup.
"""
import sys

import numpy as np
import pandas as pd
from pathlib import Path

D6 = Path(__file__).resolve().parents[1] / "experiments" / "d6_xsource_ensemble"
sys.path.insert(0, str(D6))
import ensemble as EN  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay.parquet", "official_check", "official-check",
             "predictions_official_truth"]
# Selection/audit/official must never feed any weight or model choice.
DEV_ONLY_BANS = ("selection", "audit", "official")
WINDOW_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
                "2026-05-29", "2026-05-31", "2026-09-26")


def _synth_dev(n=500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(450, 80, n)
    return {"y": y, "strip": y + rng.normal(0, 60, n),
            "phys": y + rng.normal(0, 70, n), "rec": y + rng.normal(0, 120, n)}


def test_simplex_grid_covers_corners_and_sums_to_one():
    g = EN.simplex_grid()
    assert len(g) == 231  # (20+1)(20+2)/2
    for w in g:
        assert abs(sum(w) - 1.0) < 1e-12 and min(w) >= 0.0
    assert (1.0, 0.0, 0.0) in g and (0.0, 1.0, 0.0) in g and (0.0, 0.0, 1.0) in g


def test_fit_weights_dev_only_improves_over_members():
    dev = _synth_dev()
    w, table = EN.fit_weights(dev)
    assert abs(sum(w) - 1.0) < 1e-12 and min(w) >= 0.0
    y = dev["y"]
    mse = lambda p: float(np.mean((np.asarray(p) - y) ** 2))
    best_member = min(mse(dev[m]) for m in EN.MEMBERS)
    assert table[f"{w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f}"] <= best_member
    assert len(table) == 231


def test_fit_weights_signature_is_dev_only():
    import inspect
    assert list(inspect.signature(EN.fit_weights).parameters) == ["dev"]
    assert list(inspect.signature(EN.apply_blend).parameters) == ["strip", "phys", "rec", "w"]


def test_blend_identity_and_formula():
    rng = np.random.default_rng(1)
    s, p, r = (rng.normal(400, 50, 100) for _ in range(3))
    assert np.allclose(EN.apply_blend(s, p, r, (1.0, 0.0, 0.0)), s)
    assert np.allclose(EN.apply_blend(s, p, r, (0.5, 0.3, 0.2)), 0.5 * s + 0.3 * p + 0.2 * r)


def test_recurrence_is_causal_lookup():
    import runner as R
    idx = pd.date_range("2025-01-01", periods=24 * 60, freq="h", tz="UTC")
    filled = pd.Series(np.arange(len(idx), dtype=float) + 300.0, index=idx)
    origins = pd.DatetimeIndex([idx[40 * 24], idx[45 * 24]])
    h = np.arange(1, 73)
    got = R.recurrence_for(pd.Series(np.repeat(origins, 72)), np.tile(h, len(origins)), filled)
    for k, o in enumerate(origins):
        for hh in (1, 24, 72):
            t = o + pd.Timedelta(hours=hh)
            exp = np.mean([filled[t - pd.Timedelta(days=d)] for d in (26, 27, 28)])
            assert got[k * 72 + hh - 1] == exp
    # every lookup predates its origin: max input time < min origin
    assert (origins.min() + pd.Timedelta(hours=72) - pd.Timedelta(days=26)) < origins.min()


def test_no_forbidden_references_or_window_dates_in_new_code():
    for name in ("ensemble.py", "runner.py"):
        text = (D6 / name).read_text()
        assert not [p for p in FORBIDDEN if p in text], name
        assert not [s for s in WINDOW_DATES if s in text], name


def test_ensemble_module_is_dev_only_static():
    src = (D6 / "ensemble.py").read_text().lower()
    for token in DEV_ONLY_BANS + ("xgb", "fit(", "predict("):
        assert token not in src, f"ensemble.py must not contain {token!r}"


def test_runner_reuses_members_without_refit_and_fits_weights_on_dev():
    src = (D6 / "runner.py").read_text()
    assert "XGBRegressor" not in src and "xgb" not in src.lower().replace("xsource", "")
    assert "fit_weights(dev)" in src  # weights from the dev-OOF frame only
    assert "fit_weights(frames" not in src and "fit_weights(m" not in src.replace("fit_weights(dev)", "")
    assert "E0002" in src and "E0003" in src  # pilot/full strip members
    assert '"refit": "none' in src or "'refit': 'none" in src or "refit" in src


def test_output_contract_columns():
    import runner as R
    assert set(R.REC_DAYS) == {26, 27, 28}
    assert R.STRIP_EXP == {"pilot": "E0002", "full": "E0003"} and R.PHYS_EXP == "E0001"


def test_select_weights_guards_f1_and_minimizes_mse():
    mse = {"1.00/0.00/0.00": 100.0, "0.00/1.00/0.00": 110.0, "0.00/0.00/1.00": 200.0,
           "0.50/0.50/0.00": 90.0, "0.50/0.45/0.05": 85.0}
    f1 = {"1.00/0.00/0.00": 0.42, "0.00/1.00/0.00": 0.43, "0.00/0.00/1.00": 0.27,
          "0.50/0.50/0.00": 0.44, "0.50/0.45/0.05": 0.40}
    w, bar = EN.select_weights(mse, f1)
    assert bar == 0.43  # best member corner
    assert w == (0.5, 0.5, 0.0)  # min MSE among F1 >= bar, not the raw MSE best


def test_select_weights_never_empty_and_none_safe():
    mse = {"1.00/0.00/0.00": 100.0, "0.00/1.00/0.00": 110.0, "0.00/0.00/1.00": 200.0}
    w, bar = EN.select_weights(mse, {k: None for k in mse})
    assert bar == -1.0 and w == (1.0, 0.0, 0.0)
