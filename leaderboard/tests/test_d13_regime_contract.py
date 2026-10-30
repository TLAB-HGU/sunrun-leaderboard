"""D13 regime-router contract: synthetic CPU-only checks, no data reads.

Covers the assignment acceptance surface without running the pipeline:
regime purity/causality on synthetic frames, dev-OOF-only selection shape,
file budget, forbidden-source hygiene, cuda device, scratch isolation and
the expos output contract keys.
"""
import json
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT))

from d13_regime import regime as RG  # noqa: E402

D13 = Path(__file__).resolve().parents[1] / "experiments" / "d13_regime"
RUNNER_SRC = (D13 / "runner.py").read_text()
REGIME_SRC = (D13 / "regime.py").read_text()

FORBIDDEN_TOKENS = ["truth.parquet", "folds.parquet", "naive.parquet",
                    "official-predictions", "baseline-replay", "official_check",
                    "official-score", "scorer", "donki", "DONKI", "noaa",
                    "NOAA", "gong", "GONG", "ENLIL", "enlil",
                    "huggingface", "snapshots"]


def _frame(rows):
    return pd.DataFrame(rows, columns=["origin", RG.V0_COL, RG.LAG24_COL])


def test_file_budget_max_five():
    files = [p for p in D13.iterdir() if p.is_file() and not p.name.startswith("__")]
    assert len(files) <= 5, [p.name for p in files]
    assert (D13 / "regime.py").exists() and (D13 / "runner.py").exists()


def test_assign_shock_hss_quiet():
    df = _frame([("2025-01-01", 600.0, 400.0),   # rise 200 -> shock (both cands)
                 ("2025-01-02", 560.0, 540.0),   # rise 20, v0 high -> hss
                 ("2025-01-03", 400.0, 410.0)])  # quiet
    for cand in RG.CANDIDATES:
        got = list(RG.assign(df, cand))
        assert got == ["shock", "hss", "quiet"], (cand, got)


def test_assign_nan_safe_quiet_and_horizon_blind():
    df = _frame([("2025-01-01", float("nan"), 400.0),
                 ("2025-01-02", 600.0, float("nan"))])
    df["h"] = [1, 72]
    df["y"] = [999.0, 999.0]
    for cand in RG.CANDIDATES:
        # NaN v0 -> quiet; NaN lag24 -> rise unknown but v0=600 still hss
        assert list(RG.assign(df, cand)) == ["quiet", "hss"]
    # horizon/target columns cannot change the assignment
    df2 = df.copy()
    df2["h"] = [72, 1]
    df2["y"] = [1.0, 1.0]
    for cand in RG.CANDIDATES:
        assert list(RG.assign(df2, cand)) == list(RG.assign(df, cand))


def test_assign_uses_origin_row_only():
    # shuffling row order must not change any row's own label
    df = _frame([(f"2025-01-0{d}", v, v - r)
                 for d, (v, r) in enumerate(
                     [(600, 200), (560, 20), (400, -10), (700, 150)], start=1)])
    for cand in RG.CANDIDATES:
        fwd = list(RG.assign(df, cand))
        rev = list(RG.assign(df.iloc[::-1], cand))[::-1]
        assert fwd == rev


def test_pick_winner_argmin_first_registered_tiebreak():
    assert RG.pick_winner({"A_rise80_v480": 2.0, "B_rise120_v550": 1.0}) == "B_rise120_v550"
    assert RG.pick_winner({"A_rise80_v480": 1.0, "B_rise120_v550": 1.0}) == "A_rise80_v480"


def test_no_forbidden_sources_in_d13_code():
    for name, src in (("regime.py", REGIME_SRC), ("runner.py", RUNNER_SRC)):
        for tok in FORBIDDEN_TOKENS:
            assert tok not in src, (name, tok)


def test_cuda_device_and_no_selection_fed_choice():
    assert '"device": "cuda"' in RUNNER_SRC or "'device': 'cuda'" in RUNNER_SRC
    assert "device" in RUNNER_SRC
    # router pick path references only the dev window
    assert 'PERIODS["dev"]' in RUNNER_SRC
    assert "oof_select" in RUNNER_SRC
    # no hard-coded foreign experiment ids or official-window writes
    assert "predictions_official" not in RUNNER_SRC
    assert "freeze" not in RUNNER_SRC


def test_scratch_isolation_and_owner():
    assert "store/scratch/worker-d13" in RUNNER_SRC
    assert "worker-d16" not in RUNNER_SRC and "worker-d8" not in RUNNER_SRC


def test_output_contract_keys_present():
    for key in ["predictions_{dev,selection,audit}", "poison.json",
                "manifest.json", "importance.json", "learning_curves.json",
                '"passed"']:
        assert key in RUNNER_SRC, key
    # predictions use the reference writer (origin_last_input_utc, horizon_hours, pred_kms)
    assert "from existing_all import" in RUNNER_SRC and "write(te," in RUNNER_SRC


def test_experience_ids_cited_in_manifest_or_hypothesis():
    assert "X0044" in RUNNER_SRC and "X0008" in RUNNER_SRC
