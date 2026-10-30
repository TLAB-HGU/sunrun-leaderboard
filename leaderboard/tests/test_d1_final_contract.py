"""Contract for D1-final: E0003-identical pipeline plus the official 2833x72 grid.

Fast checks only: module bindings, official-grid constants, the pure
check_official validator on synthetic frames, and source-text guards (no
forbidden strings, no foreign experiment reads, no other worker's scratch).
No fits, no registration, no network, no official reads.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LB = Path(__file__).resolve().parents[1]
PKG = LB / "experiments/d1_longitude_strips"
sys.path.insert(0, str(LB / "experiments"))
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
sys.path.insert(0, str(LB / "experiments/expos_runners"))

from d1_longitude_strips import final_official as FO  # noqa: E402
from d1_longitude_strips import runner as D1RUN  # noqa: E402
import existing_all as EA  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay.parquet", "official_check", "official-check",
             "predictions_official_truth"]


def test_final_is_the_only_new_file():
    pys = sorted(p.name for p in PKG.glob("*.py"))
    assert pys == ["final_official.py", "repro_seed.py", "runner.py", "strips.py"]


def test_pipeline_single_sourced_from_e0003():
    assert FO.add_strips is D1RUN.add_strips
    assert FO.D1 is D1RUN
    assert FO.PERIODS == ia.PERIODS == dict(D1RUN.PERIODS)
    assert D1RUN.ia.PARAMS == ia.PARAMS
    assert ia.PARAMS["random_state"] == 0  # E0003 seed, not the repro seed 1
    assert len(FO.D1.ST.strip_columns()) == 17
    text = (PKG / "final_official.py").read_text()
    assert "xgb.XGBRegressor(**D1.ia.PARAMS)" in text
    assert "REPRO_SEED" not in text and "alt_seed" not in text
    assert "72h" in text and "D1.ST.poison_test" in text and "D1.ST.strip_features" not in text
    assert "D1.base_grid" in text and "D1.write" in text


def test_official_procedure_bindings():
    assert EA.OFFICIAL_ORIGINS == ("2026-05-31 23:00", "2026-09-26 23:00")
    assert EA.CUTOFF == pd.Timestamp("2026-05-29", tz="UTC")
    assert FO.OFFICIAL_N_ORIGINS == 2833
    assert FO.OFFICIAL_HORIZONS == 72
    assert FO.OFFICIAL_ROWS == 203976 == 2833 * 72
    assert FO.OFFICIAL_COLUMNS == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert FO.FIRST_ORIGIN_UTC == "2026-05-31T23:00:00Z"
    assert FO.LAST_ORIGIN_UTC == "2026-09-26T23:00:00Z"
    origins = pd.date_range(*EA.OFFICIAL_ORIGINS, freq="h", tz="UTC")
    assert len(origins) == FO.OFFICIAL_N_ORIGINS
    assert origins[0].strftime("%Y-%m-%dT%H:%M:%SZ") == FO.FIRST_ORIGIN_UTC
    assert origins[-1].strftime("%Y-%m-%dT%H:%M:%SZ") == FO.LAST_ORIGIN_UTC


def _good_frame():
    base = datetime(2026, 5, 31, 23, 0, 0, tzinfo=timezone.utc)
    origins = [(base + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(2833)]
    return pd.DataFrame({
        "origin_last_input_utc": np.repeat(origins, 72),
        "horizon_hours": np.tile(np.arange(1, 73), 2833),
        "pred_kms": np.full(2833 * 72, 400.0),
    })


def test_check_official_accepts_full_grid():
    spec = FO.check_official(_good_frame())
    assert spec["rows"] == 203976 and spec["origins"] == 2833
    assert spec["first_origin_utc"] == FO.FIRST_ORIGIN_UTC
    assert spec["last_origin_utc"] == FO.LAST_ORIGIN_UTC


def test_check_official_fails_closed():
    good = _good_frame()
    with pytest.raises(ValueError):
        FO.check_official(good.drop(columns=["pred_kms"]))
    with pytest.raises(ValueError):
        FO.check_official(good.iloc[:-72])  # one origin short
    bad_h = good.copy()
    bad_h.loc[bad_h.index[:10], "horizon_hours"] = 0
    with pytest.raises(ValueError):
        FO.check_official(bad_h)
    bad_nan = good.copy()
    bad_nan.loc[bad_nan.index[0], "pred_kms"] = float("nan")
    with pytest.raises(ValueError):
        FO.check_official(bad_nan)
    with pytest.raises(ValueError):
        FO.check_official(pd.concat([good, good.iloc[:1]], ignore_index=True))  # dup key
    bad_o = good.copy()
    bad_o.loc[bad_o.index[0], "origin_last_input_utc"] = "2026-05-31T22:00:00Z"
    with pytest.raises(ValueError):
        FO.check_official(bad_o)


def test_no_forbidden_foreign_or_freeze_paths():
    text = (PKG / "final_official.py").read_text()
    assert not [p for p in FORBIDDEN if p in text]
    assert "E0040" not in text  # never touch the Claude submission
    assert "runs/E00" not in text and "research/expos/runs" not in text
    assert "freeze" not in text and "official-score" not in text and "official_score" not in text
    for token in ("shutil", "blend", "average", "NOAA", "GONG", "DONKI",
                  "noaa-goes18.s3", "s3.amazonaws"):
        assert token not in text, token
    assert "store/scratch/worker-d" not in text  # never another worker's scratch
    assert "worker-repro" not in text
    assert "store/scratch/worker-final" in text and "EXPOS_OUT" in text


def test_causal_outputs_present():
    text = (PKG / "final_official.py").read_text()
    assert "poison.json" in text and '"passed"' in text
    assert "availability_manifest.json" in text
    assert "predictions_official.parquet" in text
    assert "for name, (lo, hi) in PERIODS" in text
    assert "predictions_{dev,selection,audit}" in text
    # Own just-written official file is the only parquet ever read back.
    assert text.count("read_parquet") == 1
    assert "OUT / \"predictions_official.parquet\"" in text
