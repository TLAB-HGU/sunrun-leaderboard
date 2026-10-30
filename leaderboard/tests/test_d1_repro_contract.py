"""Contract for D1 seed-repro: E0003-identical pipeline, seed-only change, no copy/blend."""
import difflib
import sys
from pathlib import Path

import pytest

LB = Path(__file__).resolve().parents[1]
PKG = LB / "experiments/d1_longitude_strips"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay.parquet", "official_check", "official-check",
             "predictions_official_truth"]


def test_repro_seed_differs_only_by_seed():
    base = (PKG / "runner.py").read_text().splitlines()
    repro = (PKG / "repro_seed.py").read_text().splitlines()
    assert (PKG / "repro_seed.py").exists()
    assert 'REPRO_SEED = 1' in "\n".join(repro)
    assert "random_state" in "\n".join(repro)
    # No wholesale rewrite: most lines shared with runner.py
    sm = difflib.SequenceMatcher(None, base, repro)
    assert sm.ratio() > 0.80, round(sm.ratio(), 3)
    # Seed actually differs from E0003 (ia.PARAMS random_state=0)
    sys.path.insert(0, str(LB / "experiments/headroom_audit"))
    import info_ablation as ia
    assert ia.PARAMS["random_state"] == 0
    assert 1 != ia.PARAMS["random_state"]


def test_no_forbidden_or_external_recollection():
    text = (PKG / "repro_seed.py").read_text()
    assert not [p for p in FORBIDDEN if p in text]
    for token in ("NOAA", "GONG", "DONKI", "noaa-goes18.s3", "s3.amazonaws"):
        # strips.py owns S3 HEAD manifest; the repro runner must not add its own fetch
        assert token not in text, token
    assert "2026-06" not in text and "2026-05-29" not in text and "2026-05-31" not in text


def test_no_e0003_prediction_read_copy_or_blend():
    text = (PKG / "repro_seed.py").read_text()
    # E0003 may be named as pipeline provenance; its prediction files must never be read/copied/blended
    assert "runs/E00" not in text and "research/expos/runs" not in text
    low = text.lower()
    assert "shutil" not in low and "blend" not in low and "average" not in low
    assert "read_parquet" not in low  # refit only; predictions are written, never read
    assert "from existing_all import" in text and "import strips as ST" in text
    # Fresh fit with the new-seed params on the same causal grid
    assert "xgb.XGBRegressor(**PARAMS)" in text
    assert '72h' in text and "ST.poison_test" in text and "ST.strip_features" in text


def test_causal_outputs_and_scratch_isolation():
    text = (PKG / "repro_seed.py").read_text()
    assert "poison.json" in text and '"passed"' in text
    assert "availability_manifest.json" in text
    assert "predictions_{dev,selection,audit}" in text or "for name, (lo, hi) in PERIODS" in text
    assert "worker-repro" not in text or "scratch" not in text  # runner writes only $EXPOS_OUT
    assert "store/scratch/worker-d" not in text  # never another worker's scratch
    assert "EXPOS_OUT" in text
