"""D133 T1 rocv contract: E0102 recipe under rocv protocol, scratch separation,
no blending, no forbidden inputs, no vendored strips. Fast: no data reads, no fits."""

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d133_t1rocv"))
sys.path.insert(0, str(LB / "experiments" / "d33_hsplit"))
sys.path.insert(0, str(LB / "experiments" / "d32_bandexpert"))
sys.path.insert(0, str(LB / "experiments" / "d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import bands as D32B  # noqa: E402  (read-only band definition)
import hsplit as HS  # noqa: E402
import strips as ST  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import rocv  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_owned_files_capped_and_scratch_separated():
    pkg = LB / "experiments" / "d133_t1rocv"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 6, files
    assert "rocv_runner.py" in files
    runner = (pkg / "rocv_runner.py").read_text()
    assert "store/scratch/worker-d133" in runner
    for other in ("worker-d33", "worker-d134", "worker-d135", "worker-d21", "worker-d28",
                  "worker-d30", "worker-d22", "worker-d23", "worker-d32", "worker-d31"):
        assert other not in runner


def test_no_forbidden_reads_and_no_vendored_strips():
    pkg = LB / "experiments" / "d133_t1rocv"
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"
        assert "predictions_official" not in text
        assert "official-score" not in text and "official_score" not in text


def test_rocv_protocol_full_grid_purge_naming():
    runner = (LB / "experiments" / "d133_t1rocv" / "rocv_runner.py").read_text()
    assert "len(rocv.FOLDS)" in runner or "range(len(rocv.FOLDS))" in runner
    assert "fold_bounds" in runner
    assert 'pd.Timedelta("72h")' in runner or "Timedelta('72h')" in runner
    assert "target_t" in runner
    assert 'f{k}_s{s}.parquet' in runner
    assert "origin_last_input_utc" in runner and "horizon_hours" in runner and "pred_kms" in runner
    assert "SEEDS = (0, 1, 2)" in runner
    assert '"3h"' in runner or "'3h'" in runner
    assert "base_grid" in runner
    assert "rocv" in runner.lower()
    assert len(rocv.FOLDS) == 8


def test_e0102_head_hard_switch_no_blending_seed_only():
    runner = (LB / "experiments" / "d133_t1rocv" / "rocv_runner.py").read_text()
    assert "SPLIT_H" in runner and "hard switch" in runner.lower()
    assert "no blending" in runner.lower() and "no weight" in runner.lower()
    assert "blend(" not in runner.lower() and "sample_weight" not in runner
    assert "m_long" in runner and "m_s1" in runner and "m_s2" in runner
    assert "random_state" in runner
    assert "check_split" in runner
    assert HS.check_split()["split_h"] == 24
    assert HS.BANDS == D32B.BANDS == ((1, 6), (7, 24), (25, 48), (49, 72))
    assert len(ST.strip_columns()) == 17


def test_full_params_match_e0102_infra_only_and_gpu_budget():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "d133_rocv_runner", str(LB / "experiments" / "d133_t1rocv" / "rocv_runner.py"))
    assert spec is not None
    runner = (LB / "experiments" / "d133_t1rocv" / "rocv_runner.py").read_text()
    assert '"device": "cuda"' in runner or "'device': 'cuda'" in runner or "device=cuda" in runner
    assert "n_jobs" in runner
    assert HS.FIT_NJOBS == 6
    assert HS.FIT_NJOBS <= 6
    p = HS.fit_params()
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert "EXPOS_SMOKE" in runner


def test_real_poison_and_fixed_rule_record():
    runner = (LB / "experiments" / "d133_t1rocv" / "rocv_runner.py").read_text()
    assert "poison.json" in runner and '"passed"' in runner
    assert "ST.poison_test" in runner
    assert "SystemExit" in runner
    assert "selection.json" in runner
    assert "fitted_choice" in runner and "None" in runner
