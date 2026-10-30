"""D133 T1b-redo rocv contract: E0105 head structure with genuine per-cell seed
propagation (level random_state=s, event random_state=100+s) under the rocv
protocol. Scratch separation (worker-t1br only), no blending, no forbidden
inputs, no vendored strips, no stored-prediction reuse.
Fast: no data reads, no fits."""

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d133_t1rocv"))
sys.path.insert(0, str(LB / "experiments" / "d33_hsplit"))
sys.path.insert(0, str(LB / "experiments" / "d32_bandexpert"))
sys.path.insert(0, str(LB / "experiments" / "d45_lvlmean"))
sys.path.insert(0, str(LB / "experiments" / "d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import bands as D32B  # noqa: E402  (read-only band definition)
import hsplit as HS  # noqa: E402
import lvlmean as LM  # noqa: E402
import strips as ST  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import rocv  # noqa: E402

RUNNER = "e0105r_rocv.py"
FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]
OTHER_SCRATCH = ("worker-d133", "worker-t1b/", "worker-t1c", "worker-d134", "worker-d135",
                 "worker-d21", "worker-d28", "worker-d30", "worker-d22", "worker-d23",
                 "worker-d32", "worker-d31", "worker-d33", "worker-d45", "worker-d46",
                 "worker-d48")


def _text():
    return (LB / "experiments" / "d133_t1rocv" / RUNNER).read_text()


def test_owned_files_capped_and_scratch_separated():
    pkg = LB / "experiments" / "d133_t1rocv"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 6, files
    assert RUNNER in files
    assert _text().find("store/scratch/worker-t1br") >= 0
    for other in OTHER_SCRATCH:
        assert other not in _text(), f"cross-worker scratch {other}"


def test_no_forbidden_reads_and_no_vendored_strips():
    text = _text()
    for s in FORBIDDEN:
        assert s not in text, f"forbidden {s!r}"
    assert "noaa-goes" not in text.lower()
    for s in COPY_BANS:
        assert s not in text, f"vendors {s!r}"
    assert "predictions_official" not in text
    assert "official-score" not in text and "official_score" not in text
    assert "fitted_choice" in text


def test_no_stored_prediction_reuse():
    text = _text()
    assert "read_parquet" not in text or "featcache" in text
    for s in ("predictions_dev", "predictions_selection", "predictions_audit",
              "stored prediction", "stored_prediction", "blend of", "weight sweep"):
        assert s not in text, f"reuse/combine {s!r}"


def test_rocv_protocol_full_grid_purge_naming():
    text = _text()
    assert "len(rocv.FOLDS)" in text
    assert "fold_bounds" in text
    assert 'pd.Timedelta("72h")' in text
    assert "target_t" in text
    assert "f{k}_s{s}.parquet" in text
    assert "origin_last_input_utc" in text and "horizon_hours" in text and "pred_kms" in text
    assert "SEEDS = (0, 1, 2)" in text
    assert '"3h"' in text
    assert "base_grid" in text
    assert len(rocv.FOLDS) == 8


def test_redo_head_level_s_event_100ps_no_blending_per_cell_fit():
    text = _text()
    assert "SPLIT_H" in text and "hard switch" in text.lower()
    assert "no blending" in text.lower() and "no weight" in text.lower()
    assert "blend(" not in text.lower() and "sample_weight" not in text
    assert "mean_predictions" not in text, "redo has no intra-cell seed mean"
    assert "run_fold_seed" in text
    assert "def level_seed" in text and "def event_seed" in text
    assert "100 + int(s)" in text or "100+int(s)" in text or "OFFSET + int(s)" in text
    assert "level_seed(s)" in text and "event_seed(s)" in text
    assert "identical-by-construction" not in text
    assert "byte-identical" not in text
    assert "SE=0" not in text and "seed SE=0" not in text
    assert "m_long" in text
    assert "check_split" in text
    assert HS.check_split()["split_h"] == 24
    assert LM.check_split()["split_h"] == 24
    assert HS.BANDS == D32B.BANDS == ((1, 6), (7, 24), (25, 48), (49, 72))
    assert len(ST.strip_columns()) == 17


def test_redo_seed_functions_propagate():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "d133_e0105r_rocv", str(LB / "experiments" / "d133_t1rocv" / RUNNER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert [mod.level_seed(s) for s in (0, 1, 2)] == [0, 1, 2]
    assert [mod.event_seed(s) for s in (0, 1, 2)] == [100, 101, 102]
    assert len({mod.event_seed(s) for s in (0, 1, 2)} | {mod.level_seed(s) for s in (0, 1, 2)}) == 6


def test_full_params_match_e0105_infra_only_and_gpu_budget():
    text = _text()
    assert '"device": "cuda"' in text or "'device': 'cuda'" in text or "device=cuda" in text
    assert "n_jobs" in text
    assert LM.FIT_NJOBS == 4
    assert LM.FIT_NJOBS <= 4
    p = LM.fit_params()
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert "EXPOS_SMOKE" in text


def test_real_poison_and_fixed_rule_record():
    text = _text()
    assert "poison.json" in text and '"passed"' in text
    assert "ST.poison_test" in text
    assert "SystemExit" in text
    assert "selection.json" in text
    assert "None" in text
