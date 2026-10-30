"""D133 T1-official contract: E0102 hard-switch head retrained on the official
window (train targets < 2026-05-29, 2833 hourly origins x 72h, seeds 0/1/2
with seed changing random_state only). Scratch separation (worker-t1o only),
no blending, no forbidden inputs, no vendored strips, no stored-prediction
reuse or recombination.
Fast: no data reads, no fits."""

import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d133_t1rocv" / "official"))
sys.path.insert(0, str(LB / "experiments" / "d33_hsplit"))
sys.path.insert(0, str(LB / "experiments" / "d32_bandexpert"))
sys.path.insert(0, str(LB / "experiments" / "d1_longitude_strips"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB))

import bands as D32B  # noqa: E402  (read-only band definition)
import hsplit as HS  # noqa: E402
import strips as ST  # noqa: E402
import info_ablation as ia  # noqa: E402

RUNNER = "e0102_official.py"
PKG = LB / "experiments" / "d133_t1rocv" / "official"
FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG", "CME"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]
OTHER_SCRATCH = ("worker-d133", "worker-t1b/", "worker-t1c", "worker-t1br",
                 "worker-d134", "worker-d135", "worker-d21", "worker-d28",
                 "worker-d30", "worker-d22", "worker-d23", "worker-d32",
                 "worker-d31", "worker-d33", "worker-d45", "worker-d46",
                 "worker-d48", "worker-d100")


def _text():
    return (PKG / RUNNER).read_text()


def test_owned_files_capped_and_scratch_separated():
    files = sorted(f.name for f in PKG.glob("*.py"))
    assert len(files) <= 5, files
    assert RUNNER in files
    assert _text().find("store/scratch/worker-t1o") >= 0
    for other in OTHER_SCRATCH:
        assert other not in _text(), f"cross-worker scratch {other}"


def test_no_forbidden_reads_and_no_vendored_strips():
    text = _text()
    for s in FORBIDDEN:
        assert s not in text, f"forbidden {s!r}"
    assert "noaa-goes" not in text.lower()
    for s in COPY_BANS:
        assert s not in text, f"vendors {s!r}"
    assert "official-score" not in text and "official_score" not in text
    assert "core.freeze" not in text
    assert "fitted_choice" in text


def test_no_stored_prediction_reuse_or_recombination():
    text = _text()
    assert "read_parquet" not in text or "featcache" in text
    for s in ("predictions_dev", "predictions_selection", "predictions_audit",
              "stored prediction", "stored_prediction", "blend of", "weight sweep",
              "recombin", "observations.csv"):
        assert s not in text, f"reuse/combine {s!r}"
    assert "to_parquet" in text


def test_official_protocol_cutoff_origins_naming_grid():
    text = _text()
    assert "2026-05-29" in text
    assert "target_t" in text and "CUTOFF" in text
    assert "2026-05-31 23:00" in text and "2026-09-26 23:00" in text
    assert 'freq="h"' in text or "freq='h'" in text
    assert "predictions_official_s{s}.parquet" in text
    assert "origin_last_input_utc" in text and "horizon_hours" in text and "pred_kms" in text
    assert "SEEDS = (0, 1, 2)" in text
    assert '"3h"' in text
    assert "base_grid" in text
    assert "len(origins) * 72" in text
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "d133_e0102_official", str(PKG / RUNNER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert len(mod.ORIGINS) == 2833
    assert list(mod.SEEDS) == [0, 1, 2]


def test_hard_switch_head_seed_only_no_blending():
    text = _text()
    assert "SPLIT_H" in text and "hard switch" in text.lower()
    assert "no blending" in text.lower() and "no weight" in text.lower()
    assert "blend(" not in text.lower() and "sample_weight" not in text
    assert "mean_predictions" not in text
    assert "run_seed" in text
    assert "m_s1" in text and "m_s2" in text and "m_long" in text
    assert "check_split" in text
    assert HS.check_split()["split_h"] == 24
    assert HS.BANDS == D32B.BANDS == ((1, 6), (7, 24), (25, 48), (49, 72))
    assert len(ST.strip_columns()) == 17


def test_seed_propagates_random_state_only():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "d133_e0102_official_params", str(PKG / RUNNER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ps = [mod.fit_params(s) for s in (0, 1, 2)]
    assert [p["random_state"] for p in ps] == [0, 1, 2]
    for p in ps:
        q = {k: v for k, v in p.items() if k != "random_state"}
        ref = {k: v for k, v in ps[0].items() if k != "random_state"}
        assert q == ref


def test_full_params_match_e0102_infra_only_and_gpu_budget():
    text = _text()
    assert '"device": "cuda"' in text or "'device': 'cuda'" in text or "device=cuda" in text
    assert "n_jobs" in text
    assert HS.FIT_NJOBS == 6
    assert HS.FIT_NJOBS <= 6
    p = HS.fit_params()
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
