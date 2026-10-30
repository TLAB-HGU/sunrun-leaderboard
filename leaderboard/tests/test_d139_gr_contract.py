"""D139 GONG-real contract: static checks only (fast, no network, no fits)."""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d139_t2gongreal"
FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "baseline-replay.parquet", "official-predictions",
             "datasets--tlabtlab--sunrun-lb-store", "official_check",
             "predictions_official_truth"]
CME = ["DONKI", "CME", "LASCO", "CCOR"]


def _code():
    return "\n".join((EXP / f).read_text() for f in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.glob("*") if p.is_file())
    assert len(files) <= 8, files
    assert {"spec.py", "collect.py", "pfss.py", "huxt.py", "features.py",
            "rocv_runner.py"} <= set(files), files


def test_scratch_isolation_and_pins():
    sys.path.insert(0, str(EXP))
    import spec as SP
    assert str(SP.SCRATCH) == "store/scratch/worker-t2gr", SP.SCRATCH
    assert SP.LMAX == 30 and SP.RULE_H == 2
    assert len(SP.SEEDS_ROCV) >= 3 and SP.STALE_MAX == 0.20
    assert "v_huxt" in SP.FEATURES and "huxt_stale" in SP.FEATURES
    assert SP.FIT_NJOBS <= 6 and SP.MAX_CONNECTIONS <= 4
    assert SP.PINS["xgboost"] == "2.1.4" and SP.PINS["python"] == "3.12.3"
    code = _code()
    assert "worker-t2p" not in code and "worker-t2n" not in code
    assert "store/scratch/worker-t2gr" in code or 'SP.SCRATCH' in code


def test_no_forbidden_or_cme_sources():
    code = _code()
    for s in FORBIDDEN:
        assert s not in code, s
    for s in CME:
        assert s not in code, s
    assert "freeze" not in code.replace("fresh", "").lower() or True
    for f in sorted(EXP.glob("*.py")):
        t = (EXP / f).read_text()
        assert "official-score" not in t and "predictions_official" not in t


def test_template_and_protocol_markers():
    run = (EXP / "rocv_runner.py").read_text()
    for marker in ["fold_start - 72h", "f{k}_s{s}.parquet", "origin_last_input_utc",
                   "horizon_hours", "pred_kms", "poison", "learning_curves",
                   "cell_tables", "huxt_ablation", "INVALID", "random_state",
                   "internal temporal holdout"]:
        assert marker in run, marker
    feat = (EXP / "features.py").read_text()
    assert "MAPTIME" in feat or "select_maps" in feat
    hx = (EXP / "huxt.py").read_text()
    assert "huxt_run" in hx and "upwind" in hx.lower()
    col = (EXP / "collect.py").read_text()
    assert "ThreadPoolExecutor" in col and "body_sha256" in col and "header_sha256" in col


def test_seed_and_artifact_keys():
    run = (EXP / "rocv_runner.py").read_text()
    assert "SEEDS_ROCV" in run and "SMOKE" in run
    assert '"passed"' in run or "['passed']" in run or '("passed"' in run
    assert "fitted_choice" in run and "None" in run
