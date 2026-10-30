"""D140 NOAA-real contract: static checks only (fast, no network, no fits)."""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d140_t2noaareal"
FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "baseline-replay.parquet", "official-predictions",
             "datasets--tlabtlab--sunrun-lb-store", "official_check",
             "predictions_official_truth"]
CME = ["DONKI", "LASCO", "CCOR"]


def _code():
    return "\n".join((EXP / f).read_text() for f in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.glob("*") if p.is_file())
    assert len(files) <= 8, files
    assert {"spec.py", "noaa_collect.py", "noaa_features.py",
            "rocv_runner.py"} <= set(files), files


def test_scratch_isolation_and_pins():
    sys.path.insert(0, str(EXP))
    import spec as SP
    assert SP.DIRECTION == "D140" and SP.OWNER == "worker-t2nr"
    assert str(SP.SCRATCH) == "store/scratch/worker-t2nr", SP.SCRATCH
    assert SP.RULE_H == 12 and SP.STALE_MAX == 0.20
    assert len(SP.SEEDS_ROCV) >= 3
    assert SP.FEATURES == ["v_enlil", "enlil_age_h", "enlil_stale"]
    assert SP.FIT_NJOBS <= 6 and SP.MAX_CONNECTIONS <= 4
    assert SP.PINS["xgboost"] == "2.1.4"
    code = _code()
    assert "worker-t2n2" not in code and "worker-t2gr" not in code
    assert "worker-t2p" not in code and "worker-t2g" not in code
    assert "store/scratch/worker-t2nr" in code or "SP.SCRATCH" in code


def test_no_forbidden_or_cme_sources():
    code = _code()
    for s in FORBIDDEN:
        assert s not in code, s
    for s in CME:
        assert s not in code, s
    for f in sorted(EXP.glob("*.py")):
        t = (EXP / f).read_text()
        assert "official-score" not in t and "predictions_official" not in t
        assert "core.freeze" not in t and "core.register(" not in t
    assert "CME catalog" not in code or "NOT opened" in code


def test_network_and_tar_only_in_collect():
    for f in sorted(EXP.glob("*.py")):
        t = (EXP / f).read_text()
        if f.name == "noaa_collect.py":
            continue
        assert "urlopen" not in t, f.name
        assert "tarfile" not in t, f.name
        assert "Request(" not in t, f.name


def test_template_and_protocol_markers():
    run = (EXP / "rocv_runner.py").read_text()
    for marker in ["fold_start - 72h", "f{k}_s{s}.parquet", "origin_last_input_utc",
                   "horizon_hours", "pred_kms", "poison", "learning_curves",
                   "cell_tables", "enlil_ablation", "INVALID", "random_state",
                   "raw_retained", "no stored-prediction"]:
        assert marker in run, marker
    col = (EXP / "noaa_collect.py").read_text()
    assert "ThreadPoolExecutor" in col and "body_sha256" in col
    assert "header_sha256" in col and "suball.nc" in col
    assert "Earth_V1" in col and "Earth_Density" in col
    assert "unlink" in col  # raw tar deleted immediately
    feat = (EXP / "noaa_features.py").read_text()
    for marker in ["RULE_H", "FALLBACK_V", "poison_test", "stale_fraction"]:
        assert marker in feat, marker


def test_seed_and_artifact_keys():
    run = (EXP / "rocv_runner.py").read_text()
    assert "SEEDS_ROCV" in run and "SMOKE" in run
    assert '"passed"' in run
    assert "fitted_choice" in run and "None" in run
