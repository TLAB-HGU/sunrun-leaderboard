"""D140 NOAA-official contract: static checks only (fast, no network, no fits)."""
import importlib.util
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d140_t2noaareal"
OFF = EXP / "official"
FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "baseline-replay.parquet", "official-predictions",
             "datasets--tlabtlab--sunrun-lb-store", "official_check",
             "predictions_official_truth", "research/expos/runs"]
CME = ["DONKI", "LASCO", "CCOR"]
OTHER_WORKERS = ["worker-t2n2", "worker-t2gr", "worker-t2ga", "worker-t2gf",
                 "worker-t2gd", "worker-t2p", "worker-t2g", "worker-t2b",
                 "worker-t3g", "worker-d133", "worker-d135"]


def _code():
    return "\n".join((OFF / f).read_text() for f in sorted(OFF.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in OFF.iterdir() if p.is_file())
    assert len(files) <= 6, files
    assert {"spec.py", "collect.py", "features.py", "runner.py"} <= set(files), files


def _spec():
    """Load official spec without touching sys.path or the spec module cache."""
    loc = importlib.util.spec_from_file_location("d140_official_spec", OFF / "spec.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def test_spec_owner_scratch_seeds_rule():
    SP = _spec()
    assert SP.DIRECTION == "D140" and SP.OWNER == "worker-t2no"
    assert str(SP.SCRATCH) == "store/scratch/worker-t2no", SP.SCRATCH
    assert str(SP.READONLY_SCRATCH) == "store/scratch/worker-t2nr", SP.READONLY_SCRATCH
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.RULE_H == 12 and SP.STALE_MAX == 0.20
    assert SP.FEATURES == ["v_enlil", "enlil_age_h", "enlil_stale"]
    assert SP.FIT_NJOBS == 4 and SP.MAX_CONNECTIONS == 4
    assert SP.PINS["xgboost"] == "2.1.4"
    assert SP.COLLECT_START == "2022-10-01" and SP.COLLECT_END == "2026-09-25"
    assert "lb-eval" in SP.SELECT_RULE and "E0147" in SP.SELECT_RULE
    assert "gpu0" in SP.GPU_REASON and "gpu1" in SP.GPU_REASON
    assert "CPU" in SP.GPU_REASON and "60s" in SP.GPU_REASON
    assert "n_jobs<=20" in SP.GPU_REASON
    code = _code()
    for w in OTHER_WORKERS:
        assert w not in code, w


def test_no_forbidden_or_cme_or_gong():
    code = _code()
    for s in FORBIDDEN:
        assert s not in code, s
    for s in CME:
        assert s not in code, s
    assert "GONG" not in code
    for f in sorted(OFF.glob("*.py")):
        t = (OFF / f).read_text()
        assert "core.freeze" not in t and "official-score" not in t
        assert "freeze" not in t.lower()
        assert "submit" not in t.lower()
        assert "upload" not in t.lower()
        assert "push" not in t.lower()


def test_network_and_tar_only_in_collect():
    for f in sorted(OFF.glob("*.py")):
        t = (OFF / f).read_text()
        if f.name == "collect.py":
            continue
        assert "urlopen" not in t, f.name
        assert "tarfile" not in t, f.name
        assert "Request(" not in t, f.name
    col = (OFF / "collect.py").read_text()
    assert "urlopen" in col and "tarfile" in col and "suball.nc" in col
    assert "Earth_V1" in col and "Earth_Density" in col
    assert "unlink" in col
    assert "header_sha256" in col and "body_sha256" in col
    assert "parquet_sha256" in col


def test_official_template_markers():
    run = (OFF / "runner.py").read_text()
    for marker in ["target_t", "CUTOFF", "official_s", "origin_last_input_utc",
                   "horizon_hours", "pred_kms", "poison", "random_state",
                   "raw_retained", "no stored-prediction", "fitted_choice",
                   "SEEDS", "2833"]:
        assert marker in run, marker
    assert '"passed"' in run
    feat = (OFF / "features.py").read_text()
    for marker in ["RULE_H", "FALLBACK_V", "poison_test", "stale_fraction"]:
        assert marker in feat, marker
    spec_src = (OFF / "spec.py").read_text()
    assert "2026-05-31" in spec_src and "2026-09-26" in spec_src


def test_seed_variation_and_artifact_keys():
    run = (OFF / "runner.py").read_text()
    assert "for s in seeds" in run
    assert "fit_params(s)" in run or "fit_params(seed" in run
    assert "random_state" in run and "int(seed)" in run
    assert "n_estimators" in run
    assert "predictions_official_s" not in (OFF / "collect.py").read_text()
    assert "predictions_official_s" not in (OFF / "features.py").read_text()
    assert "predictions_official_s" not in (OFF / "spec.py").read_text()


def test_scratch_writes_only_to_worker_t2no():
    col = (OFF / "collect.py").read_text()
    assert "store/scratch/worker-t2no" in col or "SP.SCRATCH" in col
    assert "READONLY" in col  # read-only source is named, never written
    assert "copy2" in col  # seed copy reads readonly, writes own scratch
    run = (OFF / "runner.py").read_text()
    assert "store/scratch/worker-t2nr" not in run
    feat = (OFF / "features.py").read_text()
    assert "NC.EARTH" in feat and "NC.READONLY_EARTH" in feat
    spec_src = (OFF / "spec.py").read_text()
    assert "store/scratch/worker-t2no" in spec_src
