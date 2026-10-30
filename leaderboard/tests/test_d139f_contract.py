"""D139 finalize contract (worker-t2gf): static checks only, no network, no fits."""
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d139_t2gongreal"
FIN = EXP / "finalize/rocv_gf.py"
FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "baseline-replay.parquet", "official-predictions",
             "datasets--tlabtlab--sunrun-lb-store", "official_check",
             "predictions_official_truth"]
CME = ["DONKI", "CME", "LASCO", "CCOR"]
OTHER_SCRATCH = ["worker-t2gr", "worker-t2n2", "worker-t2nr", "worker-t2p",
                 "worker-t2n/", "worker-t2n\""]


def test_finalize_file_budget():
    pys = [p for p in EXP.rglob("*.py") if "__pycache__" not in p.parts]
    base = {"spec.py", "collect.py", "pfss.py", "huxt.py", "features.py",
            "rocv_runner.py"}
    names = {p.name for p in pys}
    assert base <= names, names
    new = [p for p in pys if p.name not in base]
    assert len(new) <= 3, [str(p) for p in new]
    assert FIN.exists(), [str(p) for p in pys]


def test_finalize_scratch_and_runner_path():
    code = FIN.read_text()
    assert "store/scratch/worker-t2gf" in code, code[:500]
    for s in OTHER_SCRATCH:
        assert s not in code, s
    assert "rocv_runner" in code and "RR.main()" in code
    assert "read_parquet" not in code
    assert "EXPOS_OUT" not in code


def test_finalize_no_forbidden_or_cme():
    code = FIN.read_text()
    for s in FORBIDDEN:
        assert s not in code, s
    for s in CME:
        assert s not in code, s
    assert "official-score" not in code


def test_r3_fix_lives_in_runner():
    run = (EXP / "rocv_runner.py").read_text()
    assert "len(maps) // 2" in run
    assert "interior" in run.lower() or "Interior" in run
    assert "SEEDS_ROCV" in run and "f{k}_s{s}.parquet" in run


def test_pins_and_protocol_unchanged():
    import sys
    sys.path.insert(0, str(EXP))
    import spec as SP
    assert SP.PINS["xgboost"] == "2.1.4" and SP.PINS["python"] == "3.12.3"
    assert SP.LMAX == 30 and SP.RULE_H == 2 and SP.STALE_MAX == 0.20
    assert len(SP.SEEDS_ROCV) >= 3 and SP.FIT_NJOBS <= 6
    assert "v_huxt" in SP.FEATURES
