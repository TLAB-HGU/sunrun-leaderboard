"""D139 ablation contract (worker-t2ga): static checks only, no network, no fits."""
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d139_t2gongreal"
ABL = EXP / "ablation"
FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "baseline-replay.parquet", "official-predictions",
             "datasets--tlabtlab--sunrun-lb-store", "official_check",
             "predictions_official_truth"]
CME = ["DONKI", "CME", "LASCO", "CCOR"]
OTHER_SCRATCH = ["worker-t2gr", "worker-t2gf", "worker-t2n2", "worker-t2nr",
                 "worker-t2p", "worker-t2n/", "worker-t2n\""]
# Physics lives in the parent pipeline; ablation files must reuse it by import.
PHYSICS_DUP = ["lpmv", "upwind", "ThreadPoolExecutor", "body_sha256", "pfss_brss"]


def _code():
    return {p.name: p.read_text() for p in sorted(ABL.glob("*.py"))}


def test_ablation_file_budget():
    assert ABL.is_dir(), ABL
    pys = [p for p in ABL.rglob("*.py") if "__pycache__" not in p.parts]
    assert {p.name for p in pys} == {"rocv_alone.py", "rocv_v1.py"}, [str(p) for p in pys]
    assert len(pys) <= 4, [str(p) for p in pys]


def test_ablation_scratch_and_runner_path():
    code = _code()
    for name, t in code.items():
        assert "store/scratch/worker-t2ga" in t, name
        for s in OTHER_SCRATCH:
            assert s not in t, (name, s)
        for mod in ["import collect", "import features", "import pfss",
                    "import spec", "import rocv_runner"]:
            assert mod in t, (name, mod)
        assert "read_parquet" not in t, name
        assert "EXPOS_OUT" in t and "os.environ" in t, name


def test_ablation_no_forbidden_or_cme():
    for name, t in _code().items():
        for s in FORBIDDEN:
            assert s not in t, (name, s)
        for s in CME:
            assert s not in t, (name, s)
        assert "official-score" not in t, name
        assert "predictions_official" not in t, name
        assert "freeze" not in t.replace("fresh", ""), name


def test_ablation_no_physics_duplication():
    for name, t in _code().items():
        for s in PHYSICS_DUP:
            assert s not in t, (name, s)


def test_ablation_protocol_markers():
    code = _code()
    for name, t in code.items():
        for marker in ["f{k}_s{s}.parquet", "origin_last_input_utc",
                       "horizon_hours", "pred_kms", "poison", "cell_tables",
                       "SEEDS_ROCV", "range(len(rocv.FOLDS))", "freq=\"3h\"",
                       "72h", "fitted_choice", "None",
                       "len(maps) // 2", "poison_test", "cell_table"]:
            assert marker in t, (name, marker)
    v1 = code["rocv_v1.py"]
    assert 'EXTRA = ["v_huxt"]' in v1
    assert "RR.fit_fold" in v1 and "RR.fit_params" in v1
    alone = code["rocv_alone.py"]
    assert "no fit" in alone and "XGBRegressor" not in alone and ".fit(" not in alone
    assert "bitwise identical" in alone


def test_ablation_pins_and_bounds():
    import sys
    sys.path.insert(0, str(EXP))
    import spec as SP
    assert SP.PINS["xgboost"] == "2.1.4" and SP.PINS["python"] == "3.12.3"
    assert SP.LMAX == 30 and SP.RULE_H == 2 and SP.STALE_MAX == 0.20
    assert len(SP.SEEDS_ROCV) >= 3 and SP.FIT_NJOBS <= 6
    assert "v_huxt" in SP.FEATURES
    for name, t in _code().items():
        assert "LMAX = " not in t and "RULE_H = " not in t, name
