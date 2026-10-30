"""D148 Step 4 HUXt-no-GONG contract: static + synthetic checks (fast, no HUXt run)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d148_cme_timing" / "huxt"
FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "baseline-replay.parquet", "official-predictions",
             "from scorer", "import scorer", "from expos", "import expos",
             "official-score", "official_score", "freeze", "submit",
             "predictions_official_s", "D139", "E0229", "E0230",
             "GONG", "PFSS", "WSA", "CDAW", "isMostAccurate", "--gpu"]
# NOTE: E0471/E0472/e0147/D141/nn-*/sun-nn are contact-banned, numeric
# citation only (assignment). SELECT_RULE cites E0472 3281.31 as a number;
# contact would be a path/cmd/id reference, covered by OTHER_IDS + the
# experiment-path check below.
OTHER_IDS = ["worker-t1e0", "worker-t1e1", "worker-t1e2", "worker-t1e3",
             "e0147-abl"]


def _load(name):
    if str(EXP) not in sys.path:
        sys.path.insert(0, str(EXP))
    loc = importlib.util.spec_from_file_location(f"d148huxt_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec_huxt.py", "corotation.py", "run_pilot.py"} <= set(pys), files
    assert (EXP / "venv_receipt.json").exists(), files


def test_spec_identity_and_scope():
    SP = _load("spec_huxt")
    assert SP.DIRECTION == "D148" and SP.OWNER == "worker-t1e4"
    assert SP.KIND == "pilot" and SP.REFERENCE == "E0408"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h"
    assert SP.SCRATCH == "store/scratch/worker-t1e4"
    assert int(SP.FIT_NJOBS) <= 6
    assert SP.VENV == ".venv-huxt" and SP.HUXT_PIN == "huxt==5.0.3"
    assert abs(float(SP.ROT_H) - 27.2753 * 24) < 1e-9
    assert int(SP.N_PILOT) == 30 and int(SP.PILOT_STEP) == 94
    assert int(SP.SMOKE_N) == 2
    assert SP.AXES_TAG.startswith("[axes:") and "huxt" in SP.AXES_TAG
    assert "E0472" in SP.SELECT_RULE and "3281.31" in SP.SELECT_RULE
    assert "0.0833" in SP.SELECT_RULE


def test_venv_receipt_pinned():
    rec = json.loads((EXP / "venv_receipt.json").read_text())
    assert rec["huxt"] == "huxt==5.0.3" and rec["venv"] == ".venv-huxt"
    assert rec["n_packages"] >= 100 and len(rec["freeze_sha256"]) == 64


def test_corotation_causal_and_missing_fallback():
    CR = _load("corotation")
    idx = pd.date_range("2026-05-01", periods=24 * 60, freq="h", tz="UTC")
    ace = pd.DataFrame({"v": 400.0 + np.arange(len(idx)) * 0.01,
                        "miss": np.zeros(len(idx), dtype=int)}, index=idx)
    o = pd.Timestamp("2026-06-10T00:00:00Z", tz="UTC")
    v0, used, fb, ok = CR.boundary_for_origin(o, ace)
    want = (o - pd.Timedelta(hours=654.6072)).floor("h")
    assert used == want and fb == 0.0 and ok is True
    assert abs(v0 - float(ace.loc[want, "v"])) < 1e-12
    # exact hour missing -> causal walk-back only, never forward
    ace2 = ace.copy()
    ace2.loc[want, "miss"] = 1
    v0b, usedb, fbb, _ = CR.boundary_for_origin(o, ace2)
    assert usedb == want - pd.Timedelta(hours=1) and fbb == 1.0
    assert abs(v0b - float(ace.loc[usedb, "v"])) < 1e-12
    # all missing in window -> refuse
    ace3 = ace.copy()
    ace3["miss"] = 1
    try:
        CR.boundary_for_origin(o, ace3)
    except ValueError:
        pass
    else:
        raise AssertionError("all-missing window must raise, not fill")
    # every consulted timestamp is at or before the origin
    for oo in (o, o + pd.Timedelta(hours=5)):
        _, u2, _, _ = CR.boundary_for_origin(oo, ace)
        assert u2 <= oo


def test_no_forbidden_paths_or_sources():
    code = _code()
    low = code.lower()
    for s in FORBIDDEN:
        assert s.lower() not in low, s
    for oid in OTHER_IDS:
        assert oid not in code, oid
    assert "store/ch-v1/ace.parquet" in code, "only ACE speed series may be read"
    assert "store/donki-cme" not in code, "pilot is ACE-only; no DONKI cone yet"


def test_pilot_frame_schema():
    CR = _load("corotation")
    idx = pd.date_range("2026-05-01", periods=24 * 60, freq="h", tz="UTC")
    ace = pd.DataFrame({"v": 400.0, "miss": np.zeros(len(idx), dtype=int)}, index=idx)
    te = pd.DataFrame({"origin": [idx[1000]] * 72, "h": np.arange(1, 73)})
    assert (te["h"].to_numpy() == np.arange(1, 73)).all()
    v0, used, _, _ = CR.boundary_for_origin(idx[1000], ace)
    assert used <= idx[1000] and v0 == 400.0
