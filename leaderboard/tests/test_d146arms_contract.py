"""D146 arms contract: static + synthetic checks for the 3-channel comparison (fast, no fits)."""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d146_fe171_channel"
FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth",
             "research/expos/runs"]
OTHER_IDS = ["e0147-abl", "d141_e0147", "ablate", "worker-d76", "worker-t1d7",
             "worker-t1a1", "worker-t1a2", "worker-t1a3", "worker-t1a5",
             "worker-d135", "worker-d133", "worker-t2no", "nn-"]
BANNED_SRC = ["GONG", "WSA", "ADAPT", "CME", "DONKI"]
PINNED_W = {"1-6": {"e0003_line": 0.21536233464375223, "d33_hsplit": 0.3899845818930272,
                    "d45_lvlmean": 0.3946530834632206},
            "7-24": {"e0003_line": 0.3152013131693896, "d33_hsplit": 0.3436571856823209,
                     "d45_lvlmean": 0.3411415011482895},
            "25-48": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333},
            "49-72": {"e0003_line": 0.3333333333333333, "d33_hsplit": 0.3333333333333333,
                      "d45_lvlmean": 0.3333333333333333}}


def _load(name, alias=None):
    loc = importlib.util.spec_from_file_location(alias or f"d146arms_{name}", EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(loc)
    loc.loader.exec_module(mod)
    return mod


def _code():
    return "\n".join(p.read_text() for p in sorted(EXP.glob("*.py")))


def test_owned_file_budget():
    files = sorted(p.name for p in EXP.iterdir() if p.is_file())
    pys = [f for f in files if f.endswith(".py")]
    assert len(pys) <= 6, files
    assert {"spec171.py", "fe171_hourly.py", "strips171.py", "build_cache.py",
            "spec.py", "run_official.py"} <= set(pys), files


def test_spec_identity_and_scope():
    SP = _load("spec", "d146arms_spec")
    assert SP.DIRECTION == "D146" and SP.OWNER == "worker-t1d1"
    assert SP.KIND == "full" and SP.REFERENCE == "E0408"
    assert tuple(SP.SEEDS) == (0, 1, 2) and SP.SMOKE_SEED == 0
    assert SP.CUTOFF_ISO == "2026-05-29T00:00:00Z"
    assert SP.N_OFFICIAL_ORIGINS == 2833
    assert SP.TRAIN_FREQ == "3h" and SP.SPLIT_H == 24
    assert SP.SCRATCH == "store/scratch/worker-t1d1"
    assert SP.FIT_NJOBS <= 6 and SP.DEVICE == "cpu"
    assert SP.DEPTH == 3 and SP.N_TREES == 900 and abs(SP.ETA - 0.03) < 1e-12
    assert SP.BASE_PARAMS["max_depth"] == 3 and SP.BASE_PARAMS["n_estimators"] == 900
    assert abs(SP.BASE_PARAMS["learning_rate"] - 0.03) < 1e-12
    assert SP.BASE_PARAMS["device"] == "cpu" and SP.BASE_PARAMS["n_jobs"] <= 6
    assert SP.WEIGHTS == PINNED_W
    assert tuple(SP.ARMS) == ("fe195", "fe171", "both")
    assert SP.N_FE195 == 17 and SP.N_FE171 == 17 and SP.N_BOTH == 34
    assert "[axes:" in SP.AXES_TAG and "fe171-channel" in SP.AXES_TAG


def test_hypotheses_have_axes_tag_and_single_channel_claim():
    SP = _load("spec", "d146arms_spec_b")
    assert set(SP.HYPOTHESES.keys()) == {"fe195", "fe171", "both"}
    for arm, hyp in SP.HYPOTHESES.items():
        assert re.search(r"\[axes:\s*[^\]]+\]", hyp, re.I), arm
        assert "fe171-channel" in hyp, arm
        low = hyp.lower()
        assert "e0408" in low, arm
        assert "3657.8" in hyp, arm
        assert "0.0833" in hyp, arm
        assert "5188" in hyp and "4158" in hyp, arm
        assert "no corrector" in low, arm
        assert "3h" in hyp and "2026-05-29" in hyp, arm
    assert "control" in SP.HYPOTHESES["fe195"].lower()
    assert "in place of" in SP.HYPOTHESES["fe171"].lower()
    assert "34" in SP.HYPOTHESES["both"]


def test_single_axis_vs_recipe():
    SP = _load("spec", "d146arms_spec_c")
    sys.path.insert(0, str(LB / "experiments/headroom_audit"))
    import info_ablation as ia
    for k, v in ia.PARAMS.items():
        if k in ("max_depth", "n_estimators", "learning_rate", "n_jobs", "random_state"):
            continue
        assert SP.BASE_PARAMS.get(k) == v, k
    assert SP.BASE_PARAMS.get("tree_method") == "hist"
    assert SP.BASE_PARAMS.get("device") == "cpu"


def test_arm_column_contracts():
    RO = _load("run_official", "d146arms_run")
    ST = _load("strips171", "d146arms_st171")
    sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
    import strips as STF
    assert len(STF.strip_columns()) == 17
    assert len(ST.strip_columns()) == 17
    assert all(c.startswith("fe171_") for c in ST.strip_columns())
    assert ST.strip_columns()[-2:] == ["fe171_img_age_h", "fe171_img_stale"]
    assert set(STF.strip_columns()) & set(ST.strip_columns()) == set()
    fake_base = ["h", "v_lag0", "rec26", "rec_mean", "ace_speed_kms_last",
                 "ch_meridian_area", "speed_d24", "epam_p1_d6"]
    f195 = RO.arm_columns(fake_base + STF.strip_columns(), "fe195")
    f171 = RO.arm_columns(fake_base + ST.strip_columns(), "fe171")
    both = RO.arm_columns(fake_base + STF.strip_columns() + ST.strip_columns(), "both")
    assert f195[-17:] == STF.strip_columns()
    assert f171[-17:] == ST.strip_columns()
    assert both[-34:] == STF.strip_columns() + ST.strip_columns()
    assert len([c for c in both if c in STF.strip_columns()]) == 17
    assert len([c for c in both if c in ST.strip_columns()]) == 17


def test_seeds_really_vary_the_fit():
    RO = _load("run_official", "d146arms_run_b")
    SP = _load("spec", "d146arms_spec_d")
    blocks = [RO.level_seeds(s) for s in SP.SEEDS]
    assert len({r for b in blocks for r in b}) == 9, blocks
    assert [RO.event_seed(s) for s in SP.SEEDS] == [b[1] for b in blocks]
    params = [RO.fit_params(r, SP.N_TREES) for b in blocks for r in b]
    assert all(p["max_depth"] == 3 for p in params)
    assert len({p["random_state"] for p in params}) == 9
    assert all(p["n_estimators"] == 900 for p in params)
    assert all(p["device"] == "cpu" and p["n_jobs"] <= 6 for p in params)


def test_blend_convex_and_stitch_partition():
    RO = _load("run_official", "d146arms_run_c")
    RO.check_weights(_load("spec", "d146arms_spec_e").WEIGHTS)
    rng = np.random.default_rng(0)
    n = 72 * 4
    h = np.tile(np.arange(1, 73), 4)
    mem = {m: rng.normal(400, 100, n) for m in RO.MEMBERS}
    out = RO.blend_per_lead(mem, PINNED_W, h)
    assert out.shape == (n,) and np.isfinite(out).all()
    band = (h >= 7) & (h <= 24)
    w = PINNED_W["7-24"]
    assert np.allclose(out[band], sum(w[m] * mem[m][band] for m in RO.MEMBERS))
    m1, m2, m3 = h <= 6, (h >= 7) & (h <= 24), h > 24
    s1 = {s: rng.normal(size=int(m1.sum())) for s in (0, 1, 2)}
    s2 = {s: rng.normal(size=int(m2.sum())) for s in (0, 1, 2)}
    long_all = rng.normal(size=n)
    tail = rng.normal(size=int(m3.sum()))
    st = RO.stitch_members(s1, s2, long_all, tail, m1, m2, m3, (0, 1, 2), 1)
    assert np.array_equal(st["e0003_line"], long_all)
    assert np.array_equal(st["d33_hsplit"][m1], s1[1])
    assert np.array_equal(st["d45_lvlmean"][m2], sum(s2[s] for s in (0, 1, 2)) / 3)
    assert np.array_equal(st["d45_lvlmean"][m3], tail)


def test_frame_schema():
    RO = _load("run_official", "d146arms_run_d")
    te = pd.DataFrame({"origin": pd.to_datetime(["2026-06-01T00:00:00Z"] * 72, utc=True),
                       "h": np.arange(1, 73)})
    fr = RO.frame_predictions(te, np.full(72, 400.0))
    assert list(fr.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
    assert (fr["horizon_hours"].to_numpy() == np.arange(1, 73)).all()
    assert fr["origin_last_input_utc"].iloc[0] == "2026-06-01T00:00:00Z"


def test_runner_writes_seed_files_and_dual_poison_gate():
    src = (EXP / "run_official.py").read_text()
    assert "predictions_official_s{s}.parquet" in src
    assert '"poison.json"' in src and '"meta.json"' in src
    assert "refusing to present predictions as causal" in src
    assert "ST.poison_test" in src and "ST171.poison_test" in src
    assert "fit_seed_models" in src and "predict_members" in src
    assert "bitwise identical" in src
    assert "--arm" in src and '"fe195"' in src and '"fe171"' in src and '"both"' in src
    assert "worker-t1d1" in src
    assert "store/scratch/worker-t1d1" in src or "SP.SCRATCH" in src


def test_runner_applies_channel_to_train_and_grid():
    src = (EXP / "run_official.py").read_text()
    assert "attach_fe195" in src and "attach_fe171" in src and "attach_arm" in src
    assert "build_train" in src and "base_grid(origins)" in src
    assert "arm_columns" in src
    assert "ST.select_slots" in src and "ST171.select_slots" in src
    assert "ST.strip_features" in src and "ST171.strip_features" in src
    assert "ST.load_inputs" in src and "ST171.load_inputs" in src


def test_runner_touches_no_forbidden_source():
    code = _code()
    for s in FORBIDDEN:
        assert s not in code, s
    low = code.lower()
    for s in BANNED_SRC:
        assert s.lower() not in low, s
    for s in OTHER_IDS:
        assert s not in code, s
    assert "cuda" not in low
    assert "--gpu" not in code
    for f in (EXP / "spec.py", EXP / "run_official.py"):
        t = f.read_text()
        assert "core.freeze" not in t and "official-score" not in t
        for w in ("freeze", "submit", "upload"):
            assert w not in t.lower(), (f.name, w)
