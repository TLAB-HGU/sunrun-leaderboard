"""D84 bootstrap-CI contract: dual tag, preregistered resampling, CPU-only, report-only.

Synthetic + static checks only (no GPU, no fits, no network).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LB = Path(__file__).resolve().parents[1]
PKG = LB / "experiments" / "d84_bootci"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import bootci as BC  # noqa: E402
import bootspec as SP  # noqa: E402
import info_ablation as ia  # noqa: E402
from expos import core, evaluate  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay",
             "official_check", "official-check", "predictions_official_truth"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W", "def unique_event_metrics",
             "def unique_event_report", "def stack_predictions", "def hsplit",
             "def lvlmean"]
ALLOWED_IDS = {"E0102", "E0140", "E0144", "E0001", "E0003"}


def _blobs():
    return {f.name: (PKG / f.name).read_text() for f in
            [PKG / "bootspec.py", PKG / "bootci.py", PKG / "runner.py"]}


def test_dual_tag_and_experience_citation():
    assert SP.FAMILY_TAG == "[family=xgb]"
    assert SP.AXES_TAG == "[axes: val=bootstrap-ci]"
    assert SP.HYPOTHESIS.startswith("[family=xgb] [axes:")
    assert SP.HYPOTHESIS.index("[family=") < SP.HYPOTHESIS.index("[axes:")
    for eid in ("E0102", "E0140", "E0144", "E0001"):
        assert eid in SP.HYPOTHESIS
    assert SP.check_spec()["experience"] == ["X0096", "X0126", "X0131"]
    assert "X0134" in SP.HYPOTHESIS  # density caveat cited


def test_resampling_preregistered_and_periods_match_evaluator():
    assert (SP.BOOT_B, SP.BOOT_SEED, SP.BOOT_BLOCK_DAYS) == (2000, 7, 7)
    assert SP.CI_LEVEL == 0.95 and SP.JACK_BLOCK_WEEKS == 4
    assert dict(SP.PERIODS) == dict(evaluate.PERIODS)
    assert tuple(SP.CANDIDATES) == ("E0102", "E0140", "E0144")
    assert tuple(SP.REFERENCES) == ("E0001", "E0003")
    assert "D83" in SP.CPU_REASON and "D86" in SP.CPU_REASON


def test_fit_params_single_axis_val_only_cpu():
    p = SP.fit_params()
    assert p["device"] == "cpu" and p["n_jobs"] == 6
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k in ("n_jobs", "random_state"):
            continue
        assert p[k] == v
    assert SP.PERIOD_JOBS * SP.FIT_NJOBS <= 20


def test_rules_preregistered_report_only():
    assert "report-only" in SP.SELECT_RULE and "never executed" in SP.SELECT_RULE
    assert "72h purge" in SP.SELECT_RULE and "future-poison" in SP.SELECT_RULE
    assert "full 3h" in SP.FULL_RULE and "excludes 0" in SP.FULL_RULE
    assert "no freeze" in SP.FULL_RULE.lower() and "official-score" in SP.FULL_RULE
    assert "screening" in SP.FULL_RULE


def _synthetic_frame(n_origins=84, step="6h", seed=0):
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2025-08-04", periods=n_origins, freq=step, tz="UTC")
    n = len(origins)
    df = pd.DataFrame({"origin": np.repeat(origins, 72), "h": np.tile(np.arange(1, 73), n)})
    t = df["origin"] + pd.to_timedelta(df["h"], unit="h")
    y = 400 + 50 * np.sin(np.arange(len(df)) / 24) + rng.normal(0, 20, len(df))
    filled = pd.Series(y, index=pd.DatetimeIndex(t)).sort_index()
    filled = filled[~filled.index.duplicated(keep="last")]
    observed = pd.Series(True, index=filled.index)
    cand = df.assign(y_pred=y + rng.normal(0, 15, len(df)))[["origin", "h", "y_pred"]]
    ref = df.assign(y_pred=y + rng.normal(0, 20, len(df)))[["origin", "h", "y_pred"]]
    return cand, ref, filled, observed


def test_matched_se_and_relative_exact():
    cand, ref, filled, observed = _synthetic_frame(n_origins=4)
    frame = BC.matched_se(cand, ref, filled, observed)
    assert len(frame) == 4 * 72
    rel = BC.relative_mse(frame)
    assert rel == pytest.approx(float(frame["se_c"].mean() / frame["se_r"].mean() - 1))
    assert -0.9 < rel < 0.9


def test_time_blocks_partition():
    origins = pd.date_range("2025-08-04", periods=84, freq="6h", tz="UTC")
    blocks = BC.time_blocks(origins, 7)
    assert len(blocks) == 3
    assert sorted(np.concatenate(blocks).tolist()) == list(range(84))
    assert all(len(b) > 0 for b in blocks)


def test_boot_relative_degenerate_and_deterministic():
    cand, ref, filled, observed = _synthetic_frame(n_origins=84)
    frame = BC.matched_se(cand, ref, filled, observed)
    blocks = BC.time_blocks(frame["origin"].unique(), 7)
    a = BC.boot_relative(frame, blocks, b=100, seed=7)
    b2 = BC.boot_relative(frame, blocks, b=100, seed=7)
    assert a == b2
    assert a["lo"] <= a["point"] <= a["hi"]
    assert a["lo"] < a["hi"]  # noisy synthetic => non-degenerate CI
    same = frame.copy()
    same["se_c"] = same["se_r"]
    c = BC.boot_relative(same, blocks, b=50, seed=7)
    assert c["point"] == 0.0 and c["lo"] <= 0.0 <= c["hi"]
    with pytest.raises(ValueError):
        BC.boot_relative(frame.iloc[:0], blocks, b=10)


def test_boot_mean_joint_replicates():
    frames, blks = {}, {}
    for i, name in enumerate(("dev", "selection", "audit")):
        cand, ref, filled, observed = _synthetic_frame(n_origins=60, seed=i)
        f = BC.matched_se(cand, ref, filled, observed)
        frames[name] = f
        blks[name] = BC.time_blocks(f["origin"].unique(), 7)
    m1 = BC.boot_mean(frames, blks, b=100, seed=7)
    m2 = BC.boot_mean(frames, blks, b=100, seed=7)
    assert m1 == m2
    assert m1["lo"] <= m1["point"] <= m1["hi"]
    assert 0.0 <= m1["p_ge2"] <= 1.0
    assert m1["point"] == pytest.approx(float(np.mean([BC.relative_mse(f) for f in frames.values()])))


def test_jackknife_mechanics_with_stub(monkeypatch):
    calls = []

    def stub(canonical, filled, observed, origins):
        calls.append(sorted(pd.DatetimeIndex(origins).tolist()))
        n = len(origins)
        return {"f1": 0.4 + 0.01 * (n % 7), "tp": n, "fp": 1, "fn": 2, "indeterminate": None}

    monkeypatch.setattr(BC, "event_report", stub)
    cand, _, filled, observed = _synthetic_frame(n_origins=200)
    out = BC.jackknife_f1(cand, filled, observed, block_weeks=4)
    assert len(calls) == 1 + len(out["drops"])  # full + one per kept block
    assert out["full"]["f1"] == pytest.approx(0.4 + 0.01 * (200 % 7))
    assert len(out["drops"]) >= 1
    for d in out["drops"]:
        assert d["n_kept"] < 200  # each drop really removes a block
    vals = [d["f1"] for d in out["drops"]]
    assert out["min"] == pytest.approx(min(vals)) and out["max"] == pytest.approx(max(vals))


def test_owned_files_scratch_no_vendor():
    files = sorted(p.name for p in PKG.iterdir() if p.is_file() and not p.name.startswith("__"))
    files = [f for f in files if not f.endswith(".pyc")]
    assert len(files) <= 5, files
    assert {"bootspec.py", "bootci.py", "runner.py"} <= set(files)
    blobs = _blobs()
    runner = blobs["runner.py"]
    for bad in FORBIDDEN:
        assert bad not in runner and bad not in blobs["bootci.py"], bad
    for bad in COPY_BANS:
        for name, text in blobs.items():
            assert bad not in text, f"{name}: {bad}"
    assert "cuda" not in runner  # CPU-only, no GPU lease path
    assert "gpu_lease" not in runner and "EXPOS_GPUS" not in runner
    assert "expos run" not in runner and "run E0" not in runner
    ids = set(re.findall(r"E0\d{3}", runner + blobs["bootci.py"]))
    assert ids <= ALLOWED_IDS, ids - ALLOWED_IDS
    assert 'store/scratch/worker-d84' in runner
    assert "72h" in runner and "poison.json" in runner and "ci_table.json" in runner
    assert "PERIODS = dict(ia.PERIODS)" in runner
    assert "import hsplit" not in runner and "import lvlmean" not in runner
    assert "import stack" not in runner  # runner imports are headroom/existing_all only
    for name in ("freeze", "official_score"):
        assert name not in runner


def test_register_blob_would_not_flag():
    blob = "\n".join(_blobs().values())
    assert core.scan(blob) == []
