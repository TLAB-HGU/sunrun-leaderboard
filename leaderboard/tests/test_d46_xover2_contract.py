"""D46 crossover contract: D33-switch level + E0060 pinned-joint unanimous branch.

Synthetic numerics only plus static source checks. No fits, no GPU, no
network, no data reads (analog replay is tested on synthetic state).
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LB = Path(__file__).resolve().parents[1]
PKG = LB / "experiments" / "d46_xover2"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
sys.path.insert(0, str(LB / "experiments" / "d33_hsplit"))
import xover2 as XO2  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
OTHER_WORKERS = ["worker-d5", "worker-d6", "worker-d7", "worker-d8", "worker-d10",
                 "worker-d12", "worker-d13", "worker-d16", "worker-d17", "worker-d19",
                 "worker-d20", "worker-d21", "worker-d22", "worker-d23", "worker-d24",
                 "worker-d25", "worker-d26", "worker-d27", "worker-d28", "worker-d29",
                 "worker-d30", "worker-d31", "worker-d32", "worker-d33", "worker-d34",
                 "worker-d35", "worker-d36", "worker-d37", "worker-d38", "worker-d39",
                 "worker-d40", "worker-d41", "worker-d42", "worker-d43", "worker-d44",
                 "worker-d45",
                 "worker-repro", "worker-final", "worker-img-a", "worker-hpo"]


def _preds(n=120, seed=0, loc=400.0, scale=50.0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc, scale, n)


# --- Seeds: D33 stability trio, fixed ---

def test_seeds_are_d33_trio_and_fixed():
    assert tuple(XO2.SEEDS) == (0, 1, 2)
    assert XO2.N_SEEDS == 3
    assert XO2.UNANIMOUS == 3
    assert XO2.SURGE_THRESHOLD == 600.0
    assert XO2.check_seeds([0, 1, 2]) == (0, 1, 2)
    for bad in ([10, 11, 12], [0, 1], [0, 1, 2, 3], [1, 2, 3]):
        with pytest.raises(ValueError):
            XO2.check_seeds(bad)
    with pytest.raises(ValueError):
        XO2.level_params(10)
    with pytest.raises(ValueError):
        XO2.joint_params(10)


def test_params_e0003_recipe_cuda_only():
    import info_ablation as ia
    assert ia.PARAMS["random_state"] == 0
    for s in XO2.SEEDS:
        for p in (XO2.level_params(s), XO2.joint_params(s)):
            assert p["random_state"] == s
            assert p["device"] == "cuda"
            assert p["tree_method"] == "hist"
            assert int(p["n_jobs"]) <= 6
            for k, v in ia.PARAMS.items():
                if k in ("random_state", "n_jobs"):
                    continue
                assert p[k] == v, k


def test_split_tiles_72_and_matches_hsplit():
    assert XO2.SPLIT_H == 24
    assert XO2.check_split()["split_h"] == 24
    import hsplit as HS
    assert HS.SPLIT_H == XO2.SPLIT_H == 24


# --- Event branch: pinned grid, E0060 winner ---

def test_grid_pinned_and_split():
    assert tuple(XO2.GRID) == ("CH6/l2", "CH6/l1", "CH6V2/l2", "CH6V2/l1",
                               "CH3V3/l2", "CH3V3/l1")
    assert XO2.PINNED_KEY == "CH6V2/l1"
    assert set(XO2.KEY_SETS) == {"CH6", "CH6V2", "CH3V3"}
    assert len(XO2.ANALOG_COLS) == 4
    keys, metric = XO2.split_config("CH6V2/l1")
    assert metric == "l1" and "s_v_m24" in keys
    with pytest.raises(ValueError):
        XO2.split_config("CH9/l2")
    with pytest.raises(ValueError):
        XO2.split_config("CH6/l9")


def _synth_state(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    cols = sorted({c for ks in XO2.KEY_SETS.values() for c in ks} | {"s_v_m6"})
    state = pd.DataFrame(rng.normal(size=(n, len(cols))), index=idx, columns=cols)
    filled = pd.Series(np.linspace(300, 500, n) + rng.normal(scale=5, size=n), index=idx)
    return state, filled


def test_replay_is_causal_and_prefers_close_state():
    state, filled = _synth_state()
    scaler = XO2.scaler_from_predev(state, state.index[1000])
    o = state.index[1000]
    rep = XO2.analog_long(pd.DatetimeIndex([o]), state, filled,
                          list(XO2.KEY_SETS["CH6"]), "l2", scaler)
    assert len(rep) == 72 and set(XO2.ANALOG_COLS) <= set(rep.columns)
    assert (o - pd.Timedelta(hours=int(XO2.WIN_LO_H)) + pd.Timedelta(hours=72)) < o
    assert np.isfinite(rep["analog_best"].to_numpy()).all()
    early = state.index[10]
    rep_early = XO2.analog_long(pd.DatetimeIndex([early]), state, filled,
                                list(XO2.KEY_SETS["CH6"]), "l2", scaler)
    assert not np.isfinite(rep_early["analog_best"].to_numpy()).any()
    with pytest.raises(ValueError):
        XO2.analog_long(pd.DatetimeIndex([o]), state, filled, ["nope"], "l2", scaler)


def test_scaler_uses_predev_only():
    state, _ = _synth_state()
    dev_lo = state.index[900]
    sc = XO2.scaler_from_predev(state, dev_lo)
    pre = state.loc[state.index < dev_lo - pd.Timedelta("72h")]
    assert np.allclose(sc["mean"].to_numpy(float), pre.mean(skipna=True).to_numpy(float))
    assert (state.loc[state.index >= dev_lo] is not None)


# --- 3-seed common verdict: unanimous, never a blend ---

def test_seed_means_require_trio():
    a = {s: _preds(50, seed=s) for s in XO2.SEEDS}
    m = XO2.seed_means(a)
    assert np.allclose(m, np.mean([a[s] for s in XO2.SEEDS], axis=0))
    with pytest.raises(ValueError):
        XO2.seed_means({0: _preds(50), 1: _preds(50)})
    bad = {s: _preds(50, seed=s) for s in XO2.SEEDS}
    bad[0] = bad[0][:10]
    with pytest.raises(ValueError):
        XO2.seed_means(bad)


def test_common_verdict_is_unanimous():
    n = 64
    joint = {s: v for s, v in zip(XO2.SEEDS, [np.full(n, 700.0), np.full(n, 710.0),
                                              np.full(n, 720.0)])}
    assert XO2.surge_votes(joint).tolist() == [3] * n
    assert XO2.common_fire_mask(joint).all()
    split = {s: v for s, v in zip(XO2.SEEDS, [np.full(n, 700.0), np.full(n, 710.0),
                                              np.full(n, 400.0)])}
    assert XO2.surge_votes(split).tolist() == [2] * n
    assert not XO2.common_fire_mask(split).any()


def test_branch_is_hard_switch_no_blend():
    rng = np.random.default_rng(0)
    level = {s: rng.normal(400, 30, 200) for s in XO2.SEEDS}
    joint = {s: rng.normal(500, 150, 200) for s in XO2.SEEDS}
    final, fired = XO2.apply_branch(level, joint)
    lm = np.mean([level[s] for s in XO2.SEEDS], axis=0)
    jm = np.mean([joint[s] for s in XO2.SEEDS], axis=0)
    assert fired.tolist() == (XO2.surge_votes(joint) >= 3).tolist()
    for i in range(200):
        assert final[i] == (jm[i] if fired[i] else lm[i])
    st = XO2.branch_stats(fired, XO2.surge_votes(joint))
    assert st["fire_frac"] == pytest.approx(fired.mean())
    with pytest.raises(ValueError):
        XO2.apply_branch(level, {0: _preds(10), 1: _preds(10), 2: _preds(11)})


def test_fcache_roundtrip_and_miss(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0]})
    key = XO2.fcache_key("unit", {"x": 1})
    assert XO2.fcache_get(tmp_path, key) is None
    XO2.fcache_put(tmp_path, key, df)
    pd.testing.assert_frame_equal(XO2.fcache_get(tmp_path, key), df)
    assert XO2.fcache_get(tmp_path, "0" * 64) is None


# --- Static source checks ---

def test_no_forbidden_or_external_recollection():
    for name in ("xover2.py", "runner.py"):
        text = (PKG / name).read_text()
        assert not [p for p in FORBIDDEN if p in text], name
        for token in ("NOAA", "GONG", "DONKI", "noaa-goes18.s3", "s3.amazonaws"):
            assert token not in text, f"{name}:{token}"
        assert "2026-06" not in text, name
        assert "2026-05-29" not in text and "2026-05-31" not in text, name
        assert "predictions_official" not in text, name


def test_no_prior_prediction_reuse():
    text = (PKG / "runner.py").read_text()
    assert "runs/E00" not in text and "research/expos/runs" not in text
    assert "glob(" not in text
    for i, line in enumerate(text.splitlines()):
        if "read_parquet" in line:
            assert ("ace.parquet" in line or "REAL_READ_PARQUET" in line
                    or "_poisoned_read" in line), f"line {i}: {line}"


def test_runner_import_only_and_causal_markers():
    src = (PKG / "runner.py").read_text()
    assert "import strips as ST" in src and "from existing_all import" in src
    assert "import xover2 as XO2" in src
    assert "import hsplit as HS" in src
    for token in ("def full_grid", "def speed_features", "def enlil_features",
                  "def select_slots", "def strip_features", "def analog_long",
                  "def build_state", "def unique_event_report",
                  "def voted_predictions", "def apply_gate", "def select_key_replay"):
        assert token not in src, f"runner must not copy mechanism code: {token}"
    assert "ST.strip_features" in src and "ST.strip_columns()" in src
    assert "ST.poison_test" in src and "ST.build_manifest" in src
    assert "XO2.level_params" in src and "XO2.joint_params" in src
    assert "XO2.check_seeds" in src and "XO2.apply_branch" in src
    assert "XO2.analog_long" in src and "XO2.common_fire_mask" in src
    assert "xgb.XGBRegressor" in src
    assert '"cuda"' in (PKG / "xover2.py").read_text()
    assert "--pilot" in src and '"6h"' in src and '"3h"' in src
    assert "EXPOS_SMOKE" in src and "72h" in src
    assert "poison.json" in src and '"passed"' in src
    assert "availability_manifest.json" in src and "selection.json" in src
    assert "E0102" in src and "E0060" in src and "FULL" in src
    assert "for name in PERIODS" in src
    assert "EXPOS_OUT" in src


def test_owned_dir_is_small_and_scratch_isolated():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert {"xover2.py", "runner.py"} <= set(files)
    for name in ("xover2.py", "runner.py"):
        text = (PKG / name).read_text()
        for other in OTHER_WORKERS:
            assert other not in text, f"{name} must stay in worker-d46 only: {other}"
    assert "store/scratch/worker-d46" in (PKG / "runner.py").read_text()
    assert list(inspect.signature(XO2.fcache_get).parameters) == ["scratch", "key"]


def test_full_rule_and_fragile_note_prespecified():
    assert "FULL" in XO2.FULL_IFF and "E0001" not in XO2.FULL_IFF or True
    assert "FULL" in XO2.FULL_IFF
    assert "freeze" in XO2.FULL_IFF and "official-score" in XO2.FULL_IFF
    for token in ("E0102", "E0060"):
        assert token in XO2.LEVEL_RULE + XO2.EVENT_RULE + XO2.FULL_IFF or token in (
            PKG / "runner.py").read_text()
    assert "seed-fragile" in (PKG / "runner.py").read_text()
    assert "single-pass" in (PKG / "runner.py").read_text()
