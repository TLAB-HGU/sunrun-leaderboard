"""D44 crossover contract: E0074 vote level + E0060 analog-key hard gate.

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
PKG = LB / "experiments" / "d44_xover"
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))
import xover as XO  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
OTHER_WORKERS = ["worker-d5", "worker-d6", "worker-d7", "worker-d8", "worker-d10",
                 "worker-d12", "worker-d13", "worker-d16", "worker-d17", "worker-d19",
                 "worker-d20", "worker-d21", "worker-d22", "worker-d23", "worker-d24",
                 "worker-d25", "worker-d26", "worker-d27", "worker-d28", "worker-d29",
                 "worker-d30", "worker-d31", "worker-d32", "worker-d33", "worker-d34",
                 "worker-d35", "worker-d36", "worker-d37", "worker-d38", "worker-d39",
                 "worker-d40", "worker-d41", "worker-d42", "worker-d43",
                 "worker-repro", "worker-final", "worker-img-a", "worker-hpo"]


def _preds(n=120, seed=0, loc=400.0, scale=50.0):
    rng = np.random.default_rng(seed)
    return rng.normal(loc, scale, n)


# --- Level branch: E0074 vote rule verbatim ---

def test_seeds_fresh_fixed_and_disjoint():
    assert tuple(XO.SEEDS) == (10, 11, 12, 13, 14)
    assert XO.N_SEEDS == 5
    assert XO.MAJORITY == 3
    assert XO.SURGE_THRESHOLD == 600.0
    assert set(XO.SEEDS).isdisjoint(set(XO.EARLY_SEEDS))
    assert XO.check_seeds([10, 11, 12, 13, 14]) == (10, 11, 12, 13, 14)
    for bad in ([0, 1, 2, 3, 4], [10, 11, 12, 13], [10, 11, 12, 13, 14, 15]):
        with pytest.raises(ValueError):
            XO.check_seeds(bad)
    with pytest.raises(ValueError):
        XO.seed_params(0)


def test_seed_params_e0003_recipe_cuda_only():
    import info_ablation as ia
    assert ia.PARAMS["random_state"] == 0
    for s in XO.SEEDS:
        p = XO.seed_params(s)
        assert p["random_state"] == s
        assert p["device"] == "cuda"
        assert p["tree_method"] == "hist"
        assert int(p["n_jobs"]) <= 4
        for k, v in ia.PARAMS.items():
            if k in ("random_state", "n_jobs"):
                continue
            assert p[k] == v, k


def test_vote_majority_semantics():
    n = 64
    pred = {s: v for s, v in zip(XO.SEEDS, [np.full(n, 700.0), np.full(n, 710.0),
                                            np.full(n, 720.0), np.full(n, 400.0),
                                            np.full(n, 410.0)])}
    assert np.allclose(XO.voted_predictions(pred), np.full(n, 710.0))
    assert XO.surge_votes(pred).tolist() == [3] * n
    assert XO.majority_surge_mask(pred).all()
    calm = {s: np.full(n, 400.0 + 10 * i) for i, s in enumerate(XO.SEEDS)}
    calm[XO.SEEDS[0]] = np.full(n, 900.0)  # single outlier loses 1-4
    assert np.allclose(XO.voted_predictions(calm), np.full(n, 425.0))
    assert not XO.majority_surge_mask(calm).any()


def test_vote_rejects_bad_inputs():
    with pytest.raises(ValueError):
        XO.voted_predictions({10: _preds(50), 11: _preds(50)})
    bad = {s: _preds(50, seed=s) for s in XO.SEEDS}
    bad[XO.SEEDS[0]] = bad[XO.SEEDS[0]][:10]
    with pytest.raises(ValueError):
        XO.voted_predictions(bad)


# --- Event branch: pinned analog-key grid ---

def test_grid_pinned_and_split():
    assert tuple(XO.GRID) == ("CH6/l2", "CH6/l1", "CH6V2/l2", "CH6V2/l1",
                             "CH3V3/l2", "CH3V3/l1")
    assert set(XO.KEY_SETS) == {"CH6", "CH6V2", "CH3V3"}
    assert len(XO.KEY_SETS["CH6"]) == 6 and len(XO.ANALOG_COLS) == 4
    keys, metric = XO.split_config("CH6V2/l2")
    assert metric == "l2" and "s_v_m24" in keys
    with pytest.raises(ValueError):
        XO.split_config("CH9/l2")
    with pytest.raises(ValueError):
        XO.split_config("CH6/l9")


def _synth_state(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    cols = sorted({c for ks in XO.KEY_SETS.values() for c in ks} | {"s_v_m6"})
    state = pd.DataFrame(rng.normal(size=(n, len(cols))), index=idx, columns=cols)
    filled = pd.Series(np.linspace(300, 500, n) + rng.normal(scale=5, size=n), index=idx)
    return state, filled


def test_replay_is_causal_and_prefers_close_state():
    state, filled = _synth_state()
    scaler = XO.scaler_from_predev(state, state.index[1000])
    o = state.index[1000]
    rep = XO.analog_long(pd.DatetimeIndex([o]), state, filled,
                         list(XO.KEY_SETS["CH6"]), "l2", scaler)
    assert len(rep) == 72 and set(XO.ANALOG_COLS) <= set(rep.columns)
    # every replayed target predates the query origin (pool is O-35d..O-20d)
    assert (o - pd.Timedelta(hours=int(XO.WIN_LO_H)) + pd.Timedelta(hours=72)) < o
    assert np.isfinite(rep["analog_best"].to_numpy()).all()
    early = state.index[10]  # no pool history -> NaN, gate closed
    rep_early = XO.analog_long(pd.DatetimeIndex([early]), state, filled,
                               list(XO.KEY_SETS["CH6"]), "l2", scaler)
    assert not np.isfinite(rep_early["analog_best"].to_numpy()).any()
    with pytest.raises(ValueError):
        XO.analog_long(pd.DatetimeIndex([o]), state, filled, ["nope"], "l2", scaler)


def test_scaler_uses_predev_only():
    state, _ = _synth_state()
    dev_lo = state.index[900]
    sc = XO.scaler_from_predev(state, dev_lo)
    pre = state.loc[state.index < dev_lo - pd.Timedelta("72h")]
    assert np.allclose(sc["mean"].to_numpy(float), pre.mean(skipna=True).to_numpy(float))
    assert (state.loc[state.index >= dev_lo] is not None)


# --- Selection: dev-OOF only ---

def test_select_key_replay_picks_best_feasible():
    mse = {k: 100.0 for k in XO.GRID}
    f1 = {k: 0.30 for k in XO.GRID}
    mse["CH6V2/l2"], f1["CH6V2/l2"] = 90.0, 0.45
    mse["CH3V3/l1"], f1["CH3V3/l1"] = 80.0, 0.44
    w, info = XO.select_key_replay(mse, f1, 95.0)
    assert w == "CH6V2/l2" and not info["abstained"]
    assert info["feasible"] == ["CH3V3/l1", "CH6V2/l2"]


def test_select_key_replay_abstains_without_feasible_f1():
    mse = {k: 100.0 for k in XO.GRID}
    f1 = {k: 0.30 for k in XO.GRID}  # below floor even when feasible
    w, info = XO.select_key_replay(mse, f1, 200.0)
    assert w is None and info["abstained"]
    mse_bad = {k: 500.0 for k in XO.GRID}  # nothing beats the voted level
    w2, info2 = XO.select_key_replay(mse_bad, {k: 0.9 for k in XO.GRID}, 100.0)
    assert w2 is None and info2["abstained"] and info2["feasible"] == []
    with pytest.raises(ValueError):
        XO.select_key_replay({"CH6/l2": 1.0}, {k: 0.5 for k in XO.GRID}, 2.0)
    with pytest.raises(ValueError):
        XO.select_key_replay({k: 1.0 for k in XO.GRID}, {k: 0.5 for k in XO.GRID},
                             float("nan"))


def test_select_signature_is_dev_only():
    assert list(inspect.signature(XO.select_key_replay).parameters) == [
        "dev_mse", "dev_f1", "vote_dev_mse"]


# --- Hard gate: never a blend ---

def test_gate_is_a_hard_switch():
    rng = np.random.default_rng(0)
    voted = rng.normal(400, 30, 200)
    analog = rng.normal(500, 150, 200)
    final, fired = XO.apply_gate(voted, analog)
    assert fired.tolist() == (np.isfinite(analog) & (analog >= 600.0)).tolist()
    for i in range(200):
        assert final[i] == (analog[i] if fired[i] else voted[i])
    assert XO.gate_stats(fired)["fire_frac"] == pytest.approx(fired.mean())


def test_gate_boundary_and_nan():
    voted = np.full(8, 400.0)
    analog = np.array([600.0, 599.9, np.nan, 900.0, 100.0, 650.0, np.inf, -np.inf])
    final, fired = XO.apply_gate(voted, analog)
    assert fired.tolist() == [True, False, False, True, False, True, False, False]
    assert final[0] == 600.0 and final[2] == 400.0
    with pytest.raises(ValueError):
        XO.apply_gate(voted, analog[:4])
    with pytest.raises(ValueError):
        XO.apply_gate(np.array([np.nan, 1.0]), np.array([700.0, 1.0]))


def test_fcache_roundtrip_and_miss(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0]})
    key = XO.fcache_key("unit", {"x": 1})
    assert XO.fcache_get(tmp_path, key) is None
    XO.fcache_put(tmp_path, key, df)
    pd.testing.assert_frame_equal(XO.fcache_get(tmp_path, key), df)
    assert XO.fcache_get(tmp_path, "0" * 64) is None


# --- Static source checks ---

def test_no_forbidden_or_external_recollection():
    for name in ("xover.py", "runner.py"):
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
    assert "import xover as XO" in src
    assert "unique_event_report" in src
    for token in ("def full_grid", "def speed_features", "def enlil_features",
                  "def select_slots", "def strip_features", "def analog_long",
                  "def build_state", "def unique_event_report",
                  "def voted_predictions", "def apply_gate", "def select_key_replay"):
        assert token not in src, f"runner must not copy mechanism code: {token}"
    assert "ST.strip_features" in src and "ST.strip_columns()" in src
    assert "ST.poison_test" in src and "ST.build_manifest" in src
    assert "XO.seed_params" in src and "XO.voted_predictions" in src
    assert "XO.check_seeds" in src and "XO.apply_gate" in src
    assert "XO.select_key_replay" in src and "XO.analog_long" in src
    assert "xgb.XGBRegressor" in src
    assert '"cuda"' in (PKG / "xover.py").read_text()
    assert "--pilot" in src and '"6h"' in src and '"3h"' in src
    assert "EXPOS_SMOKE" in src and "72h" in src
    assert "poison.json" in src and '"passed"' in src
    assert "availability_manifest.json" in src and "selection.json" in src
    assert "E0074" in src and "E0060" in src and "FULL" in src
    assert "for name in PERIODS" in src
    assert "EXPOS_OUT" in src


def test_owned_dir_is_small_and_scratch_isolated():
    files = sorted(p.name for p in PKG.iterdir() if p.suffix == ".py")
    assert len(files) <= 5, files
    assert {"xover.py", "runner.py"} <= set(files)
    for name in ("xover.py", "runner.py"):
        text = (PKG / name).read_text()
        for other in OTHER_WORKERS:
            assert other not in text, f"{name} must stay in worker-d44 only: {other}"
    assert "store/scratch/worker-d44" in (PKG / "runner.py").read_text()
    assert list(inspect.signature(XO.fcache_get).parameters) == ["scratch", "key"]
