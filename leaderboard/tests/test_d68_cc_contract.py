"""D68 corr-cluster contract: one representative per correlated cluster, dev-only.

Fast: no data reads, no heavy fits (numpy synthetic only, tiny XGB).
"""

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d68_corrcluster"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import ccspec as CC  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_dual_tag_and_experience_citation():
    assert CC.FAMILY_TAG == "[family=xgb]"
    assert CC.AXES_TAG.startswith("[axes:") and "sel=" in CC.AXES_TAG
    assert "corr-cluster" in CC.AXES_TAG
    assert CC.HYPOTHESIS.startswith("[family=xgb] [axes:")
    for token in ("E0133", "X0121", "E0001"):
        assert token in CC.HYPOTHESIS, f"hypothesis must cite {token}"
    assert "dev-OOF" in CC.HYPOTHESIS or "dev-only" in CC.HYPOTHESIS
    assert "freeze" in CC.HYPOTHESIS.lower() and "official-score" in CC.HYPOTHESIS
    assert "6h" in CC.HYPOTHESIS and "gpu1" in CC.HYPOTHESIS


def test_cluster_rule_fixed_and_dev_only():
    assert CC.RHO_THR == 0.85
    assert CC.check_spec()["rho_thr"] == 0.85
    assert CC.VAL_LO_UTC == "2025-10-15T00:00:00Z" and CC.PURGE_H == 72 and CC.SEED == 0
    assert "dev-OOF" in CC.SELECT_RULE and "selection/audit" in CC.SELECT_RULE
    assert "never read" in CC.SELECT_RULE
    assert "E0133" in CC.SELECT_RULE and "X0121" in CC.SELECT_RULE
    assert "|r| >= 0.85" in CC.SELECT_RULE
    import pandas as pd
    assert pd.Timestamp(CC.DEV_LO_UTC) < pd.Timestamp(CC.VAL_LO_UTC) < pd.Timestamp(CC.DEV_END_UTC)
    n = 200
    rng = np.random.default_rng(0)
    origins = pd.date_range("2025-08-04", "2025-11-27 23:00", freq="6h", tz="UTC")
    df = pd.DataFrame({"origin": rng.choice(origins, n),
                       "target_t": rng.choice(origins, n), "y": rng.normal(400, 50, n)})
    df = df.sort_values("origin").reset_index(drop=True)
    tr, va = CC.split_dev_oof(df)
    assert (va["origin"] >= pd.Timestamp(CC.DEV_LO_UTC)).all()
    assert (va["origin"] <= pd.Timestamp(CC.DEV_END_UTC)).all()
    assert (tr["target_t"] < pd.Timestamp(CC.VAL_LO_UTC) - pd.Timedelta(hours=72)).all()


def _dev_frame(n=600, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2025-08-04", "2025-10-01", freq="6h", tz="UTC")
    o = rng.choice(origins, n)
    tt = o + pd.to_timedelta(rng.integers(1, 72, n), unit="h")
    y = rng.normal(400, 50, n)
    # a correlates strongly with y, b is a + tiny noise (|r|>>0.85), c is noise
    a = y * 0.5 + rng.normal(0, 2, n)
    b = a + rng.normal(0, 0.5, n)
    c = rng.normal(0, 1, n)
    df = pd.DataFrame({"origin": o, "target_t": tt, "y": y,
                       "h": rng.integers(1, 73, n).astype(float),
                       "a": a, "b": b, "c": c})
    return df.sort_values("target_t").reset_index(drop=True)


def test_clusters_merge_correlated_and_pick_signal_rep():
    df = _dev_frame()
    cols = ["h", "a", "b", "c"]
    info = CC.cluster_columns(df, cols)
    clusters = [sorted(m) for m in info["clusters"]]
    assert sorted(["a", "b"]) in clusters, clusters
    assert ["c"] in clusters, clusters
    reps = CC.select_representatives(df, cols)
    assert reps[0] == "h" and "h" in reps
    assert ("a" in reps) != ("b" in reps), reps  # exactly one of the pair
    assert "c" in reps, reps
    assert len(reps) == 3, reps
    assert CC.select_representatives(df, cols) == reps  # deterministic


def test_singletons_kept_and_h_never_clustered():
    import pandas as pd
    df = _dev_frame()
    cols = ["h", "a", "b", "c"]
    info = CC.cluster_columns(df, cols)
    for members in info["clusters"]:
        assert "h" not in members, "h must never enter a cluster"
    reps = CC.select_representatives(df, cols)
    assert reps[0] == "h"
    assert len(set(reps)) == len(reps) and set(reps) <= set(cols)


def test_fit_params_match_e0003_recipe_infra_only():
    p = CC.fit_params("cuda:1")
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert str(p["device"]).startswith("cuda")
    assert p["n_jobs"] == CC.FIT_NJOBS == 3
    assert CC.PERIOD_JOBS * CC.FIT_NJOBS <= 10


def test_devices_round_robin_visible_gpus(monkeypatch):
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert CC.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "1")
    assert CC.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert CC.device_for(0) == "cuda:0"
    assert CC.device_for(1) == "cuda:1"
    assert CC.device_for(2) == "cuda:0"


def test_full_rule_prespecified_and_report_only():
    assert "full 3h" in CC.FULL_RULE and "E0001" in CC.FULL_RULE
    assert "F1" in CC.FULL_RULE and "0.30" in CC.FULL_RULE
    assert "no full run" in CC.FULL_RULE
    assert "freeze" in CC.FULL_RULE.lower() and "official-score" in CC.FULL_RULE
    assert "E0003" in CC.FULL_RULE and "seed-repro" in CC.FULL_RULE


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d68_corrcluster"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"ccspec.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import ccspec" in runner or "import CC" in runner
    assert "store/scratch/worker-d68" in runner
    assert "worker-d67" not in runner
    for other in ("worker-d59", "worker-d60", "worker-d61", "worker-d63",
                  "worker-d64", "worker-d65", "worker-d66", "worker-d45", "worker-d33"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"


def test_runner_cluster_poison_gpu_cache_and_smoke():
    runner = (LB / "experiments" / "d68_corrcluster" / "runner.py").read_text()
    assert "dev_oof_cluster" in runner
    assert "cluster_columns" in runner and "select_representatives" in runner
    assert "representative" in runner
    assert "cluster_table" in runner
    assert "ST.poison_test" in runner and '"passed"' in runner
    assert "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner
    assert "device" in runner and "cuda" in runner
    assert "for i, n in enumerate(names)" in runner
    assert "ST.build_manifest" in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "design_cols" in runner and "dict.fromkeys" in runner
    assert "E0133" in runner and "X0121" in runner
