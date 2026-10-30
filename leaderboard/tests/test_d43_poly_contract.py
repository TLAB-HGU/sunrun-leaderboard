"""D43 poly-Lasso contract: cubic standalone + dev-OOF alpha/w, E0003-identical
params/rows + cuda, no forbidden inputs, scratch separation.
Fast: no data reads, no heavy fits (numpy/sklearn synthetic only)."""

import sys
from pathlib import Path

import numpy as np

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d43_polylasso"))
sys.path.insert(0, str(LB / "experiments" / "headroom_audit"))

import polylasso as PL  # noqa: E402
import info_ablation as ia  # noqa: E402

FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay", "official_check",
             "official-check", "predictions_official_truth",
             "datasets--tlabtlab--sunrun-lb-store", "DONKI", "GONG"]
COPY_BANS = ["def strip_features", "def select_slots", "def strip_weights",
             "def lam(", "PARKER_WEST", "CELL_W"]


def test_poly_rule_fixed_and_causal():
    assert PL.DEGREE == 3
    assert len(PL.POLY_BASE_COLS) == 12 and len(set(PL.POLY_BASE_COLS)) == 12
    assert "h" in PL.POLY_BASE_COLS and "rec_mean" in PL.POLY_BASE_COLS
    assert any(c.startswith("strip_") for c in PL.POLY_BASE_COLS)
    assert any(c.startswith("ch_") for c in PL.POLY_BASE_COLS)
    assert any(c.startswith("ace_") for c in PL.POLY_BASE_COLS)
    assert tuple(PL.LASSO_ALPHAS) == (0.5, 2.0, 8.0)
    assert tuple(PL.BLEND_WS) == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert PL.F1_FLOOR == 0.41121495
    assert PL.DEV_SPLIT == 0.7 and PL.PURGE_H == 72
    assert PL.n_poly_features(12, 3) == 454
    assert PL.check_poly()["n_expanded"] == 454
    assert "dev-OOF" in PL.SELECT_RULE and "selection/audit" in PL.SELECT_RULE
    assert "X0090" in PL.SELECT_RULE and "Son" in PL.SELECT_RULE
    assert "72h purge" in PL.SELECT_RULE


def test_fit_params_match_e0003_recipe_infra_only():
    p = PL.fit_params("cuda:1")
    assert set(p) == set(ia.PARAMS) | {"device"}
    for k, v in ia.PARAMS.items():
        if k == "n_jobs":
            continue
        assert p[k] == v, f"param {k} differs from E0003 recipe"
    assert p["tree_method"] == "hist"
    assert str(p["device"]).startswith("cuda")
    assert p["n_jobs"] == PL.FIT_NJOBS == 3
    assert PL.PERIOD_JOBS * PL.FIT_NJOBS <= 10


def test_devices_round_robin_visible_gpus(monkeypatch):
    monkeypatch.delenv("EXPOS_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert PL.device_for(0) == "cuda"
    monkeypatch.setenv("EXPOS_GPUS", "0,1")
    assert PL.device_for(0) == "cuda:0"
    assert PL.device_for(1) == "cuda:1"
    assert PL.device_for(2) == "cuda:0"


def test_full_rule_prespecified_and_report_only():
    assert "full 3h" in PL.FULL_RULE and "E0001" in PL.FULL_RULE
    assert "F1" in PL.FULL_RULE and "0.41121495" in PL.FULL_RULE
    assert "no full run" in PL.FULL_RULE
    assert "freeze" in PL.FULL_RULE.lower() and "official" in PL.FULL_RULE.lower()


def _synthetic_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    import pandas as pd

    data = {c: rng.standard_normal(n) * (10.0 if c == "h" else 50.0) + (36.0 if c == "h" else 400.0)
            for c in PL.POLY_BASE_COLS}
    df = pd.DataFrame(data)
    df["y"] = (400.0 + 0.5 * df["v_lag0"] + 0.02 * (df["h"] ** 2)
               + 5.0 * rng.standard_normal(n))
    return df


def test_poly_expansion_width_and_lasso_sparsity():
    df = _synthetic_frame()
    for a in PL.LASSO_ALPHAS:
        m = PL.build_poly(float(a))
        m.fit(df[list(PL.POLY_BASE_COLS)].to_numpy(), df["y"].to_numpy())
        poly = m.named_steps["poly"]
        assert poly.degree == 3 and poly.include_bias is False
        xt = poly.transform(df[list(PL.POLY_BASE_COLS)].to_numpy()[:8])
        assert xt.shape == (8, 454)
        assert np.isfinite(m.predict(df[list(PL.POLY_BASE_COLS)].to_numpy()[:8])).all()
    # stronger alpha is sparser (or equal) on the same synthetic frame
    nzs = []
    for a in PL.LASSO_ALPHAS:
        m = PL.build_poly(float(a))
        m.fit(df[list(PL.POLY_BASE_COLS)].to_numpy(), df["y"].to_numpy())
        coef = np.asarray(m.named_steps["lasso"].coef_)
        assert len(coef) == 454
        nzs.append(int((np.abs(coef) > 1e-12).sum()))
    assert nzs[0] >= nzs[-1], f"stronger alpha should be sparser {nzs}"
    assert 0 < nzs[0] <= 454


def test_blend_endpoints_and_selection_rules():
    base = np.array([350.0, 500.0, 650.0])
    poly = np.array([300.0, 520.0, 700.0])
    assert np.allclose(PL.blend(base, poly, 0.0), base)
    assert np.allclose(PL.blend(base, poly, 1.0), poly)
    assert np.allclose(PL.blend(base, poly, 0.5), 0.5 * base + 0.5 * poly)
    assert PL.select_alpha({0.5: 100.0, 2.0: 90.0, 8.0: 95.0}) == 2.0
    assert PL.select_alpha({0.5: 90.0, 2.0: 90.0, 8.0: 90.0}) == 0.5
    assert PL.select_weight({0.0: 100.0, 0.25: 95.0, 0.5: 96.0, 0.75: 97.0, 1.0: 98.0},
                            {0.0: 0.42, 0.25: 0.42, 0.5: 0.40, 0.75: 0.39, 1.0: 0.38}) == 0.25
    assert PL.select_weight({0.0: 100.0, 0.25: 95.0, 0.5: 96.0, 0.75: 97.0, 1.0: 98.0},
                            {0.0: 0.30, 0.25: 0.30, 0.5: 0.30, 0.75: 0.30, 1.0: 0.30}) == 0.0
    assert PL.select_weight({0.0: 100.0, 0.25: 95.0, 0.5: 96.0, 0.75: 97.0, 1.0: 98.0},
                            {0.0: 0.30, 0.25: 0.42, 0.5: 0.43, 0.75: 0.44, 1.0: 0.45}) == 0.25
    assert PL.select_weight({a: 100.0 for a in PL.BLEND_WS},
                            {a: None for a in PL.BLEND_WS}) == 0.0


def test_owned_files_and_no_vendored_strips():
    pkg = LB / "experiments" / "d43_polylasso"
    files = sorted(f.name for f in pkg.glob("*.py"))
    assert len(files) <= 5, files
    assert {"polylasso.py", "runner.py"} <= set(files)
    runner = (pkg / "runner.py").read_text()
    assert "import strips" in runner or "import ST" in runner
    assert "import polylasso" in runner or "import PL" in runner
    assert "store/scratch/worker-d43" in runner
    for other in ("worker-d41", "worker-d42", "worker-d40", "worker-d39",
                  "worker-d28", "worker-d22", "worker-d23", "worker-d21"):
        assert other not in runner
    for f in pkg.glob("*.py"):
        text = f.read_text()
        for s in FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        assert "noaa-goes" not in text.lower()
        for s in COPY_BANS:
            assert s not in text, f"{f.name} vendors {s!r}"


def test_runner_blend_poison_gpu_cache_and_smoke():
    runner = (LB / "experiments" / "d43_polylasso" / "runner.py").read_text()
    assert "dev_oof_select" in runner
    assert "unique_event_report" in runner
    assert "PL.blend" in runner and "PL.predict_poly" in runner and "PL.fit_poly" in runner
    assert "select_alpha" in runner and "select_weight" in runner
    assert "ST.poison_test" in runner and '"passed"' in runner
    assert "SystemExit" in runner
    assert "ThreadPoolExecutor" in runner
    assert "device" in runner and "cuda" in runner
    assert "for i, n in enumerate(names)" in runner
    assert "ST.build_manifest" in runner
    assert "fcache" in runner.lower() and "featcache" in runner
    assert "EXPOS_SMOKE" in runner
    assert "design_cols" in runner and "dict.fromkeys" in runner
    assert "X0090" in runner or "Son" in runner
