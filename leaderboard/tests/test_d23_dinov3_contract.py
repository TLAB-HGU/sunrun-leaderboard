"""D23 DINOv3 contract: frozen encoder, solar-geometry strip pooling, PCA<=20, causal."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LB = Path(__file__).resolve().parents[1]
PKG = LB / "experiments" / "d23_dinov3"
sys.path.insert(0, str(PKG))
import dinov3 as D3  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet",
             "naive.parquet", "official-predictions", "baseline-replay.parquet",
             "official_check", "official-check", "predictions_official_truth"]
WINDOW_DATES = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
                "2026-05-29", "2026-05-31", "2026-09-26", "2026-06")
OTHER_SCRATCH = ["worker-d22", "worker-final", "worker-d21", "worker-d20",
                 "worker-repro", "worker-d16"]


def fake_header(rsun=100.0, size=256):
    return {"CRPIX1": (size + 1) / 2, "CRPIX2": (size + 1) / 2, "RSUN": rsun,
            "CROTA": 0.0, "SOLAR_B0": 0.0}


def test_owned_dir_is_small():
    files = sorted(p.name for p in PKG.iterdir() if p.is_file())
    assert len(files) <= 5, files
    pys = [f for f in files if f.endswith(".py")]
    assert set(pys) == {"dinov3.py", "runner.py"}, pys


def test_no_forbidden_or_window_dates_in_new_code():
    for name in ("dinov3.py", "runner.py"):
        text = (PKG / name).read_text()
        for s in FORBIDDEN:
            assert s not in text, (name, s)
        for s in WINDOW_DATES:
            assert s not in text, (name, s)


def test_no_geometry_copy_import_strips():
    for name in ("dinov3.py", "runner.py"):
        text = (PKG / name).read_text()
        assert "def lam(" not in text, name
        assert "PARKER_WEST" not in text, name
    assert "import strips as ST" in (PKG / "runner.py").read_text()


def test_scratch_is_worker_d23_only():
    src = (PKG / "runner.py").read_text() + (PKG / "dinov3.py").read_text()
    assert "worker-d23" in src
    for other in OTHER_SCRATCH:
        assert other not in src, other


def test_model_identity_and_env_pins():
    assert D3.MODEL_ID == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert len(D3.MODEL_SHA256) == 64 and D3.MODEL_BYTES == 342662192
    assert D3.TRANSFORMERS_PIN == "4.56.2" and D3.TORCH_PIN == "2.13.0+cu126"
    lock = (PKG / "env.lock.txt").read_text()
    assert "transformers==4.56.2" in lock and "torch==2.13.0+cu126" in lock


def test_dino_columns_20_total_4_per_strip():
    cols = D3.dino_columns()
    assert len(cols) == 20
    for v in D3.SPEEDS:
        assert [c for c in cols if c.startswith(f"dino_v{v}_")] == \
            [f"dino_v{v}_pca{k}" for k in range(4)]
    assert D3.PCA_PER_STRIP == 4 and D3.N_STRIPS * D3.PCA_PER_STRIP <= 20


def test_patch_lonlat_geometry():
    lon, lat, ondisk = D3.patch_lonlat(fake_header())
    assert lon.shape == (14, 14) and lat.shape == (14, 14)
    assert 60 <= int(ondisk.sum()) <= 140
    assert abs(float(lon[6:8, 6:8].mean())) < 8.0  # disk centre ~0 deg
    west_col = int(np.where(ondisk.any(axis=0))[0][-1])
    east_col = int(np.where(ondisk.any(axis=0))[0][0])
    north_row = int(np.where(ondisk.any(axis=1))[0][0])
    assert float(lon[:, west_col][ondisk[:, west_col]].max()) > 60.0  # west positive
    assert float(lon[:, east_col][ondisk[:, east_col]].min()) < -60.0  # east negative
    assert float(lat[north_row, :][ondisk[north_row, :]].max()) > 30.0  # north up
    row7 = lon[7, :][ondisk[7, :]]
    assert np.all(np.diff(row7) > 0)  # lon grows westward


def test_strip_means_membership_and_empty_nan():
    rng = np.random.RandomState(0)
    toks = rng.normal(size=(196, 768)).astype(np.float32)
    lon = np.linspace(-95, 95, 196)
    use = np.ones(196, dtype=bool)
    means, cnt = D3.strip_means(toks, lon, use, np.array([[0.0] * 5]))
    assert means.shape == (1, 5, 768)
    member = np.abs(lon) <= 5.0
    assert cnt[0, 0] == member.sum()
    np.testing.assert_allclose(means[0, 0], toks[member].mean(0), rtol=1e-5)
    means2, cnt2 = D3.strip_means(toks, lon, use, np.array([[500.0] * 5]))
    assert cnt2.sum() == 0 and bool(np.isnan(means2).all())


def test_pca_fit_transform_shapes_and_nan():
    rng = np.random.RandomState(1)
    sample = (rng.normal(size=(200, 5, 768)) * 10).astype(np.float32)
    pcas = D3.fit_pcas(sample)
    assert len(pcas) == 5 and all(p.n_components_ == 4 for p in pcas)
    vecs = rng.normal(size=(50, 5, 768)).astype(np.float32)
    vecs[3, 2, :] = np.nan
    vecs[4, :, :] = np.nan
    out = D3.transform_pcas(pcas, vecs)
    assert out.shape == (50, 20)
    assert bool(np.isnan(out[3, 8:12]).all())  # only strip 2 NaN
    assert bool(np.isfinite(out[3, :8]).all()) and bool(np.isfinite(out[3, 12:]).all())
    assert bool(np.isnan(out[4]).all())  # fully missing row stays NaN
    assert int(np.isfinite(out).all(axis=1).sum()) == 48
    for p in pcas:
        assert float(p.explained_variance_ratio_.sum()) <= 1.0


def test_poison_check_logic():
    rng = np.random.RandomState(2)
    base = rng.normal(size=(100, 20)).astype(np.float32)
    mod = base.copy()
    mod[80:] += 1.0
    pre = np.arange(100) < 80
    post = ~pre
    chk = D3.poison_check(base, mod, pre, post, D3.dino_columns())
    assert chk["passed"] and chk["pre_invariant"] and len(chk["moved_columns"]) == 20
    chk2 = D3.poison_check(base, base.copy(), pre, post, D3.dino_columns())
    assert not chk2["passed"] and chk2["moved_columns"] == []


def test_cache_roundtrip_and_version_guard(tmp_path):
    payload = {"tokens": np.zeros((2, 768), np.float32),
               "lon_deg": np.zeros(2, np.float32), "lat_deg": np.zeros(2, np.float32),
               "ondisk": np.array([True, False]), "valid_patch": np.ones(2, np.float32),
               "low": np.float32(0.1), "high": np.float32(5.0),
               "model_sha": np.array(D3.MODEL_SHA256),
               "cache_version": np.int64(D3.CACHE_VERSION)}
    D3.save_cache("2000-01-01T00:00:00Z", payload)
    try:
        got = D3.load_cache("2000-01-01T00:00:00Z", 0.1, 5.0)
        assert got is not None and got["tokens"].shape == (2, 768)
        assert D3.load_cache("2000-01-01T00:00:00Z", 0.2, 5.0) is None
    finally:
        D3.cache_file("2000-01-01T00:00:00Z").unlink(missing_ok=True)
    assert D3.geometry_record()["pca_total"] == 20


def test_runner_contract_pilot_purge_poison_fixedseed():
    src = (PKG / "runner.py").read_text()
    assert "--pilot" in src and "6h" in src
    assert "72h" in src and "XGBRegressor(**ia.PARAMS)" in src
    assert "poison.json" in src and "availability_manifest.json" in src
    assert "RandomState(0)" in src and "fit_org" in src
    assert "Optuna" not in src and "optuna" not in src
    assert "fine-tun" not in src.lower() or "no fine-tuning" in src.lower()


def test_e0055_e0058_harness_fixes_locked():
    """Livraison check: E0055-58 rc!=0 chain fixes stay in the code.

    E0055 ragged-train assertion -> dino_for_rows maps arbitrary (origin, h)
    rows via unique/inverse indices; E0056 numpy.str_ Timestamp TypeError ->
    explicit str() conversions; E0057 DatetimeIndex.median AttributeError ->
    Series-wrapped median; E0058 ndarray.to_numpy AttributeError -> Series
    comparison before to_numpy. rc!=0 without outputs is a harness bug, never
    a scientific negative (X0061); these markers keep the fixes from rotting.
    """
    src = (PKG / "runner.py").read_text()
    # E0055: ragged-row handler (unique + inverse map + per-row h index).
    assert "def dino_for_rows" in src
    assert "np.unique(row_o, return_inverse=True)" in src
    assert "(inv[m] - s) * 72 + row_h[m]" in src
    # E0056: numpy.str_ must never reach pd.Timestamp unconverted.
    assert "np.asarray(row_o, dtype=str)" in src
    assert "[str(x) for x in u[s:s + chunk]]" in src
    assert "[str(tr_orgs[i]) for i in fit_org]" in src
    # E0057: median over a Series, never over a DatetimeIndex.
    assert "pd.Series(pd.to_datetime(" in src
    assert ".median().isoformat()" in src
    assert ".median()" in src and "DatetimeIndex" not in src.split(".median()")[0].split("\n")[-1]
    # E0058: boolean masks come from a Series comparison, then to_numpy().
    assert "sall = pd.Series(pd.to_datetime(sall, utc=True))" in src
    assert "(sall <=" in src and "(sall >" in src
    assert ".to_numpy()" in src
    # Import-shape fix companion: strips geometry shared by import, never copied.
    assert "import strips as ST" in src
    assert "from features import read_suvi_frame" in src


def test_scratch_is_worker_d23b():
    src = (PKG / "runner.py").read_text() + (PKG / "dinov3.py").read_text()
    assert "worker-d23b" in src
    for other in OTHER_SCRATCH:
        assert other not in src, other


def test_f1_guard_contract_dev_only():
    """F1-guard variant: --f1-guard 2-arm gate on dev-OOF only, floor 0.4112."""
    src = (PKG / "runner.py").read_text()
    assert "--f1-guard" in src and "F1_FLOOR" in src
    assert "0.41121495" in src and "selection.json" in src
    assert "unique_event_report" in src and "dev_oof_scores" in src
    assert "select_arm_f1guard" in src
    assert "PERIODS[\"dev\"]" in src
    assert "BASE_ARM" in src and "DINO_ARM" in src
    # The gate consumes dev-window scores only: its only inputs are the two
    # arms' dev-OOF scores, so selection/audit can never feed any choice.
    assert "def select_arm_f1guard(base_mse, base_f1, dino_mse, dino_f1)" in src
    gate_fn = src.split("def select_arm_f1guard", 1)[1].split("def ", 1)[0]
    assert "PERIODS[" not in gate_fn
    assert "abstain" in gate_fn
    assert "Optuna" not in src and "optuna" not in src


def test_directive7_parallel_period_fits_capped():
    """Directive-7: 3 period fits in parallel workers, total CPU <= 20, cache reused."""
    src = (PKG / "runner.py").read_text()
    assert "--jobs" in src and "FIT_PARAMS" in src
    assert '"n_jobs": FIT_NJOBS' in src and "FIT_NJOBS = 6" in src and "18 <= 20" in src
    assert "ThreadPoolExecutor(max_workers=a.jobs)" in src
    assert "XGBRegressor(**FIT_PARAMS)" in src
    assert "XGBRegressor(**ia.PARAMS)" in src  # gate arms keep the reference params
    d3src = (PKG / "dinov3.py").read_text()
    assert "CACHE_VERSION" in d3src and "MODEL_SHA256" in d3src
    assert "cache_version" in d3src and "model_sha" in d3src


torch = pytest.importorskip("torch")


def test_encoder_frozen_and_token_shape_cpu():
    from huggingface_hub import snapshot_download

    try:
        snap = snapshot_download(D3.MODEL_ID, revision=D3.MODEL_REVISION)
    except Exception:
        pytest.skip(" gated weights not cached")
    import hashlib

    h = hashlib.sha256()
    with open(Path(snap) / "model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == D3.MODEL_SHA256
    model = D3.load_encoder("cpu")
    assert model.training is False
    assert all(p.requires_grad is False for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == 85660416
    toks = D3.encode_batch(model, np.zeros((1, 3, 224, 224), np.float32), "cpu")
    assert toks.shape == (1, 196, 768) and np.isfinite(toks).all()
