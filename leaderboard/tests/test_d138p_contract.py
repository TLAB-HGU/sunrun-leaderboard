"""D138 T2b-physics contract: monthly GONG receipt + SH-PFSS/WSA + rocv-only proof.

Fast: pure-logic checks + filesystem/receipt/graph scans only. No fits, no network
(collection evidence is recorded constants + cached bytes verified by hashes),
no forbidden reads, no foreign experiment runs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d138_t2bflux" / "physics"))

import collect as CO  # noqa: E402
import pfss as PF  # noqa: E402
import phys_features as FE  # noqa: E402
import spec as SP  # noqa: E402

PKG = LB / "experiments" / "d138_t2bflux" / "physics"
REPORT = PKG / "report.md"
SCRATCH = LB / "store" / "scratch" / "worker-t2p"
RECEIPT = SCRATCH / "collect_receipt.json"
GRAPH = LB.parent / "research" / "expos" / "graph.jsonl"

PY_FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
                "official-predictions", "baseline-replay", "predictions_official_truth",
                "official_check", "official-check",
                "datasets--tlabtlab--sunrun-lb-store",
                "official-score", "official_score",
                "core.freeze", "core.register(", "core.run(",
                "research/expos/runs/E", "read_parquet",
                "requests.get", "urlretrieve", "tarfile.open", "fits.open",
                "gpu_lease"]
NET_OK_ONLY_IN = {"collect.py"}
EXPOS_IMPORT_OK_ONLY_IN = {"rocv_runner.py"}
OTHER_WORKERS = ["worker-t2b", "worker-t2g", "worker-t1b", "worker-t1c", "worker-t3b",
                 "worker-t3c", "worker-t3d", "worker-t3e", "worker-t3f", "worker-t2n",
                 "worker-d133", "worker-d33", "worker-d52", "worker-img"]


def _synth_maps():
    return [
        {"period": "2025-12", "path": "synth-a",
         "date_obs": pd.Timestamp("2025-12-01T11:14:00Z"), "lon0": 0.0,
         "body_sha256": "0" * 64,
         "brss_abs": np.full(360, 0.15), "brfoot_abs": np.full(360, 10.0),
         "theta_b": np.full(360, 8.0)},
        {"period": "2026-01", "path": "synth-b",
         "date_obs": pd.Timestamp("2026-01-01T11:04:00Z"), "lon0": 0.0,
         "body_sha256": "1" * 64,
         "brss_abs": np.full(360, 0.10), "brfoot_abs": np.full(360, 14.0),
         "theta_b": np.full(360, 12.0)},
    ]


def test_spec_tags_seeds_and_rule():
    assert SP.DIRECTION == "D138" and SP.OWNER == "worker-t2p"
    assert SP.FAMILY_TAG == "[family=xgb]" and SP.AXES_TAG == "[axes: data=gong-pfss-wsa]"
    assert SP.HYPOTHESIS.startswith("[family=xgb]") and "[axes: data=gong-pfss-wsa]" in SP.HYPOTHESIS
    for cite in ("E0003", "E0206", "X0005", "X0004", "rocv", "6/8", "mean+2SE<0"):
        assert cite in SP.HYPOTHESIS, f"hypothesis missing {cite}"
    assert "vintage" in SP.HYPOTHESIS and "reprocess" in SP.HYPOTHESIS
    assert "PFSS" in SP.HYPOTHESIS and "WSA" in SP.HYPOTHESIS
    assert set(SP.EXPERIENCE) == {"X0005", "X0188", "X0004"}
    assert tuple(SP.SEEDS_ROCV) == (0, 1, 2)
    assert "selection/audit" in SP.SELECT_RULE and "rocv" in SP.SELECT_RULE
    assert SP.FIT_NJOBS <= 8
    assert len(SP.FEATURES) == 8 and "wsa_v" in SP.FEATURES and "theta_b" in SP.FEATURES
    assert "dipole" not in " ".join(SP.FEATURES)
    assert str(SP.SCRATCH) == "store/scratch/worker-t2p"
    assert SP.LMAX == 30 and SP.RSS == 2.5


def test_proxy_verdicts_and_header_hashes():
    r = json.loads(RECEIPT.read_text())
    assert len(r["files"]) == 24
    assert all(CO.verify_header_sha256(f) for f in r["files"])
    assert all(p["proxy_obs_plus_2h"] in ("PASS-timing-only", "ARCHIVE-timing-only")
               for p in r["proxy"])
    assert all(p["usable_for_features"] for p in r["proxy"])
    resids = [f["cr_resid_deg"] for f in r["files"] if f["cr_resid_deg"] is not None]
    assert len(resids) == 24 and max(abs(x) for x in resids) <= 10.0
    assert "vintage_ok=False" in r["version_drift"] or "vintage" in r["version_drift"]
    assert "<7MB" in r["transfer_budget"]


def test_receipt_matches_and_hashes():
    assert RECEIPT.exists(), "run collect.collect() before testing"
    r = json.loads(RECEIPT.read_text())
    assert r["direction"] == "D138" and r["owner"] == "worker-t2p"
    assert r["anchor"]["utc"] == SP.L0_ANCHOR_UTC
    for f in r["files"]:
        assert len(f["body_sha256"]) == 64 and f["content_length_ok"] is True
        assert f["gunzip_simple_ok"] is True and f["header_ok"] is True
        assert f["crval1"] == f["crval1"]  # WCS present, no NaN


def test_pfss_monopole_and_profiles():
    uni = np.full((180, 360), 3.0)
    out = PF.pfss_brss(uni, 8)
    assert abs(out.mean() - 3.0 * (1.0 / SP.RSS) ** 2) < 1e-6
    assert np.isfinite(out).all()
    d = PF.neutral_dist_deg(np.full((180, 360), 1.0))
    assert np.isfinite(d).all() and (d >= 0).all()
    assert list(PF.band_rows()) == list(range(67, 113))


def test_features_causal_newest_wins_and_stale():
    maps = _synth_maps()
    o = pd.DatetimeIndex(["2025-11-01T00:00:00Z", "2025-12-01T13:14:00Z",
                          "2026-01-01T13:04:00Z", "2026-05-01T00:00:00Z"])
    df = pd.DataFrame({"origin": np.repeat(o, 2), "h": np.tile([1.0, 72.0], 4),
                       "v_lag0": 400.0})
    feat = FE.add_physics(df, maps)
    assert list(feat.columns) == SP.FEATURES == FE.SP.FEATURES
    f = feat.iloc[::2].reset_index(drop=True)
    assert f.loc[0, "flux_stale"] == 1.0 and f.loc[0, "flux_age_h"] == -1.0
    assert f.loc[0, "wsa_v"] == SP.FALLBACK["wsa_v"]
    assert f.loc[1, "flux_stale"] == 0.0 and f.loc[1, "brfoot_abs"] == 10.0
    assert f.loc[2, "brfoot_abs"] == 14.0 and f.loc[3, "brfoot_abs"] == 14.0
    assert f.loc[3, "flux_age_h"] > f.loc[2, "flux_age_h"]
    assert (feat["flux_vintage_ok"] == 0.0).all()
    assert np.isfinite(feat.to_numpy(dtype=float)).all()


def test_wsa_formula_and_anchor():
    v = FE.wsa_speed(np.array([1.0]), np.array([30.0]))[0]
    assert abs(v - 677.5) < 0.01
    v2 = FE.wsa_speed(np.array([300.0]), np.array([30.0]))[0]
    assert 240.0 < v2 < 350.0
    l0 = FE.anchor_l0(pd.DatetimeIndex([pd.Timestamp(SP.L0_ANCHOR_UTC, tz="UTC")]))[0]
    assert abs(l0 - 275.91) < 0.05


def test_poison_real_with_positive_control():
    maps = _synth_maps()
    origins = pd.date_range("2025-11-01", "2026-05-01", freq="24h", tz="UTC")
    pz = FE.poison_test(origins, maps[-1]["date_obs"].isoformat(), maps)
    assert pz["passed"] is True and pz["pre_invariant"] is True
    assert set(pz) >= {"passed", "cutoff_utc", "n_pre_rows", "n_post_rows", "moved_columns"}
    assert len(pz["moved_columns"]) > 0 and pz["n_pre_rows"] > 0 and pz["n_post_rows"] > 0


def test_owned_files_capped_and_scratch_separated():
    files = sorted(f.name for f in PKG.iterdir() if f.is_file())
    py = [f for f in files if f.endswith(".py")]
    assert len(files) <= 6, files
    assert {"spec.py", "collect.py", "pfss.py", "phys_features.py", "rocv_runner.py"} <= set(py)
    blob = "\n".join((PKG / f).read_text(errors="ignore") for f in files if f.endswith(".py"))
    assert "worker-t2p" in blob
    for other in OTHER_WORKERS:
        assert other not in blob, f"foreign id {other}"
    assert "store/scratch/worker-t2p" in blob
    assert "store/scratch/worker-t2" not in blob.replace("store/scratch/worker-t2p", "")


def test_py_has_no_forbidden_execution():
    for f in PKG.glob("*.py"):
        text = f.read_text()
        for s in PY_FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        if f.name not in NET_OK_ONLY_IN:
            assert "urlopen" not in text and "Request(" not in text, f"{f.name} must stay offline"
        if f.name not in EXPOS_IMPORT_OK_ONLY_IN:
            assert "from expos import" not in text, f"{f.name} must not import expos"


def test_seed_propagated_and_budget():
    runner = (PKG / "rocv_runner.py").read_text()
    assert "SEEDS_ROCV" in runner and "random_state" in runner
    assert "FIT_NJOBS" in runner and "device" in runner and "cuda" in runner
    assert "gpu_lease" not in runner
    assert "EARLY_STOP_PATIENCE" in runner and "eval_set" in runner
    assert "target_t" in runner and "quantile(0.9)" in runner


def test_registration_is_single_full_vs_e0206():
    recs = [json.loads(l) for l in GRAPH.read_text().splitlines() if l.strip()]
    mine = [r for r in recs if r.get("event") == "register" and r.get("owner") == "worker-t2p"]
    assert len(mine) <= 1, "one full registration for worker-t2p"
    if mine:
        assert mine[0]["kind"] == "full" and mine[0]["reference"] == "E0206"
        assert mine[0]["direction"] == "D138"


def test_report_carries_rocv_evidence():
    assert REPORT.exists(), "write physics/report.md after rocv-eval before testing"
    rep = REPORT.read_text()
    for token in ["PASS-timing-only", "wsa_v", "theta_b", "vintage_ok", "poison", "E0206",
                  "worker-t2p", "X0005", "rocv", "F1", "MSE", "mean+2SE", "PFSS", "WSA",
                  "flux_stale", "f0", "f7", "seeds"]:
        assert token in rep, f"report missing {token!r}"
