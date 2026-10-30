"""D138 T2b-flux contract: collect receipt + causal PFSS-free features + pilot screen, no overreach.

Fast: pure-logic checks + filesystem/receipt/graph scans only. No fits, no network
(collection evidence is recorded constants + cached bytes verified by hashes),
no forbidden reads, no foreign experiment runs.
"""

import json
import sys
from pathlib import Path

import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d138_t2bflux"))

import flux_collect as FC  # noqa: E402
import flux_features as FF  # noqa: E402
import spec as SP  # noqa: E402

PKG = LB / "experiments" / "d138_t2bflux"
REPORT = PKG / "report.md"
SCRATCH = LB / "store" / "scratch" / "worker-t2b"
RECEIPT = SCRATCH / "collect_receipt.json"
GRAPH = LB.parent / "research" / "expos" / "graph.jsonl"

PY_FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
                "official-predictions", "baseline-replay", "predictions_official_truth",
                "official_check", "official-check",
                "datasets--tlabtlab--sunrun-lb-store",
                "predictions_official", "official-score", "official_score",
                "core.freeze", "core.register(", "core.run(", "from expos import",
                "research/expos/runs/E", "read_parquet",
                "requests.get", "urlretrieve", "tarfile.open", "fits.open"]
NET_OK_ONLY_IN = {"flux_collect.py"}
OTHER_WORKERS = ["worker-d35", "worker-d134", "worker-d135", "worker-t1b", "worker-t1c",
                 "worker-t3b", "worker-t2g", "worker-t2n", "worker-d133", "worker-d33",
                 "worker-d52", "worker-img"]


def _synth_files():
    return [
        {"period": "a", "path": "synth-a", "date_obs": pd.Timestamp("2025-09-15T12:14:00Z"),
         "body_sha256": "0" * 64, "unsigned": 7.0, "signed": 0.05, "actfrac": 0.02},
        {"period": "b", "path": "synth-b", "date_obs": pd.Timestamp("2026-01-15T11:14:00Z"),
         "body_sha256": "1" * 64, "unsigned": 6.0, "signed": -0.10, "actfrac": 0.015},
    ]


def test_spec_tags_seeds_and_rule():
    assert SP.DIRECTION == "D138" and SP.OWNER == "worker-t2b"
    assert SP.FAMILY_TAG == "[family=xgb]" and SP.AXES_TAG == "[axes: data=gong-flux]"
    assert SP.HYPOTHESIS.startswith("[family=xgb]") and "[axes: data=gong-flux]" in SP.HYPOTHESIS
    for cite in ("E0003", "E0206", "X0005", "X0004", "rocv", "6/8", "mean+2SE<0"):
        assert cite in SP.HYPOTHESIS, f"hypothesis missing {cite}"
    assert "vintage" in SP.HYPOTHESIS and "reprocess" in SP.HYPOTHESIS
    assert "PFSS" in SP.HYPOTHESIS
    assert set(SP.EXPERIENCE) == {"X0005", "X0188", "X0004"}
    assert SP.SEED == 0 and tuple(SP.SEEDS_ROCV) == (0, 1, 2)
    assert "selection/audit" in SP.SELECT_RULE and "dev" in SP.SELECT_RULE
    assert SP.PILOT_FREQ == "6h" and SP.FIT_NJOBS <= 6
    assert len(SP.FEATURES) == 6 and "flux_actfrac" in SP.FEATURES
    assert "wsa" not in " ".join(SP.FEATURES) and "dipole" not in " ".join(SP.FEATURES)
    assert str(SP.SCRATCH) == "store/scratch/worker-t2b"


def test_proxy_verdicts_and_header_hashes():
    assert all(FC.verify_header_sha256(s) for s in FC.SAMPLES)
    verdicts = [(p["period"], p["proxy_obs_plus_2h"], p["usable_for_features"])
                for p in (FC.proxy_row(s) for s in FC.SAMPLES)]
    assert verdicts == [("dev", "PASS-timing-only", True),
                        ("selection", "PASS-timing-only", True),
                        ("audit", "FAIL", False)]


def test_receipt_matches_and_hashes():
    assert RECEIPT.exists(), "run flux_collect.collect() before testing"
    r = json.loads(RECEIPT.read_text())
    assert r["direction"] == "D138" and r["owner"] == "worker-t2b"
    assert [p["proxy_obs_plus_2h"] for p in r["proxy"]] == ["PASS-timing-only", "PASS-timing-only", "FAIL"]
    assert r["proxy"][2]["usable_for_features"] is False  # reprocessed audit excluded
    assert "reprocess" in r["version_drift"] or "vintage" in r["version_drift"]
    assert "PFSS" in r["version_drift"]
    assert "<1MB" in r["transfer_budget"]
    for f in r["files"]:
        assert len(f["body_sha256"]) == 64 and f["content_length_ok"] is True
        assert f["gunzip_simple_ok"] is True


def test_features_causal_newest_wins_and_stale():
    files = _synth_files()
    o = pd.DatetimeIndex(["2025-08-01T00:00:00Z", "2025-09-15T14:14:00Z",
                          "2026-03-01T00:00:00Z", "2026-05-01T00:00:00Z"])
    feat = FF.add_features(o, files)
    assert list(feat.columns) == SP.FEATURES == FF.FEATURES
    assert feat.loc[o[0], "flux_stale"] == 1.0 and feat.loc[o[0], "flux_age_h"] == -1.0
    assert feat.loc[o[1], "flux_unsigned"] == 7.0 and feat.loc[o[1], "flux_stale"] == 0.0
    assert feat.loc[o[1], "flux_signed"] == 0.05 and feat.loc[o[1], "flux_actfrac"] == 0.02
    # newest eligible wins; late origin reuses the older file (large age, no future leak)
    assert feat.loc[o[2], "flux_unsigned"] == 6.0 and feat.loc[o[3], "flux_unsigned"] == 6.0
    assert feat.loc[o[3], "flux_age_h"] > feat.loc[o[2], "flux_age_h"]
    assert (feat["flux_vintage_ok"] == 0.0).all()


def test_poison_real_with_positive_control():
    files = _synth_files()
    origins = pd.date_range("2025-08-01", "2026-05-01", freq="24h", tz="UTC")
    pz = FF.poison_test(origins, files[-1]["date_obs"].isoformat(), files)
    assert pz["passed"] is True and pz["pre_invariant"] is True
    assert set(pz) >= {"passed", "cutoff_utc", "n_pre_rows", "n_post_rows", "moved_columns"}
    assert len(pz["moved_columns"]) > 0 and pz["n_pre_rows"] > 0 and pz["n_post_rows"] > 0


def test_owned_files_capped_and_scratch_separated():
    files = sorted(f.name for f in PKG.iterdir() if f.is_file())
    py = [f for f in files if f.endswith(".py")]
    assert len(files) <= 5, files
    assert {"spec.py", "flux_collect.py", "flux_features.py", "runner.py"} <= set(py)
    blob = "\n".join((PKG / f).read_text(errors="ignore") for f in files)
    assert "worker-t2b" in blob
    for other in OTHER_WORKERS:
        assert other not in blob, f"foreign id {other}"
    assert "store/scratch/worker-t2b" in blob
    assert "store/scratch/worker-d" not in blob


def test_py_has_no_forbidden_execution():
    for f in PKG.glob("*.py"):
        text = f.read_text()
        for s in PY_FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        if f.name not in NET_OK_ONLY_IN:
            assert "urlopen" not in text and "Request(" not in text, f"{f.name} must stay offline"


def test_seed_propagated_and_budget():
    runner = (PKG / "runner.py").read_text()
    assert "SP.SEED" in runner and "random_state" in runner
    assert "FIT_NJOBS" in runner
    assert "cuda" not in runner and "gpu_lease" not in runner  # CPU-only pilot, no GPU touched


def test_no_foreign_runs_in_graph():
    recs = [json.loads(l) for l in GRAPH.read_text().splitlines() if l.strip()]
    mine = [r for r in recs if r.get("event") == "register" and r.get("owner") == "worker-t2b"]
    assert len(mine) <= 1, "one pilot registration for worker-t2b"
    others = [r for r in recs if r.get("event") == "run" and str(r.get("log", "")).find("worker-t2b") >= 0
              and r.get("id") not in {m.get("id") for m in mine}]
    assert others == []


def test_report_carries_receipt_features_and_screen():
    assert REPORT.exists(), "write report.md after evaluate before testing"
    rep = REPORT.read_text()
    for token in ["PASS-timing-only", "FAIL", "flux_unsigned", "flux_signed", "flux_actfrac",
                  "vintage_ok", "poison", "E0001", "worker-t2b", "X0005", "rocv",
                  "dev", "selection", "audit", "MSE"]:
        assert token in rep, f"report missing {token!r}"
