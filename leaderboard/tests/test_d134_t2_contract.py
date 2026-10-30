"""D134 T2 scout contract: recon report + proof plan, no collection/training/registration.

Fast: stdlib audit + filesystem checks + graph scan only; no data reads, no fits,
no network (evidence is recorded constants verified by header hashes)."""

import json
import os
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d134_t2scout"))

import scout as SC  # noqa: E402

PKG = LB / "experiments" / "d134_t2scout"
RECON = PKG / "recon_report.md"
PLAN = PKG / "proof_plan.md"
RESEARCH = Path(os.environ.get("EXPOS_RESEARCH", LB.parent / "research"))
GRAPH = RESEARCH / "expos" / "graph.jsonl"
RECEIPT = LB / "store" / "scratch" / "worker-d134" / "recon.json"

# Executable danger patterns: must never appear in owned code (.py only;
# the reports (.md) may name rejected sources to document their exclusion).
PY_FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
                "official-predictions", "baseline-replay", "predictions_official_truth",
                "official_check", "official-check",
                "datasets--tlabtlab--sunrun-lb-store",
                "read_parquet", "requests.get", "urlopen", "urlretrieve",
                "tarfile.open", "fits.open", "open_fits",
                "official-score", "official_score", "core.freeze", "core.run(",
                "core.register(", "from expos import",
                "research/expos/runs/E", "predictions_dev.parquet"]

OTHER_WORKERS = ["worker-d35", "worker-d36", "worker-d33", "worker-d34",
                 "worker-d31", "worker-d32", "worker-d21", "worker-d22",
                 "worker-d23", "worker-d25", "worker-d28", "worker-d30",
                 "worker-d38", "worker-d124", "worker-d126", "worker-d127"]


def _graph_records():
    return [json.loads(l) for l in GRAPH.read_text().splitlines() if l.strip()]


def test_audit_verdict_and_proxy_tables():
    r = SC.audit()
    assert r["direction"] == "D134" and r["owner"] == "worker-d134"
    assert r["verdict"] == "NON_ADMITTED_UNVERIFIED"
    assert r["full_collection_allowed"] is False
    assert r["ablation_fits"] == 0 and r["registration"] == "none"
    assert r["d134_registered_ids"] == []
    assert set(r["experience"]) == {"X0004", "X0005"}
    assert len(r["noaa_proxy"]) == 3 and len(r["gong_proxy"]) == 3
    assert all(row["proxy_init_plus_12h"] == "FAIL" for row in r["noaa_proxy"])
    assert [row["proxy_obs_plus_2h"] for row in r["gong_proxy"]] == [
        "PASS-timing-only", "PASS-timing-only", "FAIL"]
    assert all(SC.verify_header_sha256(s) for s in SC.NOAA_SAMPLES)
    assert all(SC.verify_header_sha256(s) for s in SC.GONG_SAMPLES)
    assert [d["n_files"] for d in r["gong_hourly_cadence"]] == [22, 24, 24]
    assert "0 file bodies" in r["transfer_budget"]


def test_owned_files_capped_and_scratch_separated():
    files = sorted(f.name for f in PKG.glob("*.py"))
    assert len(files) <= 4, files
    assert "scout.py" in files
    assert RECON.exists() and PLAN.exists()
    assert len([f for f in PKG.iterdir() if f.is_file()]) <= 4
    blob = "\n".join((PKG / f).read_text() for f in files)
    blob += "\n" + RECON.read_text() + "\n" + PLAN.read_text()
    assert "worker-d134" in blob
    for other in OTHER_WORKERS:
        assert other not in blob
    assert "store/scratch/worker-d134" in (PKG / "scout.py").read_text()
    assert "X0005" in blob


def test_py_has_no_forbidden_execution():
    for f in PKG.glob("*.py"):
        text = f.read_text()
        for s in PY_FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"


def test_no_d134_registration_or_foreign_runs_in_graph():
    recs = _graph_records()
    assert any(r.get("event") == "direction" and r.get("id") == "D134" for r in recs)
    bad = [r for r in recs
           if r.get("event") == "register" and r.get("direction") == "D134"]
    assert bad == []
    mine = [r for r in recs
            if r.get("event") == "run" and r.get("gpu") is not None
            and "worker-d134" in json.dumps(r)]
    assert mine == []


def test_scratch_receipt_matches_audit():
    assert RECEIPT.exists(), "run scout.py before testing"
    assert json.loads(RECEIPT.read_text()) == SC.audit()


def test_reports_carry_paths_hashes_and_gated_plan():
    recon = RECON.read_text()
    for token in ["archive.data.noaa.gov", "gong.nso.edu", "FAIL",
                  "header_sha256", "NON_ADMITTED_UNVERIFIED",
                  "0바이트", "worker-d134"]:
        assert token in recon, f"recon missing {token!r}"
    plan = PLAN.read_text()
    for token in ["E0003", "E0206", "rocv", "6/8", "-2%", "-1%", "mean+2SE<0",
                  "F1", "poison", "72h", "3h", "승인", "미등록", "venv"]:
        assert token in plan, f"plan missing {token!r}"
