"""D38 cycle contract: clean abort — no pinned F10.7 source, no registration.

Fast: filesystem existence checks and graph scan only; no data reads, no fits."""

import json
import os
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d38_cycle"))

import source_audit as SA  # noqa: E402

PKG = LB / "experiments" / "d38_cycle"
MEMO = PKG / "approval_gate_memo.md"
RESEARCH = Path(os.environ.get("EXPOS_RESEARCH", LB.parent / "research"))
GRAPH = RESEARCH / "expos" / "graph.jsonl"
RECEIPT = LB / "store" / "scratch" / "worker-d38" / "source_audit.json"

# Executable danger patterns: must never appear in owned code (.py only;
# the memo (.md) may name rejected sources to document their exclusion).
PY_FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
                "official-predictions", "baseline-replay", "predictions_official_truth",
                "official_check", "official-check",
                "datasets--tlabtlab--sunrun-lb-store",
                "read_parquet", "requests.get", "urlopen", "urlretrieve",
                "noaa-goes", "gong.nso.edu", "kp.gfz.de",
                "official-score", "official_score", "freeze"]

OTHER_WORKERS = ["worker-d35", "worker-d36", "worker-d33", "worker-d34",
                 "worker-d31", "worker-d32", "worker-d21", "worker-d22",
                 "worker-d23", "worker-d25", "worker-d28", "worker-d30"]


def _graph_records():
    return [json.loads(l) for l in GRAPH.read_text().splitlines() if l.strip()]


def test_audit_verdict_absent_and_cites_x0002():
    r = SA.audit()
    assert r["verdict"] == "ABSENT"
    assert r["f107_files"] == []
    assert r["experience"] == ["X0002"]
    assert r["d38_registered_ids"] == []
    assert r["registration"] == "none"
    assert "X0002" in (PKG / "source_audit.py").read_text()


def test_owned_files_capped_and_scratch_separated():
    files = sorted(f.name for f in PKG.glob("*.py"))
    assert len(files) <= 4, files
    assert "source_audit.py" in files
    assert MEMO.exists()
    blob = "\n".join((PKG / f).read_text() for f in files) + "\n" + MEMO.read_text()
    assert "worker-d38" in blob
    for other in OTHER_WORKERS:
        assert other not in blob
    assert "store/scratch/worker-d38" in (PKG / "source_audit.py").read_text()


def test_py_has_no_forbidden_execution():
    for f in PKG.glob("*.py"):
        text = f.read_text()
        for s in PY_FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"


def test_no_d38_registration_in_graph():
    recs = _graph_records()
    assert any(r.get("event") == "direction" and r.get("id") == "D38" for r in recs)
    bad = [r for r in recs
           if r.get("event") == "register" and r.get("direction") == "D38"]
    assert bad == []


def test_scratch_receipt_matches_audit():
    assert RECEIPT.exists(), "run source_audit.py before testing"
    assert json.loads(RECEIPT.read_text()) == SA.audit()


def test_memo_prespecifies_blocked_pilot_and_gate():
    text = MEMO.read_text()
    for token in ["X0002", "E0001", "E0003", "6h", "poison", "0.4112", "승인",
                  "깨끗중단", "origin", "worker-d38"]:
        assert token in text, f"memo missing {token!r}"
