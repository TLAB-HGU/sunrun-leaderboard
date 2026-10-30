"""D148 Step 0 collection contract: DONKI CME timing raw store (fast, no network)."""
import hashlib
import io
import json
import re
import tokenize
from pathlib import Path

LB = Path(__file__).resolve().parents[1]
EXP = LB / "experiments/d148_cme_timing"
STORE = LB / "store/donki-cme"
CODE = EXP / "collect_donki.py"
ENDPOINTS = ("CMEAnalysis", "CME", "IPS", "HSS")
MONTHS = ["%04d-%02d" % (y, m) for y in range(2022, 2027) for m in range(1, 13)
          if "2022-10" <= "%04d-%02d" % (y, m) <= "2026-10"]


def _code_logic():
    """Source with comments and docstrings removed (behavioral tokens only)."""
    toks = tokenize.generate_tokens(io.StringIO(CODE.read_text()).readline)
    return " ".join(t.string for t in toks
                    if t.type not in (tokenize.COMMENT,) and not (
                        t.type == tokenize.STRING and "\n" in t.string))


def test_owned_files_present():
    assert CODE.is_file()
    files = [p.name for p in STORE.iterdir()]
    for mo in MONTHS:
        for ep in ENDPOINTS:
            assert "%s_%s.json" % (ep, mo) in files, (ep, mo)
    for extra in ("manifest.json", "SHA256SUMS", "fetched_at.txt",
                  "VINTAGE_RULE.md", "REPORT.md",
                  "hdr_CMEAnalysis_2026-10.txt", "hdr_CME_2026-10.txt",
                  "hdr_IPS_2026-10.txt", "hdr_HSS_2026-10.txt"):
        assert extra in files, extra


def test_collector_scope_and_hygiene():
    logic = _code_logic()
    raw = CODE.read_text()
    assert "ENDPOINTS = " in raw and set(ENDPOINTS) == {"CMEAnalysis", "CME", "IPS", "HSS"}
    assert "WSAEnlilSimulations_" not in raw and "get/WSAEnlil" not in raw
    assert "donki-live" in raw
    assert "isMostAccurate" not in logic
    assert "headroom-audit" not in logic
    assert "GONG" not in logic and "ADAPT" not in logic
    assert "gpu" not in logic.lower() and "cuda" not in logic.lower()
    assert "scorer" not in logic and "expos" not in logic
    for token in ("e0147", "d141", "sun-nn", "nn_", "nn-", "D141", "E0471", "E0472"):
        assert token not in logic, token
    import os
    norm = raw.replace(os.sep, "/")
    assert "store/donki-cme" in norm or '"store", "donki-cme"' in raw
    assert '"scratch", "worker-t1e0"' in raw
    assert len(set(re.findall(r"worker-[a-z0-9]+", norm))) == 1


def test_manifest_hashes_and_counts():
    man = json.loads((STORE / "manifest.json").read_text())
    assert len(man["files"]) == 196 and man["requests"] == 196
    assert man["start"] == "2022-10" and man["end"] == "2026-10"
    assert "donki-live" in man["wsa_enlil_note"]
    seen = set()
    for e in man["files"]:
        seen.add((e["endpoint"], e["month"]))
        body = (STORE / ("%s_%s.json" % (e["endpoint"], e["month"]))).read_bytes()
        assert hashlib.sha256(body).hexdigest() == e["sha256"]
        assert len(json.loads(body)) == e["n_entries"]
    assert seen == {(ep, mo) for mo in MONTHS for ep in ENDPOINTS}
    parts = [ln.split() for ln in (STORE / "SHA256SUMS").read_text().splitlines()]
    sums = {p[1]: p[0] for p in parts}
    for e in man["files"]:
        assert sums["%s_%s.json" % (e["endpoint"], e["month"])] == e["sha256"]


def test_no_blank_months():
    man = json.loads((STORE / "manifest.json").read_text())
    zeros = [(e["endpoint"], e["month"]) for e in man["files"] if e["n_entries"] == 0]
    assert zeros == [], zeros


def test_october_2026_reachability():
    for ep in ENDPOINTS:
        rows = json.loads((STORE / ("%s_2026-10.json" % ep)).read_text())
        assert len(rows) > 0, ep
    analyses = json.loads((STORE / "CMEAnalysis_2026-10.json").read_text())
    subs = [a["submissionTime"] for a in analyses]
    assert len(subs) == len(analyses) and all(s.startswith("2026-10") for s in subs)
    t21 = [a["time21_5"] for a in analyses]
    assert all(t.startswith("2026-10") for t in t21)


def test_raw_analyses_unfiltered():
    multi = False
    non_flag = False
    for p in STORE.glob("CME_*.json"):
        for e in json.loads(p.read_text()):
            ans = e.get("cmeAnalyses") or []
            if len(ans) > 1:
                multi = True
            if any(a.get("isMostAccurate") is not True for a in ans):
                non_flag = True
    assert multi and non_flag


def test_vintage_rule_documented():
    doc = (STORE / "VINTAGE_RULE.md").read_text()
    flat = re.sub(r"\s+", "", doc)
    assert "submissionTime<=O" in flat or "submissionTime<=origin" in flat
    assert "isMostAccurate" in doc and "versionId" in doc
    assert "modelCompletionTime<=O" in flat or "modelCompletionTime<=origin" in flat
    assert "donki-live" in doc and "run.py" in doc
    assert "CDAW" in doc


def test_disk_budget():
    total = sum(p.stat().st_size for p in STORE.iterdir() if p.is_file())
    assert total < 50 * 1024 ** 3, total
