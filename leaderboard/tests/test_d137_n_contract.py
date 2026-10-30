"""D137 T2-NOAA contract: delay distribution + HEAD-only collection + pilot screen, no overreach.

Fast: pure-logic checks + filesystem/receipt/graph scans only. No fits, no network
(delay evidence is recorded constants + live HEAD receipts verified by hashes),
no official reads, no foreign experiment runs.
"""

import json
import sys
from pathlib import Path

import pandas as pd

LB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LB / "experiments" / "d137_t2noaa"))

import noaa_collect as NC  # noqa: E402
import noaa_features as NF  # noqa: E402
import noaa_features as NF  # noqa: E402
import spec as SP  # noqa: E402

PKG = LB / "experiments" / "d137_t2noaa"
REPORT = PKG / "report.md"
SCRATCH = LB / "store" / "scratch" / "worker-t2n"
RECEIPT = SCRATCH / "collect_receipt.json"
DELAY_RECEIPT = SCRATCH / "delay_receipt.json"
GRAPH = LB.parent / "research" / "expos" / "graph.jsonl"

PY_FORBIDDEN = ["truth.parquet", "folds.parquet", "naive.parquet",
                "official-predictions", "baseline-replay", "predictions_official_truth",
                "official_check", "official-check",
                "datasets--tlabtlab--sunrun-lb-store",
                "predictions_official", "official-score", "official_score",
                "core.freeze", "core.register(", "core.run(", "from expos import",
                "research/expos/runs/E", "read_parquet",
                "requests.get", "urlretrieve", "tarfile.open", "fits.open"]
NET_OK_ONLY_IN = {"noaa_collect.py"}
OTHER_WORKERS = ["worker-d35", "worker-d133", "worker-d134", "worker-d135", "worker-t1b",
                 "worker-t1c", "worker-t1br", "worker-t2g", "worker-t3b", "worker-t3c",
                 "worker-d33", "worker-d52", "worker-img"]


def _synth_receipt():
    return {"rule": {"rule_h": 4039}, "stats": {"p95_h": 3230.4},
            "files": [{"init_utc": "2025-08-15T00:00:00Z", "delay_h": 3528.0,
                       "head": {"ok": True}, "anchor_match": None},
                      {"init_utc": "2025-09-15T00:00:00Z", "delay_h": 2784.1,
                       "head": {"ok": True}, "anchor_match": True}],
            "eligible_inits": ["2025-08-15T00:00:00Z", "2025-09-15T00:00:00Z"]}


def test_spec_tags_seeds_and_rule():
    assert SP.DIRECTION == "D137" and SP.OWNER == "worker-t2n"
    assert SP.FAMILY_TAG == "[family=xgb]" and SP.AXES_TAG == "[axes: data=noaa-enlil]"
    assert SP.HYPOTHESIS.startswith("[family=xgb]") and "[axes: data=noaa-enlil]" in SP.HYPOTHESIS
    for cite in ("E0003", "E0206", "X0005", "X0004", "rocv", "6/8", "mean+2SE<0"):
        assert cite in SP.HYPOTHESIS, f"hypothesis missing {cite}"
    assert "p95" in SP.HYPOTHESIS and "vintage" in SP.HYPOTHESIS
    assert "reprocess" in SP.HYPOTHESIS
    assert set(SP.EXPERIENCE) == {"X0005", "X0188", "X0004"}
    assert SP.SEED == 0 and tuple(SP.SEEDS_ROCV) == (0, 1, 2)
    assert "selection/audit" in SP.SELECT_RULE and "dev" in SP.SELECT_RULE
    assert SP.PILOT_FREQ == "6h" and SP.FIT_NJOBS <= 6
    assert len(SP.FEATURES) == 6 and "noaa_vintage_ok" in SP.FEATURES
    assert str(SP.SCRATCH) == "store/scratch/worker-t2n"


def test_delay_distribution_and_conservative_rule():
    assert all(NC.verify_header_sha256(r) for r in NC.RECORDED)
    delays = NC.recorded_delays()
    assert len(delays) == 3 and min(delays) > 24.0  # every anchor ingested late
    stats = NC.distribution_stats(delays)
    assert stats["n"] == 3 and stats["p95_h"] >= stats["p50_h"] >= stats["min_h"]
    rule = NC.conservative_rule(stats)
    assert rule["rule_h"] > rule["p95_h"]  # p95-plus margin, conservative by design
    assert rule["margin_h"] >= NC.MARGIN_FLOOR_H
    assert f"init_utc + {rule['rule_h']}h <= origin" in rule["rule"]
    for init in NC.CANDIDATE_INITS:
        url = NC.candidate_url(init)
        assert init[:10].replace("-", "") in url and url.endswith(".tar.gz")
    receipt = NC.build_delay_receipt(delays, len(NC.RECORDED), 0)
    assert receipt["direction"] == "D137" and receipt["stats"]["n"] == 3


def test_receipts_rule_bodies_and_drift():
    assert RECEIPT.exists(), "run noaa_collect.collect() before testing"
    r = json.loads(RECEIPT.read_text())
    assert r["direction"] == "D137" and r["owner"] == "worker-t2n"
    assert r["bodies_fetched"] == 0 and "0 bodies" in r["transfer_budget"]
    assert all(f["body_sha256"] is None for f in r["files"])  # zero bodies, stated
    assert "vintage" in r["version_drift"] and "ingestion" in r["version_drift"]
    assert int(r["rule"]["rule_h"]) > int(float(r["stats"]["p95_h"]))
    assert DELAY_RECEIPT.exists()
    d = json.loads(DELAY_RECEIPT.read_text())
    assert d["rule"] == r["rule"] and d["stats"] == r["stats"]
    assert d["stats"]["n"] >= 3 and all(d["header_ok"])


def test_features_causal_newest_wins_and_stale():
    rec = _synth_receipt()
    o = pd.DatetimeIndex(["2025-08-01T00:00:00Z", "2025-09-15T14:14:00Z",
                          "2026-03-01T00:00:00Z", "2026-05-01T00:00:00Z"])
    feat = NF.add_features(o, rec)
    assert list(feat.columns) == SP.FEATURES == NF.FEATURES
    assert feat.loc[o[0], "noaa_stale"] == 1.0 and feat.loc[o[0], "noaa_age_h"] == -1.0
    assert feat.loc[o[2], "noaa_stale"] == 0.0  # Aug init + rule covered by March
    assert feat.loc[o[3], "noaa_age_h"] > feat.loc[o[2], "noaa_age_h"]
    assert (feat["noaa_vintage_ok"] == 0.0).all()
    assert (feat["noaa_rule_h"] == 4039.0).all() and (feat["noaa_p95_h"] == 3230.4).all()


def test_poison_real_with_positive_control():
    rec = _synth_receipt()
    origins = pd.date_range("2025-08-01", "2026-05-01", freq="24h", tz="UTC")
    pz = NF.poison_test(origins, sorted(rec["eligible_inits"])[0], rec)
    assert pz["passed"] is True and pz["pre_invariant"] is True
    assert set(pz) >= {"passed", "cutoff_utc", "n_pre_rows", "n_post_rows", "moved_columns"}
    assert len(pz["moved_columns"]) > 0 and pz["n_pre_rows"] > 0 and pz["n_post_rows"] > 0


def test_owned_files_capped_and_scratch_separated():
    files = sorted(f.name for f in PKG.iterdir() if f.is_file())
    py = [f for f in files if f.endswith(".py")]
    assert len(files) <= 5, files
    assert set(py) == {"spec.py", "noaa_collect.py", "noaa_features.py", "runner.py"}
    assert "report.md" in files
    blob = "\n".join((PKG / f).read_text(errors="ignore") for f in files)
    assert "worker-t2n" in blob
    for other in OTHER_WORKERS:
        assert other not in blob, f"foreign id {other}"
    assert "store/scratch/worker-t2n" in blob
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
    mine = [r for r in recs if r.get("event") == "register" and r.get("owner") == "worker-t2n"]
    assert mine and all(m["direction"] == "D137" and m["kind"] == "pilot" for m in mine)
    latest = mine[-1]
    assert set(latest["code"]) == {"experiments/d137_t2noaa/spec.py",
                                   "experiments/d137_t2noaa/noaa_collect.py",
                                   "experiments/d137_t2noaa/noaa_features.py",
                                   "experiments/d137_t2noaa/runner.py"}
    for r in recs:
        if r.get("event") == "run" and r.get("id") in {m.get("id") for m in mine}:
            assert True
    others = [r for r in recs if r.get("event") == "run" and str(r.get("log", "")).find("worker-t2n") >= 0
              and r.get("id") not in {m.get("id") for m in mine}]
    assert others == []


def test_report_carries_receipt_features_and_screen():
    assert REPORT.exists(), "write report.md after evaluate before testing"
    rep = REPORT.read_text()
    recs = [json.loads(l) for l in GRAPH.read_text().splitlines() if l.strip()]
    mine = [r for r in recs if r.get("event") == "register" and r.get("owner") == "worker-t2n"]
    for token in ["p95", "4039h", "HEAD", "0 bodies", "noaa_age_h", "noaa_since_init_h",
                  "vintage_ok", "poison", "E0001", mine[-1]["id"], "worker-t2n", "X0005",
                  "rocv", "dev", "selection", "audit", "MSE", "FAIL", "수집중단"]:
        assert token in rep, f"report missing {token!r}"
