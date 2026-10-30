"""D137 retry (worker-t2n2) contract: operational init+12h rule + E0003+NOAA rocv.

Fast pure-logic/filesystem/receipt/graph checks plus recorded rocv-evidence scans.
No fits, no network, no official reads, no foreign experiment runs.
"""

import json
from pathlib import Path

import pandas as pd

LB = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(LB / "experiments" / "d137_t2noaa" / "retry"))

import noaa_collect_op as NC  # noqa: E402
import noaa_features_op as NF  # noqa: E402
import spec as SP  # noqa: E402

PKG = LB / "experiments" / "d137_t2noaa" / "retry"
REPORT = PKG / "report.md"
SCRATCH = LB / "store" / "scratch" / "worker-t2n2"
RECEIPT = SCRATCH / "collect_receipt.json"
GRAPH = LB.parent / "research" / "expos" / "graph.jsonl"

PY_FORBIDDEN = ["truth.parquet", "naive.parquet",
                "official-predictions", "baseline-replay", "predictions_official_truth",
                "official_check", "official-check",
                "datasets--tlabtlab--sunrun-lb-store",
                "predictions_official", "official-score", "official_score",
                "core.freeze", "core.register(", "core.run(",
                "research/expos/runs/E", "read_parquet",
                "requests.get", "urlretrieve", "tarfile.open", "fits.open"]
NET_OK_ONLY_IN = {"noaa_collect_op.py"}
OTHER_SCRATCH = ["worker-t2b", "worker-t3b", "worker-t3c", "worker-t3d", "worker-t3e",
                 "worker-t3f", "worker-t1b", "worker-t1c", "worker-t2g"]


def _synth_receipt():
    return {"rule": {"rule_h": 12}, "stats": {},
            "files": [{"init_utc": "2024-06-15T00:00:00Z", "delay_h": 3528.0,
                       "head": {"ok": True}, "eligible": True},
                      {"init_utc": "2024-07-15T00:00:00Z", "delay_h": 2784.1,
                       "head": {"ok": True}, "eligible": True}],
            "eligible_inits": ["2024-06-15T00:00:00Z", "2024-07-15T00:00:00Z"]}


def _mine():
    recs = [json.loads(l) for l in GRAPH.read_text().splitlines() if l.strip()]
    return recs, [r for r in recs if r.get("event") == "register" and r.get("owner") == "worker-t2n2"]


def test_spec_tags_seeds_and_operational_rule():
    assert SP.DIRECTION == "D137" and SP.OWNER == "worker-t2n2"
    assert SP.FAMILY_TAG == "[family=xgb]" and SP.AXES_TAG == "[axes: data=noaa-enlil]"
    assert SP.HYPOTHESIS.startswith("[family=xgb]") and "[axes: data=noaa-enlil]" in SP.HYPOTHESIS
    for cite in ("E0003", "E0206", "X0195", "rocv", "6/8", "mean+2SE<0", "12h", "e0003_rocv"):
        assert cite in SP.HYPOTHESIS, f"hypothesis missing {cite}"
    assert "ingestion" in SP.HYPOTHESIS and "X0195" in SP.HYPOTHESIS
    assert set(SP.EXPERIENCE) == {"X0195", "X0005", "X0004"}
    assert tuple(SP.SEEDS_ROCV) == (0, 1, 2)
    assert "selection/audit" in SP.SELECT_RULE and "ROCV_ONLY" in SP.SELECT_RULE
    assert SP.RULE_H == 12 and SP.ROCV_FREQ == "3h" and SP.FIT_NJOBS <= 6
    assert len(SP.FEATURES) == 6 and "noaa_vintage_ok" in SP.FEATURES
    assert str(SP.SCRATCH) == "store/scratch/worker-t2n2"
    for k in ("python", "xgboost", "pandas", "numpy"):
        assert k in SP.PINS


def test_operational_collection_receipt():
    assert all(NC.verify_header_sha256(r) for r in NC.RECORDED)
    assert NC.RULE_H == 12
    assert len(NC.CANDIDATE_INITS) >= 20  # rocv-window coverage, not 9
    for init in NC.CANDIDATE_INITS:
        url = NC.candidate_url(init)
        assert init[:10].replace("-", "") in url and url.endswith(".tar.gz")
    assert RECEIPT.exists(), "run noaa_collect_op.collect() before testing"
    r = json.loads(RECEIPT.read_text())
    assert r["direction"] == "D137" and r["owner"] == "worker-t2n2"
    assert int(r["rule"]["rule_h"]) == 12 and "12h" in r["rule"]["rule"]
    assert "operational" in r["rule"]["rule"]
    assert r["bodies_fetched"] == 0 and "0 bodies" in r["transfer_budget"]
    assert all(f["body_sha256"] is None for f in r["files"])
    assert len(r["eligible_inits"]) >= 20  # 12h rule admits the window
    assert "operational_proxy" in r and "vintage" in r["operational_proxy"]
    assert set(r["experience"]) == {"X0195", "X0005", "X0004"}


def test_receipt_ignores_ingestion_delay():
    # A Jan-2026 init ingested with ~192h delay is INELIGIBLE under the old 4039h
    # rule yet ELIGIBLE under the 12h operational rule: ingestion never decides.
    r = json.loads(RECEIPT.read_text())
    jan = [f for f in r["files"] if f["init_utc"] == "2026-01-15T00:00:00Z"][0]
    assert jan["delay_h"] is not None and jan["delay_h"] > 12.0
    assert jan["eligible"] is True
    assert "2026-01-15T00:00:00Z" in r["eligible_inits"]


def test_features_causal_12h_newest_wins_and_stale():
    rec = _synth_receipt()
    o = pd.DatetimeIndex(["2024-06-01T00:00:00Z", "2024-06-15T12:00:00Z",
                          "2024-07-20T00:00:00Z", "2024-08-20T00:00:00Z"])
    feat = NF.add_features(o, rec)
    assert list(feat.columns) == SP.FEATURES == NF.FEATURES
    assert feat.loc[o[0], "noaa_stale"] == 1.0 and feat.loc[o[0], "noaa_age_h"] == -1.0
    assert feat.loc[o[1], "noaa_stale"] == 0.0  # init + exactly 12h is eligible
    assert feat.loc[o[3], "noaa_age_h"] > feat.loc[o[2], "noaa_age_h"]  # newest wins
    assert (feat["noaa_vintage_ok"] == 0.0).all()
    assert (feat["noaa_rule_h"] == 12.0).all()


def test_poison_real_with_positive_control():
    rec = _synth_receipt()
    origins = pd.date_range("2024-06-01", "2024-09-01", freq="24h", tz="UTC")
    pz = NF.poison_test(origins, sorted(rec["eligible_inits"])[0], rec)
    assert pz["passed"] is True and pz["pre_invariant"] is True
    assert set(pz) >= {"passed", "cutoff_utc", "n_pre_rows", "n_post_rows", "moved_columns"}
    assert len(pz["moved_columns"]) > 0 and pz["n_pre_rows"] > 0 and pz["n_post_rows"] > 0


def test_owned_files_capped_and_scratch_separated():
    files = sorted(f.name for f in PKG.iterdir() if f.is_file())
    py = [f for f in files if f.endswith(".py")]
    assert len(files) <= 5, files
    assert set(py) == {"spec.py", "noaa_collect_op.py", "noaa_features_op.py", "runner_rocv.py"}
    assert "report.md" in files
    blob = "\n".join((PKG / f).read_text(errors="ignore") for f in files)
    scrubbed = blob.replace("worker-t2n2", "")
    assert "worker-t2n" not in scrubbed, "retry must not touch the predecessor scratch"
    for other in OTHER_SCRATCH:
        assert other not in blob, f"foreign id {other}"
    assert "store/scratch/worker-t2n2" in blob
    assert "store/scratch/worker-t2n\"" not in scrubbed and "store/scratch/worker-t2n/" not in scrubbed


def test_py_has_no_forbidden_execution():
    for f in PKG.glob("*.py"):
        text = f.read_text()
        for s in PY_FORBIDDEN:
            assert s not in text, f"{f.name} contains forbidden {s!r}"
        if f.name not in NET_OK_ONLY_IN:
            assert "urlopen" not in text and "Request(" not in text, f"{f.name} must stay offline"
    runner = (PKG / "runner_rocv.py").read_text()
    assert "from expos import rocv" in runner  # e0003_rocv template adherence
    assert "stored-prediction" in runner or "stored prediction" in runner


def test_seed_propagated_budget_and_gpu_template():
    runner = (PKG / "runner_rocv.py").read_text()
    assert "SEEDS_ROCV" in runner and "random_state" in runner
    assert "FIT_NJOBS" in runner
    assert '"cuda"' in runner or "'cuda'" in runner  # GPU training per e0003_rocv template
    assert "gpu_lease" not in runner  # leases are taken by expos run, never in code
    assert "rocv/f{k}_s{s}.parquet" in runner or "f{k}_s{s}" in runner
    assert "72h" in runner  # fold -72h purge
    assert "learning_curves" in runner  # curves saved
    assert "EXPOS_SMOKE" in runner


def test_rocv_outputs_poison_curves_and_eval():
    recs, mine = _mine()
    assert mine and all(m["direction"] == "D137" and m["kind"] == "full" for m in mine)
    latest = mine[-1]
    assert latest["reference"] == "E0206"
    assert set(latest["code"]) == {"experiments/d137_t2noaa/retry/spec.py",
                                   "experiments/d137_t2noaa/retry/noaa_collect_op.py",
                                   "experiments/d137_t2noaa/retry/noaa_features_op.py",
                                   "experiments/d137_t2noaa/retry/runner_rocv.py"}
    assert latest.get("flags") == []
    runs = [r for r in recs if r.get("event") == "run" and r.get("id") == latest["id"]]
    assert runs and runs[-1]["rc"] == 0
    assert runs[-1]["gpu"] is not None  # empty-GPU run, lease held by expos
    out = LB.parent / "research" / "expos" / "runs" / latest["id"]
    got = sorted((out / "rocv").glob("f*_s*.parquet"))
    assert len(got) == 24, f"8 folds x 3 seeds required, got {len(got)}"
    want = {f"f{k}_s{s}.parquet" for k in range(8) for s in (0, 1, 2)}
    assert {p.name for p in got} == want
    for p in got:
        df = pd.read_parquet(p)
        assert list(df.columns) == ["origin_last_input_utc", "horizon_hours", "pred_kms"]
        assert len(df) > 0 and df["horizon_hours"].between(1, 72).all()
    pz = json.loads((out / "poison.json").read_text())
    assert pz["passed"] is True and len(pz["moved_columns"]) > 0
    curves = json.loads((out / "learning_curves.json").read_text())
    assert len(curves) == 24 and all(len(v["valid_rmse"]) > 0 for v in curves.values())
    rocv_recs = [r for r in recs if r.get("event") == "rocv" and r.get("id") == latest["id"]]
    assert rocv_recs, "run rocv-eval before testing"
    gate = rocv_recs[-1]["gate"]
    assert gate["reference"] == "E0206"
    for k in ("rel_by_fold", "rel_by_seed", "mean_relative", "seed_se",
              "n_folds_improved", "f1_seedmean", "reference_f1_seedmean"):
        assert k in gate, f"rocv gate missing {k}"


def test_no_foreign_runs_in_graph():
    recs, mine = _mine()
    ids = {m.get("id") for m in mine}
    others = [r for r in recs if r.get("event") == "run" and r.get("id") not in ids
              and "worker-t2n2" in json.dumps(r.get("log", ""))]
    assert others == []
    for r in recs:
        if r.get("event") in ("run", "rocv", "evaluate") and r.get("id") in ids:
            continue
        if r.get("event") == "register" and r.get("owner") == "worker-t2n2":
            assert r.get("id") in ids


def test_report_carries_receipt_rocv_and_poison():
    assert REPORT.exists(), "write retry/report.md after rocv-eval before testing"
    rep = REPORT.read_text()
    _, mine = _mine()
    for token in ["12h", "operational", "HEAD", "0 bodies", "noaa_age_h",
                  "noaa_since_init_h", "vintage_ok", "poison", "E0206", mine[-1]["id"],
                  "worker-t2n2", "X0195", "rocv", "6/8", "mean", "F1", "수집영수증"]:
        assert token in rep, f"report missing {token!r}"
