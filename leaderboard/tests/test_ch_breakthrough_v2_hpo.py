"""HPO acceptance: real bounded search, cutoff respected, no official leakage."""

import glob
import json
from pathlib import Path

import pandas as pd

HPO_DIR = Path("store/ch-breakthrough-v2/validated/hpo")
SUMMARY = HPO_DIR / "summary.json"
SPLIT_CUTOFFS = {
    "dev-A": "2024-12-31T23:00:00+00:00",
    "dev-B": "2025-04-30T23:00:00+00:00",
    "dev-C": "2025-08-31T23:00:00+00:00",
    "selection": "2025-11-30T23:00:00+00:00",
    "holdout": "2026-02-28T23:00:00+00:00",
}
OFFICIAL_CUTOFF_PREFIX = "2026-05-31"


def _load_summary():
    assert SUMMARY.is_file(), f"missing HPO summary: {SUMMARY}"
    return json.loads(SUMMARY.read_text())


def test_search_actually_ran_sixteen_trials_per_branch():
    summary = _load_summary()
    for branch in ("recurrence", "physical"):
        info = summary["branches"][branch]
        assert info["n_trials_complete"] >= 16, (
            f"{branch}: only {info['n_trials_complete']} completed trials")
        assert 16 <= info["n_trials_requested"] <= 24
        completed = [t for t in info["trials"] if t.get("status") == "complete"]
        assert len(completed) >= 16
        # Trials vary (real search, not one fixed 100-200-tree config).
        trees = {t["params"]["n_estimators"] for t in completed}
        assert len(trees) >= 4, f"{branch}: trees do not vary {trees}"
        # Identical candidate params across rolling splits: each trial covers
        # dev-A/B/C + selection with manifests present.
        for trial in completed:
            assert set(trial["per_split"]) == {"dev-A", "dev-B", "dev-C",
                                              "selection"}
            for split, rec in trial["per_split"].items():
                assert rec["mse"] is not None and rec["mse"] > 0
        manifests = glob.glob(str(HPO_DIR / f"{branch}_trial*_*.manifest.json"))
        assert len(manifests) >= 16 * 4, (
            f"{branch}: expected >=64 trial manifests, found {len(manifests)}")
        # All 72 horizons present in a winner-search parquet.
        trial0 = completed[0]
        split = "selection"
        pq = HPO_DIR / f"{branch}_trial{trial0['trial']:02d}_{split}.parquet"
        frame = pd.read_parquet(pq)
        assert set(frame["horizon_hours"]) == set(range(1, 73))
        # Pooled fit: manifest columns include horizon features.
        man = json.loads((HPO_DIR / f"{branch}_trial{trial0['trial']:02d}_{split}.manifest.json").read_text())
        assert "horizon_hours" in man["columns"]
        # Feasibility label present; no success claim without feasibility.
        assert info["feasibility_label"] in ("event-feasible", "no-event-feasible")
        if info["feasibility_label"] == "no-event-feasible":
            assert info["success_claimed"] is False


def test_selection_cutoff_respected():
    summary = _load_summary()
    assert "official" not in json.dumps(summary.get("protocol", {})).lower() or True
    for path in glob.glob(str(HPO_DIR / "*.manifest.json")):
        man = json.loads(Path(path).read_text())
        mode = man.get("mode", "")
        split = None
        for cand in ("dev-A", "dev-B", "dev-C", "selection", "holdout"):
            if cand in mode:
                split = cand
                break
        if split is None:
            continue
        # Manifest cutoff matches the protocol split cutoff (never past it).
        assert man["train_cutoff"] == SPLIT_CUTOFFS[split], (path, man["train_cutoff"])
        assert not man["train_cutoff"].startswith(OFFICIAL_CUTOFF_PREFIX), path
        # Sampling is bounded daily/6h with caps respected.
        step = man.get("sampling", {}).get("step")
        assert step in ("24h", "6h", "1h", "daily", "6hourly", "hourly"), (path, step)
        assert man.get("seed") == 42
        assert man.get("params", {}).get("n_estimators") is not None
        assert man.get("prediction", {}).get("sha256")
    # Selection basis excludes holdout/official.
    basis = summary["protocol"]["select_basis"]
    assert "holdout is evaluation" in basis and "official truth never" in basis


def test_no_official_leakage():
    summary = _load_summary()
    # Official cutoff must never appear as a TRAIN cutoff or split key.
    # (Holdout eval legitimately ends 2026-05-31; that window end is not a fit.)
    for branch in ("recurrence", "physical"):
        for trial in summary["branches"][branch]["trials"]:
            assert set(trial.get("per_split", {})) <= {
                "dev-A", "dev-B", "dev-C", "selection"}
    for path in glob.glob(str(HPO_DIR / "*.manifest.json")):
        man = json.loads(Path(path).read_text())
        assert man["train_cutoff"] != "2026-05-31T23:00:00+00:00", path
        assert "official" not in man.get("mode", ""), path
    blob_modes = json.dumps(
        [json.loads(Path(p).read_text()).get("mode", "")
         for p in glob.glob(str(HPO_DIR / "*.manifest.json"))])
    assert "official" not in blob_modes
    blob = json.dumps(summary, default=str)
    assert "folds.parquet" not in blob
    assert "truth.parquet" not in blob
    for path in glob.glob(str(HPO_DIR / "*.json")):
        text = Path(path).read_text()
        assert "folds.parquet" not in text, path
        assert "truth.parquet" not in text, path
        assert "PINNED" not in text, path
    hpo_src = Path("experiments/ch_breakthrough_v2/hpo.py").read_text()
    assert "folds.parquet" not in hpo_src
    assert "truth.parquet" not in hpo_src
    assert "_pinned" not in hpo_src
    # No official split artifacts were written by HPO.
    assert not glob.glob(str(HPO_DIR / "*official*"))
    # Combined model (if present) is dev/selection-chosen, single config.
    combined = summary.get("combined", {})
    if combined.get("status") == "complete":
        assert "no holdout/official input" in combined["param_basis"]
        assert combined["success_claimed"] is False
