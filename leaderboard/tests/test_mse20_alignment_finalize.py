"""Adversarial retention and subclass finalization tests, no official data."""
import copy
import importlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments.mse20_four_regimes import alignment_search as a, finalize as f, protocol as p, tree_alignment as model
from test_mse20_tree import _sources, _frames


def row(family, name, mse):
    return {"status": "ok", "family": family, "config": {"id": name},
            "metrics": {"mse": mse, "precision": 1., "recall": 1.}}


def test_dropped_selection_cannot_win():
    retained = row("tree_alignment", "retained", 100)
    dropped = row("tree", "dropped", 0)
    retention = {"retained": [{"family": "tree_alignment", "id": "retained", "config": retained["config"]}]}
    assert a.eligible_selection([dropped, retained], retention) == [retained]
    assert f._choose([{"name": "retained", "metrics": retained["metrics"]}], retained["metrics"])["name"] == "retained"


def test_selection_cannot_change_development_retention(tmp_path, monkeypatch):
    dev = [row("tree", f"d{i}", 10 + i) for i in range(10)]
    dev[0]["config"]["id"] = "champion"
    aligned = [row("tree_alignment", "a0", 1), row("tree_alignment", "a1", 2)]
    monkeypatch.setattr(a, "verify_trial", lambda record: None)
    p.write_json(tmp_path / "tree-development.json", dev)
    p.write_json(tmp_path / "tree_alignment-development.json", aligned)
    original = a.retention_rows(tmp_path)[1]
    p.write_json(tmp_path / "tree-selection.json", [row("tree", "d9", -1e9)])
    assert a.retention_rows(tmp_path)[1] == original


def test_retention_and_schema_tampering_rejected(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    p.write_json(source, {"unchanged": True})
    retained = [row("tree_alignment", "allowed", 1)]
    monkeypatch.setattr(a, "retention_rows", lambda directory: (retained, retained))
    record = {"files": {"source": p.file_ref(source)}, "retained": [{"family": "tree_alignment", "id": "forged", "config": {"id": "forged"}}]}
    p.write_json(tmp_path / "tree-retention.json", record)
    with pytest.raises(ValueError, match="retention was altered"):
        a.load_retention(tmp_path)
    meta = model.schema_identity()
    meta["delays"] = [1, 2, 3]
    with pytest.raises(ValueError, match="schema differs"):
        f.adapter_schema({"family": "tree_alignment", "metadata": meta})
    source.write_text("tampered")
    with pytest.raises(ValueError, match="stale artifact"):
        a.load_retention(tmp_path)


def test_stage_deadlines_without_clock_reset(tmp_path, monkeypatch):
    runtime = {"search_stop_utc": "2026-10-03T10:56:29Z", "optimization_stop_utc": "2026-10-03T13:11:29Z", "deadline_utc": "2026-10-03T14:26:29Z"}
    p.write_json(tmp_path / "runtime.json", runtime)
    real = pd.Timestamp
    class Clock:
        current = real("2026-10-03T10:56:29Z")
        def __new__(cls, value):
            return real(value)
        @classmethod
        def now(cls, tz=None):
            return cls.current
    monkeypatch.setattr(p.pd, "Timestamp", Clock)
    for stage, field in (("search", "search_stop_utc"), ("optimization", "optimization_stop_utc"), ("deadline", "deadline_utc")):
        Clock.current = real(runtime[field])
        with pytest.raises(TimeoutError):
            p.check_deadline(tmp_path, stage)
    assert json.loads((tmp_path / "runtime.json").read_text()) == runtime


def test_alignment_freeze_dispatch_and_saved_inference(tmp_path, monkeypatch):
    ace, ch, raw = _sources()
    for j, tag in enumerate(p.common.CH_TAGS):
        for col in p.common.CH_BASE_COLUMNS:
            ch[f"ch_{tag}_{col}"] = np.arange(len(ch)) / 100 + j
    frames = _frames(ace, ch, raw)
    cutoff = frames["speed"].index[850]
    origins = frames["speed"].index[[950, 960]]
    cfg = {"id": "alignment_fixture", "params": {"n_estimators": 3, "max_depth": 2}, "seed": 42, "max_train_origins": 12}
    pred, metadata = model.fit_predict(frames, cutoff, origins, cfg, "cpu", tmp_path / "selected-model")
    aligned = {"family": "tree_alignment", "status": "ok", "config": cfg, "metadata": metadata,
               "metrics": {"mse": 0., "precision": 1., "recall": 1.}}
    path = tmp_path / "aligned.parquet"
    p.canonical_predictions(pred, origins).to_parquet(path, index=False)
    aligned["prediction"] = p.file_ref(path)
    reference = copy.deepcopy(aligned)
    reference.update(family="tree", config={**cfg, "id": "champion"}, metrics={"mse": 1000., "precision": 1., "recall": 1.})
    reference_path = tmp_path / "reference.parquet"
    p.canonical_predictions(pred.assign(pred_kms=pred.pred_kms + 100), origins).to_parquet(reference_path, index=False)
    reference["prediction"] = p.file_ref(reference_path)
    p.write_json(tmp_path / "tree-selection.json", [reference])
    p.write_json(tmp_path / "tree_alignment-selection.json", [aligned])
    for name in ("splits", "tree-retention", "tree_alignment-development", "tree_alignment-preregistered"):
        p.write_json(tmp_path / f"{name}.json", {})
    p.write_json(tmp_path / "contract.json", {"files": {}, "official_cutoff": cutoff.isoformat(), "event_origins": [o.isoformat() for o in origins]})
    retention = {"retained": [{"family": "tree_alignment", "id": cfg["id"], "config": cfg}]}
    monkeypatch.setattr(f, "verify_trial", lambda record: None)
    monkeypatch.setattr(f, "load_retention", lambda directory: retention)
    monkeypatch.setattr(p, "require_gate0", lambda directory: None)
    monkeypatch.setattr(p, "check_deadline", lambda *args: 10000)
    monkeypatch.setattr(p, "load_frames", lambda *args: frames)
    monkeypatch.setattr(p, "split_origins", lambda *args: origins)
    monkeypatch.setattr(p, "SPLITS", {"selection": {"end": cutoff.isoformat()}, "audit": {"cutoff": cutoff.isoformat(), "end": origins.max().isoformat()}})
    expected = p.canonical_predictions(pred, origins).pred_kms.to_numpy()
    monkeypatch.setattr(p, "score_development", lambda values, *_: {"mse": float(np.square(values.pred_kms.to_numpy() - expected).mean()), "precision": 1., "recall": 1.})
    f.freeze(tmp_path)
    frozen = json.loads((tmp_path / "frozen-finalist.json").read_text())
    component = frozen["recipe"]["config"][0]
    assert component["family"] == "tree_alignment"
    assert component["adapter_schema"] == model.schema_identity()
    assert importlib.import_module(f"experiments.mse20_four_regimes.{component['family']}") is model
    f.generate("audit", "cpu", tmp_path)
    f.generate("official", "cpu", tmp_path)
    report = json.loads((tmp_path / "official-report.json").read_text())
    assert report["saved_inference_max_abs_difference"] == 0
    assert report["components"][component["key"]]["metadata"]["subclass_identity"] == model.SUBCLASS_IDENTITY
