"""Submission and source parsing contracts for the lead-first metric migration."""
import ast
import io
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "leaderboard"), str(ROOT / "leaderboard" / "scorer")]
import submit as client
from competition import VERSION
from competition_data import parse_month


def metadata():
    return {
        "team_member": "alice", "experiment": "competition_test", "description": "test",
        "no_future_leakage": True,
        "f1_training_data_end_utc": "2025-12-29T00:00:00Z",
        "f2_training_data_end_utc": "2026-03-29T00:00:00Z",
        "f3_training_data_end_utc": "2026-06-29T00:00:00Z",
        "config": {"data": {"source": "ace"}, "hyperparameters": {"tau": 48}},
        "inference_seconds_per_fold": 0.001,
        "architecture_url": "https://github.com/TLAB-HGU/sunrun-leaderboard/blob/"
                            + "a" * 40 + "/research/test/experiment.py",
    }


@pytest.fixture
def submission_environment(tmp_path, monkeypatch):
    index = pd.date_range("2025-12-30", "2026-10-01", freq="h", tz="UTC")
    observations = pd.DataFrame({"timestamp_utc": index,
                                 "speed_kms": 400.0 + np.arange(len(index)) % 151})
    # One valid forecast, two invalid outputs and all remaining forecasts omitted.
    frame = pd.DataFrame({"origin_last_input_utc": ["2026-01-01T00:00:00Z"] * 3,
                          "horizon_hours": [1, 2, 3], "pred_kms": [450.0, np.nan, -1.0]})
    path = tmp_path / "predictions.parquet"
    frame.to_parquet(path, index=False)
    calls = {"downloads": [], "commits": [], "plots": [], "pages": []}

    class API:
        def repo_info(self, repo, repo_type):
            return SimpleNamespace(sha="frozen-parent")

        def create_commit(self, **kwargs):
            calls["commits"].append(kwargs)
            return SimpleNamespace(oid="new-head")

    api = API()
    monkeypatch.setattr(client, "HfApi", lambda: api)
    monkeypatch.setattr(client, "require_token", lambda: None)
    monkeypatch.setattr(client, "require_script", lambda *args, **kwargs: "https://example.org/script.py")
    monkeypatch.setattr(client.architecture_code, "reachable", lambda _: True)

    observation_path = tmp_path / "observations.parquet"
    observations.to_parquet(observation_path, index=False)

    def store(dataset, name, revision=None):
        calls["downloads"].append((name, dataset))
        assert name == "competition-observations.parquet", "old filled truth must not be used"
        assert revision == "frozen-parent"
        return str(observation_path)

    def all_results(*args, **kwargs):
        if not calls["commits"]:
            return []
        operation = calls["commits"][-1]["operations"][1]
        return [json.loads(operation.path_or_fileobj)]

    def plots(records, output_dir):
        calls["plots"].append(records)
        return [{**record, "metric_plot_url": "https://raw.githubusercontent.com/TLAB-HGU/"
                 "sunrun-leaderboard/" + "b" * 40 + "/leaderboard/assets/metrics/test.png"}
                for record in records]

    def publish(api, records, space, **kwargs):
        calls["pages"].append({"records": records, **kwargs})
        return "https://huggingface.co/spaces/test/board"

    monkeypatch.setattr(client, "_download", store)
    monkeypatch.setattr(client, "all_results", all_results)
    monkeypatch.setattr(client, "with_extras", lambda api, dataset, records: records)
    monkeypatch.setattr(client, "publish_plots", plots)
    monkeypatch.setattr(client, "publish", publish)
    return path, frame, calls


def test_dry_run_scores_missing_predictions_without_remote_writes(submission_environment):
    path, _, calls = submission_environment
    result = client.submit(str(path), metadata(), dry_run=True)
    assert result["competition"]["version"] == VERSION
    assert result["competition"]["total"]["n_replaced"] == 1052 * 72 - 1
    assert calls["downloads"] == [("competition-observations.parquet", client.DATASET)]
    assert calls["commits"] == calls["plots"] == calls["pages"] == []
    assert "metric_plot_url" not in result


def test_live_submission_stores_original_predictions_metrics_and_plot_atomically(submission_environment):
    path, original, calls = submission_environment
    result = client.submit(str(path), metadata())
    assert len(calls["plots"]) == len(calls["commits"]) == len(calls["pages"]) == 1
    commit = calls["commits"][0]
    assert commit["parent_commit"] == "frozen-parent"
    assert commit["repo_type"] == "dataset"
    operations = commit["operations"]
    assert len(operations) == 2
    assert operations[0].path_in_repo.startswith("submissions/alice/competition_test/")
    assert operations[1].path_in_repo.startswith("results/alice/competition_test/")
    pd.testing.assert_frame_equal(pd.read_parquet(io.BytesIO(operations[0].path_or_fileobj)), original)
    stored = json.loads(operations[1].path_or_fileobj)
    assert stored["competition"] == result["competition"]
    assert stored["competition"]["total"]["n_replaced"] == 1052 * 72 - 1
    assert stored["metric_plot_url"] == result["metric_plot_url"]
    assert "/" + "b" * 40 + "/" in stored["metric_plot_url"]
    assert calls["pages"][0]["dataset"] == client.DATASET
    assert stored["competition_dataset_revision"] == "frozen-parent"
    assert len(stored["competition_provenance"]) == 64


def notebook_parser():
    notebook = json.loads((ROOT / "metric.ipynb").read_text())
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "def parse_month(" not in source:
            continue
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "parse_month")
        namespace = {"np": np, "pd": pd, "re": re, "StringIO": io.StringIO,
                     "RAW_COLUMNS": ["year", "month", "day", "hhmm", "mjd", "seconds_of_day",
                                     "status", "density", "speed_raw_kms", "temperature"],
                     "MONTH_FILE_PATTERN": re.compile(r"^\d{6}_ace_swepam_1h\.txt$")}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "metric.ipynb", "exec"), namespace)
        return namespace["parse_month"]
    raise AssertionError("notebook parser not found")


def test_monthly_parser_matches_notebook_mask_and_boundary_selection():
    payload = b"""# NASA hourly SWEPAM
: year month day hhmm mjd seconds status density speed temperature
2025 12 31 2300 60000 82800 0 5 333 10000
2026 01 01 0000 60001 0 0 5 400 10000
2026 01 01 0100 60001 3600 1 5 900 10000
2026 01 01 0200 60001 7200 0 5 -9999 10000
2026 01 01 0300 60001 10800 0 5 inf 10000
2026 01 01 0400 60001 14400 0 5 0 10000
2026 01 01 0500 60001 18000 0 5 550 10000
2026 02 01 0000 60032 0 0 5 700 10000
"""
    name = "202601_ace_swepam_1h.txt"
    expected = notebook_parser()(payload.decode(), name)[["timestamp_utc", "speed_kms"]]
    actual = parse_month(payload, name)
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True))
    assert actual.speed_kms.notna().sum() == 2


@pytest.mark.parametrize("payload", [b"bad row", b"# no data", b"2026 01 01 0030 60001 0 0 5 400 10000"])
def test_monthly_parser_rejects_invalid_input(payload):
    with pytest.raises(ValueError):
        parse_month(payload, "202601_ace_swepam_1h.txt")
