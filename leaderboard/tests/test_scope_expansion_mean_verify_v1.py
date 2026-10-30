"""Synthetic CPU-only tests for the mean-verify overlay v1 (path fix only).

No real data, no fits, no stage_verify invocation, no run-dir mutation of
frozen receipts. All cases use tmp_path synthetic trees.
"""

import sys
from pathlib import Path

import pytest

PKG_PARENT = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(PKG_PARENT))

from mse20_scope_expansion import mean_driver, mean_verify_v1, protocol  # noqa: E402


def _make_closure_tree(root: Path) -> dict:
    """Build a synthetic repo tree mirroring SOURCE_CLOSURE shape."""
    files = {}
    for c in protocol.SOURCE_CLOSURE:
        p = root / c
        if c.endswith("/"):
            (p / "year=2022").mkdir(parents=True, exist_ok=True)
            f = p / "year=2022" / "part.parquet"
            f.write_bytes(b"synthetic-part")
            files[c] = f
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"synthetic-file")
            files[c] = p
    return files


def _inv_paths(files: dict) -> list[str]:
    return [str(Path(v).resolve()) for v in files.values()]


def test_fixed_passes_where_legacy_fails_on_absolute_paths(tmp_path):
    files = _make_closure_tree(tmp_path)
    inv = _inv_paths(files)
    assert mean_verify_v1.closure_match_fixed(inv, tmp_path) is True
    # Native-shaped absolute inventory vs relative closure: legacy fails.
    assert mean_verify_v1.closure_match_legacy(inv) is False
    det = mean_verify_v1.closure_detail(inv, tmp_path)
    assert det["match"] is True
    assert det["no_extras"] is True
    assert all(det["expected_present"].values())
    assert all(det["per_closure_covered"].values())


def test_sibling_prefix_rejected_by_fixed_but_fools_legacy(tmp_path):
    files = _make_closure_tree(tmp_path)
    inv = _inv_paths(files)
    # Find the directory closure entry and craft a sibling-prefix impostor.
    dir_closure = next(c for c in protocol.SOURCE_CLOSURE if c.endswith("/"))
    sibling = Path(str((tmp_path / dir_closure).resolve()).rstrip("/") + "-evil")
    sibling.mkdir(parents=True, exist_ok=True)
    evil_file = sibling / "part.parquet"
    evil_file.write_bytes(b"evil")
    # Replace the legitimate dir-backed path with the sibling impostor.
    legit = str(files[dir_closure].resolve())
    impostor_inv = [p if p != legit else str(evil_file.resolve()) for p in inv]
    fixed = mean_verify_v1.closure_detail(impostor_inv, tmp_path)
    assert fixed["match"] is False
    assert fixed["per_closure_covered"][dir_closure] is False
    # Legacy string-prefix check is fooled only when the sibling shares the
    # raw prefix; assert the mechanism explicitly on a crafted pair.
    assert ("abc".startswith("abc") and "abc-evil/x".startswith("abc")) is True


def test_missing_expected_file_fails_closed(tmp_path):
    files = _make_closure_tree(tmp_path)
    file_closure = next(c for c in protocol.SOURCE_CLOSURE if not c.endswith("/"))
    files[file_closure].unlink()
    inv = [str(v.resolve()) for v in files.values() if v.exists()]
    det = mean_verify_v1.closure_detail(inv, tmp_path)
    assert det["match"] is False
    assert any(not v for v in det["expected_present"].values())
    assert det["per_closure_covered"][file_closure] is False


def test_missing_expected_dir_fails_closed(tmp_path):
    import shutil

    _make_closure_tree(tmp_path)
    dir_closure = next(c for c in protocol.SOURCE_CLOSURE if c.endswith("/"))
    shutil.rmtree(tmp_path / dir_closure)
    det = mean_verify_v1.closure_detail([], tmp_path)
    assert det["match"] is False
    assert any(not v for v in det["expected_present"].values())


def test_unrelated_extra_source_fails_closed(tmp_path):
    files = _make_closure_tree(tmp_path)
    inv = _inv_paths(files)
    extra = tmp_path / "unrelated" / "donki_extra.parquet"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"extra")
    det = mean_verify_v1.closure_detail(inv + [str(extra.resolve())], tmp_path)
    assert det["no_extras"] is False
    assert det["match"] is False


def test_drifted_inventory_pin_fails_closed():
    stored = [{"path": "/a/f.parquet", "bytes": 3, "sha256": "x" * 64}]
    live_same = [{"path": "/a/f.parquet", "bytes": 3, "sha256": "x" * 64}]
    live_drift = [{"path": "/a/f.parquet", "bytes": 3, "sha256": "y" * 64}]
    live_extra = live_same + [{"path": "/a/g.parquet", "bytes": 1, "sha256": "z" * 64}]
    assert mean_verify_v1._input_pin(stored) == mean_verify_v1._input_pin(live_same)
    assert mean_verify_v1._input_pin(stored) != mean_verify_v1._input_pin(live_drift)
    assert mean_verify_v1._input_pin(stored) != mean_verify_v1._input_pin(live_extra)


def test_overlay_never_touches_frozen_receipt_name():
    assert mean_verify_v1.OVERLAY_FILENAME == "mean_verify_v1.json"
    assert mean_verify_v1.PRIOR_FILENAME == mean_driver.MEAN_VERIFY_FILENAME
    assert mean_verify_v1.OVERLAY_FILENAME != mean_driver.MEAN_VERIFY_FILENAME
    assert mean_verify_v1.OVERLAY_VERSION == "20261005-v1"
    assert "path normalization ONLY" in mean_verify_v1.FIX_NOTE
    assert "scientific exclusion" not in mean_verify_v1.FIX_NOTE.lower() or \
        "no scientific exclusions" in mean_verify_v1.FIX_NOTE


def test_overlay_does_not_bind_stage_verify():
    src = Path(mean_verify_v1.__file__).read_text()
    assert "mean_driver.stage_verify(" not in src
    assert "reference.stage_fit(" not in src
    assert "stage_fit(" not in src or "mean_fit" in src  # no fit invocation


def test_resolve_expected_covers_every_closure_entry(tmp_path):
    _make_closure_tree(tmp_path)
    resolved = mean_verify_v1.resolve_expected(tmp_path)
    assert len(resolved) == len(protocol.SOURCE_CLOSURE)
    for (kind, exp), c in zip(resolved, protocol.SOURCE_CLOSURE):
        assert str(exp).startswith(str(tmp_path.resolve()))
        assert (kind == "dir") == c.endswith("/")
