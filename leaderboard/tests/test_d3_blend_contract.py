"""Contract for D3 72h-mean level + direct shape blend.

- Feature choice is explicit existing_all (no strip/SUVI path in this lane).
- Blend weight w comes from dev OOF only: blend.py never mentions the other
  windows; runner oof_weight reads PERIODS["dev"] only; no boundary-date or
  forbidden-truth literal appears in either new file.
- Blend identity pred = direct + w*(level - mean_direct): w=0 recovers
  direct (F1-safe), w=1 is level + demeaned shape; ties pick smaller w.
- Output/poison interface matches the expos contract (full 72 horizons,
  exact columns, 3 probes with exact-key + positive-control semantics).
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

D3 = Path(__file__).resolve().parents[1] / "experiments" / "d3_level_blend"
sys.path.insert(0, str(D3))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "headroom_audit"))
import blend as BL  # noqa: E402

FORBIDDEN = ["datasets--tlabtlab--sunrun-lb-store", "truth.parquet", "folds.parquet", "naive.parquet",
             "official-predictions", "baseline-replay.parquet", "official_check", "official-check",
             "predictions_official_truth"]
FORBIDDEN_DATE_STRINGS = ("2025-12-04", "2026-02-25", "2026-03-04", "2026-05-28",
                          "2026-05-29", "2026-05-31", "2026-09-26")


def test_feature_choice_is_existing_all_explicit():
    assert BL.FEATURE_CHOICE == "existing_all"
    import runner as R
    assert R.FEATURE_CHOICE == "existing_all" == BL.FEATURE_CHOICE
    for name in ("blend.py", "runner.py"):
        text = (D3 / name).read_text()
        assert "existing_all" in text, name
        assert "strip" not in text.lower() or "strips stay in D1" in text, name
        assert "suvi" not in text.lower(), name


def test_no_forbidden_or_window_dates_in_new_code():
    for name in ("blend.py", "runner.py"):
        text = (D3 / name).read_text()
        assert not [p for p in FORBIDDEN if p in text], name
        for s in FORBIDDEN_DATE_STRINGS:
            assert s not in text, f"{name} must not reference {s}"


def test_blend_module_is_dev_only_static():
    src = (D3 / "blend.py").read_text().lower()
    assert "selection" not in src and "audit" not in src
    rsrc = (D3 / "runner.py").read_text()
    assert 'PERIODS["dev"]' in rsrc
    assert 'PERIODS["selection"]' not in rsrc and 'PERIODS["audit"]' not in rsrc
    sig = inspect.signature(BL.fit_blend_weight)
    assert list(sig.parameters) == ["oof"]


def test_blend_identity_and_tie_break():
    oof = pd.DataFrame({"y": [400.0, 500.0, 600.0],
                        "direct": [410.0, 490.0, 610.0],
                        "level": [450.0, 500.0, 550.0],
                        "mean_direct": [430.0, 500.0, 570.0]})
    w, table = BL.fit_blend_weight(oof)
    assert w in BL.W_GRID and 0.0 <= w <= 1.0
    assert table[w] == min(table.values())
    d = np.array([400.0, 500.0])
    assert np.allclose(BL.apply_blend(d, np.array([450.0, 550.0]),
                                      np.array([430.0, 520.0]), 0.0), d)
    assert np.allclose(BL.apply_blend(d, np.array([450.0, 550.0]),
                                      np.array([430.0, 520.0]), 1.0),
                       d + np.array([20.0, 30.0]))
    # Tie (level == mean_direct everywhere) must pick the smallest w = 0.0.
    tie = pd.DataFrame({"y": [400.0, 500.0], "direct": [400.0, 500.0],
                        "level": [450.0, 550.0], "mean_direct": [450.0, 550.0]})
    wt, _ = BL.fit_blend_weight(tie)
    assert wt == 0.0


def test_mean_schema_and_grids():
    assert BL.N_OOF_FOLDS == 4
    assert tuple(BL.W_GRID) == tuple(i / 20 for i in range(21))
    assert "pred = direct + w*(level - mean_direct)" in BL.BLEND_FORMULA
    cols = BL.mean_feature_columns(["h", "v_lag0", "rec26", "rec_mean",
                                    "ace_speed_kms_last", "ch_x"])
    assert "h" not in cols and not [c for c in cols if c.startswith("rec")]
    assert cols[-4:] == list(BL.TRAJ_COLUMNS)
    import runner as R
    assert set(R.PROBES) == {"dev", "selection", "audit"}
    for iso in R.PROBES.values():
        o = pd.Timestamp(iso)
        assert o.tz is not None and (o.hour % 3 == 0) and o.minute == 0
    assert callable(R.poison_check)
    assert "positive_control" in inspect.getsource(R.poison_check)
    assert "oof_weight" in inspect.getsource(R.main) and "weights.json" in inspect.getsource(R.main)


def test_output_contract_helpers():
    import runner as R
    origins = pd.DatetimeIndex(["2025-09-15T00:00:00Z", "2025-09-15T03:00:00Z"])
    g = R.base_grid(origins)
    assert len(g) == 2 * 72 and not g.duplicated(["origin", "h"]).any()
    assert sorted(g["h"].unique()) == list(range(1, 73))
