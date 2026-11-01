"""Frozen labeling constants. Do not change after the final evaluation folds are published.

THRESHOLDS are the per-axis medians of T/S/F over ALL 2,833 final folds (targets
2026-06-01 00:00 .. 2026-09-29 23:00 UTC): T 0.762, S 0.174, F 0.472, rounded to 3 decimals.
FROZEN on 2026-09-30 -- do not re-derive.
A fold is "high" on an axis when its value is strictly greater than the threshold.
"""
EVAL_START = "2026-06-01"
EVAL_END = "2026-09-30"  # exclusive, UTC: last target 2026-09-29 23:00 (fixed by the team on 2026-09-30)
THRESHOLDS = {"T": 0.762, "S": 0.174, "F": 0.472}

# 2026-10-31 policy: three scored quarters, [start, end) on target time (replaces the 2026-10-26 F1 = 03..05 /
# F2 = 06..09 split). Folds straddling a boundary (the 71 origins before each start) belong to none.
# THRESHOLDS above stay frozen (derived on the old F2 folds 2026-06-01..09-29).
PERIODS = {"F1": ("2026-01-01", "2026-04-01"), "F2": ("2026-04-01", "2026-07-01"), "F3": ("2026-07-01", EVAL_END)}
# exclusive training cutoffs (period start - 72h): predictions for a period use models trained on data before it
F1_TRAIN_END = "2025-12-29T00:00:00Z"
F2_TRAIN_END = "2026-03-29T00:00:00Z"
F3_TRAIN_END = "2026-06-29T00:00:00Z"
# targets of the pre-2026-10-31 evaluation (old F1 + old F2); used to tell legacy predictions from full ones
LEGACY_EVAL_START = "2026-03-01"
