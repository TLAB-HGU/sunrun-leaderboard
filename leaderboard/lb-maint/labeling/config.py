"""Frozen labeling constants. Do not change after the final evaluation folds are published.

THRESHOLDS are the per-axis medians of T/S/F over ALL 2,833 final folds (targets
2026-06-01 00:00 .. 2026-09-29 23:00 UTC): T 0.762, S 0.174, F 0.472, rounded to 3 decimals.
FROZEN on 2026-09-30 -- do not re-derive.
A fold is "high" on an axis when its value is strictly greater than the threshold.
"""
EVAL_START = "2026-06-01"
EVAL_END = "2026-09-30"  # exclusive, UTC: last target 2026-09-29 23:00 (fixed by the team on 2026-09-30)
THRESHOLDS = {"T": 0.762, "S": 0.174, "F": 0.472}

# 2026-10-26 policy: two scored periods, [start, end) on target time. Folds straddling the
# boundary (origins 2026-05-29..05-31) belong to neither. THRESHOLDS above stay frozen for both.
PERIODS = {"F1": ("2026-03-01", "2026-06-01"), "F2": (EVAL_START, EVAL_END)}
F1_TRAIN_END = "2026-02-26T00:00:00Z"  # exclusive: F1 predictions use models trained on data before this (F1 start - 72h)
