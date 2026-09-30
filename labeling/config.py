"""Frozen labeling constants. Do not change after the final evaluation folds are published.

THRESHOLDS are the per-axis medians of T/S/F over ALL 2,833 final folds (targets
2026-06-01 00:00 .. 2026-09-29 23:00 UTC): T 0.762, S 0.174, F 0.472, rounded to 3 decimals.
FROZEN on 2026-09-30 -- do not re-derive.
A fold is "high" on an axis when its value is strictly greater than the threshold.
"""
EVAL_START = "2026-06-01"
EVAL_END = "2026-09-30"  # exclusive, UTC: last target 2026-09-29 23:00 (fixed by the team on 2026-09-30)
THRESHOLDS = {"T": 0.762, "S": 0.174, "F": 0.472}
