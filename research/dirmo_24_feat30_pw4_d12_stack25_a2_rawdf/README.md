# dirmo_24_feat30_pw4_d12_stack25_a2_rawdf (jsy301)

`dirmo_24_feat30_pw4_d12_stack25_rawdf` (DS; board Total S 100.289) followed by **one post-processing step with no
fitted parameters** on leads 25-72. The submitted file is the DS submission plus the step; no model was retrained to
make it. `experiment.py` is the full model (DS recipe + the step) and has the same code as the submitted replay script.

## What is submitted

- **The submitted predictions are the DS entry's submitted predictions.parquet with the step applied row by row.**
  Leads 1-24 are bit-identical to DS; only leads 25-72 change. So the board difference to DS is exactly the effect of
  the step (no retraining noise).
- The replay script retrains DS (same seeds 42 / 1 / 2, same df split, same code) and applies the step; its full run
  reproduced the submitted file to max |diff| 1.1e-13 km/s (454,680 rows; the DS retrain gave the same values
  as the 11-04 run, xgboost 2.1.4, 8 threads).

## The step (A2, "4-knot line")

For each final 72-lead forecast row `f` (after the stack and the short model), on its own:

1. knots = {lead 1, argmin(f), argmax(f), lead 72} over the whole row (duplicate knots merged);
2. `line` = straight lines through `f` at those knots (`np.interp`);
3. output = `f` on leads 1-24, `f + ((h - 24) / 6) (line - f)` on leads 25-30, `line` from lead 30; clip 1-3000.

No other row, no truth, no parameter. The row's own minimum and maximum values and times are kept; the rest of
leads 25-72 becomes the straight line between them. On the three board periods the step changes 96 % of the cells
on leads 25-72 by 8.4 / 12.7 / 7.0 km/s mean absolute (leads 25-30 / 31-48 / 49-72); the clip never acts.

## Why

An earlier analysis of 72 h forecast shapes (dirmo_lab, 2026-11-05) found that the values at the turning points of
a forecast matter for the metric and the wiggles between them mostly do not. Replacing the between-knot path of DS
by straight lines removed some of the error DS makes between its turning points in the lab.

## Lab evidence before submission (pre-registered, dirmo_lab round SV arm A2)

- **Data / windows / score:** as the DS entry's round ND (the same raw-only df, raw truth capped at 2025-12-31, five
  windows 2023-07..2025-12, board metric S on 6-hourly origins, block bootstrap; 10 formal arms, CI 99.5 %).

| arm (vs DS) | all leads [CI 99.5] | windows | after removing the mean level change |
|---|---|---|---|
| **A2 (this step)** | **-1.23 % [-2.34, -0.22]** | 5 / 5 better | -0.79 % |

A2 was the best of the ten arms but **failed that round's -2 % gate**, so round SV adopted nothing.

## Why submit it then: a board-transfer measurement (round BT, thresholds fixed before submission)

Five earlier lab gains under 2 % shrank to about zero on the board (MW, HW1, D12, DS, TC2). All of them mixed
retraining or loss-weight changes. This entry is the first pure post-processing step we put on the board, so its
board score tells whether a parameter-free lab gain transfers at all. Pre-registered reading of Total S vs DS
100.289: Δ ≤ -0.6 % "transfers", Δ ≥ -0.2 % "does not", between = undecided. F1 / F2 / F3 are recorded as well.

Caveats:

- **Not chosen by any board number;** the step and the thresholds were fixed before this submission.
- **Lab gains have not transferred well** so far (see above); the lab gain here is also small.
- The exploratory lab numbers in dirmo_lab LOG.md (A2 combined with other arms) are hypotheses, not part of this entry.

## Folder layout in this repository (2026-11-09)

- `experiment.py` — the submitted script, unchanged (single file; data → training → inference).
- `results/predictions.parquet`, `results/score.json` — the stored leaderboard prediction and its official score record (`tlabtlab/sunrun-lb-store`).
- `logs/leaderboard.jsonl` — the stored submission record as JSON lines.
- `weights/` — empty. The trained models and run logs of this entry exist only on jsy301's machine (none on this host or golab2); a rerun of `experiment.py` recreates them.
