# dirmo_24_feat30_pw4_d12_stack25_r300_rawdf (jsy301)

`dirmo_24_feat30_pw4_d12_stack25_rawdf` with one change: the main model's blocks for leads 25-48 and 49-72 use
300 boosting rounds instead of 900, in the main fit and in the residual stack's cross-fit alike (config
`block_rounds`). Everything else (leads 1-24, the delta short model, the stack's LightGBM, kappa 0.25) is unchanged.
`experiment.py` is the full model and has the same code as the submitted replay script.

## Model

- **Main model:** three 24-lead vector-leaf XGBoost blocks on the Top-30 inputs (21 constant + 24 x 9 lead-varying =
  237 columns), depth 3, lr 0.03, mean of seeds 42 / 1 / 2, 4x@550 masked loss; **900 rounds on leads 1-24, 300 on
  leads 25-72**.
- **Short model (leads 1-12):** the block-0 model trained on `y - v_lag0` (900 rounds); forecast `v_lag0 + prediction`.
- **Residual stack (leads 25-72):** as the parent entry. Out-of-fold forecasts (seed 42, 4 contiguous folds, 72 h
  purge) of the main recipe, now with 300 rounds on blocks 25-72, train a level-free LightGBM on `y - oof`; the
  forecast at leads 25-72 moves by 0.25 x its prediction. The short model then replaces leads 1-12.
- **Inputs / targets / training rows:** as the parent (causal fill, targets never filled, `fit_F1/F2/F3` flags).

## Why

Several of the board's top entries train fewer rounds (the team SERA entries use 300) or stop early (the TCN), and
forecast higher at long leads. In the lab, 300 rounds on leads 25-72 helped only together with the stack refit on
the new base; without it the 300-round blocks were worse.

## Lab evidence before submission (pre-registered, dirmo_lab round TC2)

- **Data / windows / score:** as the parent's round ND (same raw-only df, truth capped at 2025-12-31, five windows
  2023-07..2025-12, board metric S on 6-hourly origins, 95 % block bootstrap). The 300-round blocks used new seeds
  7 / 8 / 9 in the lab.

| arm (vs the parent entry DS) | all leads [CI] | high-speed (2025) | quiet | per window |
|---|---|---|---|---|
| **300 rounds + stack refit (this entry)** | **-1.05 % [-2.12, -0.36]** | **-1.27 %** | **-0.77 %** | -1.68 .. +0.03 % |
| same, vs DS shifted up by the same mean amount | -0.83 % [-1.87, -0.11] | -0.74 % | -0.80 % | |
| 300 rounds without the stack refit | +0.31 % [-0.81, +1.39] | -0.22 % | +1.01 % | |

All six pre-registered checks pass; the main one thinly (-1.05 % vs the -1.0 % limit).

Caveats:

- **Screened before:** the 300 count follows the team top's recipe and was first screened on the same windows
  (round TC lane 3, 3 arms); the TC2 seeds were new.
- **Level shift on the board periods:** leads 25-48 / 49-72 sit +5.6 / +8.9 km/s above the parent entry (lab mean
  about +2.8), so on the board more of the effect may be a plain upward shift.
- **Lab gains have not transferred well:** four lab gains under 2 % shrank on the board (the parent: lab -1.78 % for
  the stack, board -0.24 %).
- **No board selection:** the entry was not chosen by any board number.

## Folder layout in this repository (2026-11-09)

- `experiment.py` — the submitted script, unchanged (single file; data → training → inference).
- `results/predictions.parquet`, `results/score.json` — the stored leaderboard prediction and its official score record (`tlabtlab/sunrun-lb-store`).
- `logs/leaderboard.jsonl` — the stored submission record as JSON lines.
- `weights/` — empty. The trained models and run logs of this entry exist only on jsy301's machine (none on this host or golab2); a rerun of `experiment.py` recreates them.
