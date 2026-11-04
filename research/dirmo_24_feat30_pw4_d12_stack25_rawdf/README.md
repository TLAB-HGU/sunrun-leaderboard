# dirmo_24_feat30_pw4_d12_stack25_rawdf (jsy301)

`dirmo_24_feat30_pw4_d12_rawdf` (team DirMO-24 feat30 recipe with 4x loss weight on observed speed >= 550 km/s,
leads 1-12 from a delta-target block) plus one stage: at leads 25-72 the forecast moves by 0.25 x a level-free
LightGBM residual model, the stack of our earlier entry C rebuilt on the team raw-only df
(`data-v2/input/df.parquet`, rebuilt from `data-v2/raw` on 2026-11-02). `experiment.py` is the full model and has
the same code as the submitted replay script.

## Model

- **Main model:** three 24-lead vector-leaf XGBoost blocks on the Top-30 inputs (21 constant + 24 x 9 lead-varying =
  237 columns), 900 rounds, depth 3, lr 0.03, mean of seeds 42 / 1 / 2, 4x@550 masked loss.
- **Short model (leads 1-12):** the same block-0 model trained on `y - v_lag0` (weights on the speed itself);
  forecast `v_lag0 + prediction`.
- **Residual stack (leads 25-72):**
  1. Out-of-fold forecasts of the main recipe (seed 42) on the period's training rows: 4 contiguous folds, each
     predicted by a model fit on the other rows with a 72 h purge on both sides.
  2. LightGBM (300 trees, 7 leaves, at least 2000 rows per leaf, lr 0.03, L2 10) learns `y - oof` at leads 25-72
     (rows without truth left out) from 24 level-free columns per (origin, lead): forecast path shape (argmax, std,
     6 h change, change from lead 1, distance to its own max), recurrence range and timing, 24 h range and 6 h change
     of the latest speed, ACE density / temperature / |B|, SUVI coronal-hole areas, ENLIL arrival inputs, lead, and
     three level differences (recurrence - forecast, latest speed - forecast, recurrence max - forecast max).
     Absolute speed levels are left out, so the model reshapes the forecast rather than moving its level.
  3. The main forecast at leads 25-72 moves by 0.25 x the model's prediction (kappa fixed in the lab before this
     df, not tuned here), clipped to [1, 3000]. The short model then replaces leads 1-12.
- **Inputs:** filled causally along the hourly origin timeline (observations last value, images <= 24 h, ENLIL
  none). **Targets:** never filled; missing hours get weight 0. **Training rows:** the df's `fit_F1/F2/F3` flags.

## Why

The level model's 25-72 h forecast is too flat and too slow where a high-speed stream's shape is visible in its
inputs (recurrence, coronal holes, the forecast path itself). A small model on level-free inputs corrects the shape
without moving the level, which in our earlier board tests did not transfer from the lab.

## Lab evidence before submission (pre-registered, dirmo_lab round ND)

- **Data:** the same raw-only df, inputs filled as above; truth = df `y` (raw SWEPAM status 0), capped at 2025-12-31.
- **Windows:** 2023-07..12, 2024-01..06, 2024-07..12, 2025-01..06, 2025-07..12, each trained only on earlier rows.
- **Score:** the board metric S on 6-hourly origins; xgboost 3.4.1; mean of 3 seeds. CI = 95 % block bootstrap.

| arm (vs 4x model B) | all leads [CI] | high-speed (2025) | quiet | per window | bias change high / quiet |
|---|---|---|---|---|---|
| D12 alone | -1.42 % [-2.48, -0.48] | -0.52 % | -1.96 % | -0.1 .. -3.4 % | +0.28 / -0.10 |
| stack alone | -1.78 % [-2.56, -1.08] | -1.51 % | -2.12 % | -1.1 .. -4.0 % | +1.74 / -1.76 |
| **this entry** | **-3.20 % [-4.52, -1.88]** | **-2.03 %** | **-4.08 %** | **-1.7 .. -4.9 %** | **+2.02 / -1.86** |

The pre-registered final gate (all leads <= -1 % with CI < 0, both regimes <= 0, no window > +1 %, bias change
<= 2 km/s per regime, high-speed 49-72 h within +0.5 %) passes except the bias check, by 0.02 km/s in the high-speed
windows. Both bias changes point toward zero bias (4x model bias -10.4 high-speed, +22.9 quiet). Submitted at the
user's request.

Caveats:

- **Lab gains have not transferred well:** D12's lab -1.42 % was -0.09 % on the board against the team's fresh 4x.
- **Library:** the lab used xgboost 3.4.1 with a per-lead base_score; this entry runs 2.1.4 with a scalar
  base_score per block (as D12). The stack's inputs were checked equal to the lab's on 3000 lab origins.
- **Single-seed OOF:** the residual model is fit on seed-42 out-of-fold forecasts and applied to the 3-seed mean.
- **No board selection:** the entry was not chosen by any board number.
