# dirmo_24_feat30_pw4_d12_rawdf (jsy301)

The team DirMO-24 feat30 recipe with 4x loss weight on observed speed >= 550 km/s (seongeun
`e2000fs_dirmo_24_feat30_pointweight4`), trained on the team raw-only df (`data-v2/input/df.parquet`, rebuilt from
`data-v2/raw` on 2026-11-02), with one change: leads 1-12 come from a second block-0 model whose target is the change
from the latest observed speed (`y - v_lag0`). `experiment.py` is the full model and has the same code as the
submitted replay script.

## Model

- **Main model (leads 13-72):** three 24-lead vector-leaf XGBoost blocks on the Top-30 inputs (21 constant + 24 x 9
  lead-varying = 237 columns), 900 rounds, depth 3, lr 0.03, mean of seeds 42 / 1 / 2.
- **Short model (leads 1-12):** the same block-0 model (same inputs, rounds, seeds), trained on `y - v_lag0`; the
  4x weight is computed on the speed `y` itself, not on the change. The forecast is `v_lag0 + prediction`, clipped to
  [1, 3000].
- **Inputs:** filled causally along the hourly origin timeline. Solar-wind observations carry the last valid value
  (gaps in the df are at most 30 h); image-derived columns (`suvi_*`, `strip_*`, `img_age_h`) at most 24 h, with
  `img_age_h` growing by the hours carried; ENLIL columns are not filled.
- **Targets:** never filled. Missing hours get weight 0 (masked squared error, weights mean 1 over observed rows per
  lead); an origin with no observed target in a block is not a training row.
- **Training rows:** the df's `fit_F1` / `fit_F2` / `fit_F3` flags (origin + 144 h <= period cutoff).

## Why

A level model has to rebuild the current speed from its inputs at every lead; at 1-12 h the speed changes little, so
a target that starts from the latest observed speed leaves less to learn. In the lab the gain is largest during fast,
rising wind (the May 2024 storm), which the TIMSE half of the score weights most.

## Lab evidence before submission (pre-registered, dirmo_lab round RD)

- **Data:** the same raw-only df, inputs filled as above; truth = df `y` (raw SWEPAM status 0), capped at
  2025-12-31.
- **Windows:** 2023-07..12, 2024-01..06, 2024-07..12, 2025-01..06, 2025-07..12, each trained only on earlier rows.
- **Score:** the board metric S on 6-hourly origins; xgboost 3.4.1; mean of 3 seeds. CI = 95 % block bootstrap
  (72 h blocks within windows).

| arm (vs 4x model B) | leads 1-12 | all leads [CI] | high-speed windows (2025) | quiet windows | worst window |
|---|---|---|---|---|---|
| unweighted recipe | -1.06 % | +0.72 % | +2.80 % | -2.25 % | +3.76 % |
| **this entry (D12)** | **-14.8 % [-24.3, -5.2]** | **-1.42 % [-2.48, -0.48]** | **-0.52 %** | **-1.96 %** | **-0.10 %** |

The final gate (all leads <= -1 % with CI < 0, both regimes <= 0, no window > +1 %, bias change <= 2 km/s per regime,
high-speed 49-72 h within +0.5 %) passes; bias change is +0.28 (high-speed) / -0.10 km/s (quiet).

Caveats:

- **Concentrated gain:** about half of the 1-12 h gain is the May 2024 storm window (2024-H1, -30 % at 1-12 h); the
  other windows give -1 % to -6 % at 1-12 h.
- **Library:** the lab used xgboost 3.4.1, this entry runs 2.1.4 with a scalar `base_score` per block (the lab used a
  per-lead vector). On one window the short block differs from the lab by 1.6 km/s mean absolute, half the
  seed-to-seed difference (3.1 km/s).
- **Stale anchor:** `v_lag0` is missing in the raw data at about 30 % of origins and is then the last observed value
  (up to 30 h old); the rate is the same in training and query rows.
- **No board selection:** the entry was not chosen by any board number.
