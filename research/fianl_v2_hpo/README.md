# fianl_v2_hpo — fianl_v2 with tuned XGBoost parameters

`research/fianl_v2` (4f0b646) unchanged except the A-stage XGBoost parameters, which come from a 20-trial random
search on two internal 2025 folds (trial list below). jsy301 local run, xgboost 2.1.4, CPU.

| parameter | fianl_v2 | fianl_v2_hpo (trial 14) |
|---|---|---|
| max_depth | 3 | 2 |
| learning_rate | 0.03 | 0.0465 |
| A-stage rounds | 600 | 300 |
| min_child_weight | 20 | 6.72 |
| reg_lambda | 5 | 8.04 |
| subsample | 0.8 | 0.836 |
| colsample_bytree | 0.7 | 0.894 |

The S-cap continuation (6 x 30 trees), prior-model offset, seam xf8, 3 h phase advance and the 3-seed mean are as
in fianl_v2.

## Search (pre-registered before any trial ran)

- **Folds.**
  - V1 scores targets 2025-07-01..10-01 and V2 scores targets 2025-10-01..12-26 (UTC, end exclusive).
  - Training rows of a fold are fit_F1 rows with origin + 144 h <= fold start - 72 h. The board's rule is applied
    inside 2025.
  - Every fold target ends before the F1 cutoff 2025-12-29, so no 2026 data and no board truth was used.
- **Score.** The official lead-first S is computed on 6-hourly origins and unfilled observed speeds. It uses seed 42
  only, without the prior/offset step, with seam + phase advance applied.
- **Trials.**
  - Trial 0 is the fianl_v2 defaults. Trial 1 is the winner of an earlier fianl_v1 search (depth 2, 450 rounds).
  - The other 18 are random draws (rng 20261127). Depth is drawn from {1, 2}, rounds from {300, 450, 600, 900, 1200},
    learning_rate from 0.015-0.06 (log), min_child_weight from 5-80 (log), reg_lambda from 1-20 (log),
    subsample from 0.6-1.0 and colsample from 0.5-0.9.
- **Selection.** The lowest mean of V1 S and V2 S wins.

| trial | depth | rounds | mean S | V1 | V2 | vs trial 0 |
|---|---|---|---|---|---|---|
| 14 (selected) | 2 | 300 | 97.85 | 79.51 | 116.19 | -1.77 % |
| 11 | 2 | 300 | 98.24 | 80.33 | 116.15 | -1.38 % |
| 19 | 2 | 300 | 98.53 | 81.42 | 115.64 | -1.09 % |
| 1 (fianl_v1 winner) | 2 | 450 | 98.94 | 80.26 | 117.63 | -0.67 % |
| 0 (fianl_v2) | 3 | 600 | 99.62 | 81.19 | 118.05 | 0 |
| depth-1 trials (9) | 1 | 450-1200 | 101.0-105.3 | | | +1.4..+5.7 % |

**Caveats.**
- The selected trial is the best of 20 on one seed, so expect shrinkage. The earlier fianl_v1 winner fell from
  -2.39 % on its own search to -0.67 % here.
- The gain comes from h49-72 (V1 block S 98.5 -> 90.8).
- h1-24 is slightly worse, RMSE is worse (V1 65.9 -> 68.5) and the mean level is higher (V1 bias +3.4 -> +12.2).
  The improvement is in the tail (TIMSE) term.

All 20 trials (drawn once, before any fit):

| trial | depth | lr | rounds | mcw | lambda | subsample | colsample |
|---|---|---|---|---|---|---|---|
| 0 | 3 | 0.0300 | 600 | 20.00 | 5.00 | 0.800 | 0.700 |
| 1 | 2 | 0.0384 | 450 | 9.25 | 5.23 | 0.711 | 0.686 |
| 2 | 2 | 0.0187 | 450 | 19.22 | 2.62 | 0.975 | 0.581 |
| 3 | 2 | 0.0427 | 900 | 15.64 | 4.89 | 0.658 | 0.626 |
| 4 | 1 | 0.0180 | 1200 | 14.15 | 1.66 | 0.893 | 0.853 |
| 5 | 1 | 0.0326 | 600 | 69.75 | 1.58 | 0.754 | 0.748 |
| 6 | 1 | 0.0171 | 900 | 24.37 | 1.33 | 0.643 | 0.861 |
| 7 | 2 | 0.0195 | 450 | 44.72 | 1.16 | 0.784 | 0.605 |
| 8 | 2 | 0.0474 | 450 | 8.43 | 8.95 | 0.786 | 0.796 |
| 9 | 1 | 0.0202 | 1200 | 9.31 | 2.58 | 0.790 | 0.873 |
| 10 | 1 | 0.0442 | 600 | 51.10 | 2.81 | 0.675 | 0.872 |
| 11 | 2 | 0.0486 | 300 | 65.82 | 1.69 | 0.858 | 0.888 |
| 12 | 2 | 0.0438 | 1200 | 6.90 | 2.94 | 0.671 | 0.575 |
| 13 | 1 | 0.0162 | 450 | 60.25 | 9.38 | 0.789 | 0.524 |
| 14 | 2 | 0.0465 | 300 | 6.72 | 8.04 | 0.836 | 0.894 |
| 15 | 1 | 0.0578 | 600 | 7.41 | 3.60 | 0.677 | 0.737 |
| 16 | 2 | 0.0453 | 600 | 57.46 | 3.29 | 0.864 | 0.625 |
| 17 | 1 | 0.0162 | 1200 | 30.41 | 3.77 | 0.901 | 0.637 |
| 18 | 1 | 0.0563 | 1200 | 27.72 | 4.55 | 0.709 | 0.767 |
| 19 | 2 | 0.0238 | 300 | 8.56 | 3.61 | 0.760 | 0.636 |

## Data and patches

- **Data.** Release `data-v2-df-20261125` as of 2026-11-26 06:44 UTC (457 columns, post-fetch raw).
  - Its fianl_v2 columns are bit-identical to the 55-column build on every train and query row.
  - Live rows were dropped and query y was nulled before training.
- **Changes to `research/fianl_v2/experiment.py`** (marked `LOCAL PATCH` / `HPO trial 14`):
  - the HPO parameters above;
  - `fill_training_labels` builds the target timeline from training rows only. Training labels are identical,
    and query y is never read.
  - The run never downloads observations or scores. It writes the unscored per-seed forecasts, and the replay script
    applies the fianl_v2 post-processing.

## Reproduce

Environment of the submitted run: Python 3.12.14, numpy 2.4.6, pandas 2.3.3, pyarrow 25.0.1, xgboost 2.1.4 (CPU),
scipy 1.15.3. Data: the release asset `df.parquet` of `data-v2-df-20261125` uploaded 2026-11-26 06:44 UTC,
235,272,537 bytes, sha256 `085683704e2f67894f08e4ea96d7a970772ffa3d82faf51218f72629d0da17d3`. It can be used as
downloaded: live rows are ignored and query y is never read.

1. **Board forecast.** This trains all seeds and periods with this experiment.py, then applies seam xf8 -> 3 h phase
   advance per seed and the 3-seed mean. The output should equal the submitted file exactly.
   - Determinism check. A fresh clone of this commit, run on the downloaded asset as is at 30 threads, wrote
     weights byte-identical to the submitted run (10 threads, live rows dropped). This was checked for seed 42
     F1/F2/F3 and seed 43 F1.
   - Seed 42 F1 at 16 threads was also identical.
   - The replay post-processing reproduces the submitted file with difference 0.
   - Runtime: use 8-10 threads. Each fit has only ~30k rows, and more threads is slower: at 30 threads one
     seed-period took ~15 min, so all 9 seed-periods take ~2 h 15 m. At 10 threads a fit takes 60-90 s.

   ```sh
   python scripts/jsy301/fianl_v2_hpo_t14/34c8893d8662f4b75dfaee9d38a971ae0d73c5b3c4f6316cd21a720f2e951b62.py \
       --data df.parquet --out OUT --threads 8
   ```

2. **The search.** `scripts/jsy301/fianl_v2_hpo_t14/hpo.py` is the fold-scoring harness. The research layout keeps
   only experiment.py and README.md, so it lives there.

   ```sh
   python hpo.py make-trials-shallow RUN       # writes RUN/trials.json, the 20 trials above (rng 20261127)
   CUDA_VISIBLE_DEVICES='' python hpo.py work RUN --data df.parquet \
       --experiment research/fianl_v2_hpo/experiment.py --threads 8 [--worker K]   # run several workers in parallel
   python hpo.py report RUN                    # the table above (fold S, bias, change vs trial 0)
   ```

   - `work` reads training rows only and scores the official lead-first S on unfilled training y inside the folds.
     Workers claim trials through lock files, so any number of them can run side by side.
   - A full run took 1 h 27 m on 4 workers x 8 threads.
   - Any single trial can be trained on the board periods with `FIANL_HPO_TRIAL='{...json...}'`.
