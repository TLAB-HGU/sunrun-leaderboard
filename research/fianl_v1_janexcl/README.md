# fianl_v1_janexcl — fianl_v1 without a January 2026 training window

Same model as `research/fianl_v1` (origin/main 4877593, S3-only ENLIL, no onset700_hazard, no uplift), retrained
locally (xgboost 2.1.4, 8 threads, CPU) with one training window removed from the **F2 and F3** fits.
F1 trains on origins up to 2025-12-23, so its fit and forecast are unchanged.

| Entry | Removed training origins | Targets it removes |
|---|---|---|
| `fianl_v1_excl_jan19_sensor` (A) | 2026-01-16 10:00 .. 01-19 20:00 UTC, 83 origins | 01-19 10-21 UTC: a 9 h flat 287-291 km/s segment right after a 125 km/s drop in 2 h, read as ACE SWEPAM contamination (not verified) |
| `fianl_v1_excl_jan19_21_sensor_storm` (B) | 2026-01-16 10:00 .. 01-21 11:00 UTC, 122 origins | A plus the 1,080 km/s shock and its decay to 657 km/s (01-19 22 UTC .. 01-21 12 UTC) |

An origin is removed when any of its 72 targets falls in the window (`fit_F2`/`fit_F3` set false, all 72 rows).
Origins after the window stay. Offset windows (last 336 calibration origins: F2 03-09..03-23, F3 06-09..06-23)
do not contain the window.

Changes to `research/fianl_v1/experiment.py` (marked `LOCAL PATCH`):
- `fill_training_labels` builds the target timeline from training rows only, so query y is never read
  (training labels are identical because the training target times are contiguous);
- the run never downloads observations or scores: it always writes the unscored per-seed forecasts, and
  `finalize.py` applies fianl_v1's post-processing (seam xf8 -> 3 h phase advance -> mean of 3 seeds).

Reproduce: `CUDA_VISIBLE_DEVICES='' python run.py --variant A --data data-v2/input/df.parquet --out OUT`.
The submitted files were built from a base run (uncut table, F1) plus the cut runs (F2/F3); F1 uses the same
fit rows and data either way.

Prediction-only comparison to the uncut retrain (3-seed means, no truth used): mean shift -0.04..+1.3 km/s;
same-seed cell change 5.3-6.6 km/s against 5.1-5.6 km/s between uncut seeds; where the uncut forecast is
>= 600 km/s the cut forecasts are 1.5-4.8 km/s higher.
