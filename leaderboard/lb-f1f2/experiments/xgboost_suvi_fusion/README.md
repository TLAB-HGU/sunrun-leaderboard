# XGBoost SUVI fusion experiment

This experiment forecasts ACE solar-wind speed 1–72 hours ahead with one direct
XGBoost model per horizon. Every origin uses exactly the trailing 120 hourly ACE
slots plus GOES-18 SUVI Fe195 disk summaries. SUVI pixels outside `0.98 * RSUN`
or with `DQF != 0` are excluded.

The fixed split is:

- train targets through `2025-07-31T23:00:00Z`;
- validation targets from August 2025 through February 2026;
- one untouched holdout gate from March through May 2026;
- final leaderboard folds at dataset revision
  `bcf5d5417d99eaa331ddca40581d48953e09a58e`.

Use XGBoost 2.1.4 on this host. Newer 3.4.1 Linux wheels require a newer CUDA
driver and silently fall back to CPU; the CLI probes the selected device and
fails closed if that happens.

```bash
python experiments/xgboost_suvi_fusion/run.py \
  --mode features \
  --ace /home/t-lab01/.local/state/sundb/stage/ace_solar \
  --suvi-meta /home/t-lab01/.local/state/sundb/stage/suvi_hourly \
  --images-root /home/t-lab01/sunrun/sun-img/data \
  --feature-cache store/suvi-fusion/suvi-hourly-v2.parquet \
  --output store/suvi-fusion/origin-features-v2.parquet \
  --image-workers 12

python experiments/xgboost_suvi_fusion/run.py \
  --mode tune --trials 60 --device cuda:0,cuda:1 \
  --ace /home/t-lab01/.local/state/sundb/stage/ace_solar \
  --suvi-meta /home/t-lab01/.local/state/sundb/stage/suvi_hourly \
  --images-root /home/t-lab01/sunrun/sun-img/data \
  --feature-cache store/suvi-fusion/suvi-hourly-v2.parquet \
  --origin-features store/suvi-fusion/origin-features-v2.parquet \
  --storage sqlite:///store/suvi-fusion/optuna.sqlite3 \
  --output store/suvi-fusion/tuned.json
```

`final` refits the frozen validation winner once through February and stops if
the holdout does not improve both MSE and peak behavior over the same-config
ACE-only ablation. Only a passing holdout is refit through May and evaluated
against the official folds and current leader. The CLI writes submission
artifacts but never uploads them.

## Coronal-hole ablation (24-hour budget)

`ch_experiment.py` measures connected dark-region candidates at relative
radiance thresholds 0.35/0.45/0.55. It records deprojected area, centroid,
extent, equatorial/central-meridian area and 12×3 heliographic area cells.
These are single-channel CH candidates, not confirmed magnetic coronal holes.
The three independent XGBoost studies use ACE-only, CH-only and ACE+CH numeric
features; CH-only inference has no ACE inputs. All inputs are trailing 120h.

```bash
python experiments/xgboost_suvi_fusion/ch_experiment.py \
  --ace /home/t-lab01/.local/state/sundb/stage/ace_solar \
  --suvi-meta /home/t-lab01/.local/state/sundb/stage/suvi_hourly \
  --images-root /home/t-lab01/sunrun/sun-img/data \
  --ace-origin-features store/suvi-fusion/origin-features-v2.parquet \
  --folds /absolute/path/to/pinned/folds.parquet \
  --leader-predictions /absolute/path/to/pinned/leader.parquet \
  --output store/ch-v1 --budget-hours 24 --max-trials 24 \
  --devices cuda:0,cuda:1 --image-workers 12
```

The pinned `truth.parquet` must be next to `folds.parquet`; local targets and
missingness must match it exactly. The driver verifies the leader score and
stores input hashes before launching any candidates. The output directory is
locked to one coordinator. Repeat the exact command to resume; the original
deadline and sampler RNG are retained. Interrupted trials are marked failed,
not completed. A mismatched sampler/database checkpoint fails closed.

`--feature-set all` is the default and runs the full three-arm comparison.
`--feature-set ace_only`, `ch_only` or `ace_ch` runs isolated screening in a
separate output directory, without submission eligibility. Image extraction
checkpoints every 1,000 frames. A completed pre-December prefix can start CH
screening while later images are still processed; the complete table replaces
the prefix atomically. ACE full-horizon preselection also overlaps extraction.

Screening uses six horizons and August–November 2025 targets. Each arm gets up
to 24 candidates; comparable candidates are restricted to the common completed
count. Both GPUs can screen ACE candidates while the CPU builds CH features.
Up to three candidates per arm are then evaluated over all 72 horizons on
December–February targets, with early stopping still restricted to August–
November. Peak/rise training weights are capped at five.

The candidate and per-arm tree counts are frozen before the March–May holdout.
No model is switched after seeing holdout results. A CH-containing candidate
must beat the separately tuned ACE-only holdout MSE and pass the event gate
before official June–September evaluation. Official success requires MSE at
most half the pinned leader's MSE plus the event gate. This driver writes
artifacts and eligibility; publication remains a separate verified step.

Events are >=600 km/s runs merged across <=6h quiet gaps. Events touching either
edge of a 72h window are censored. Evaluation uses daily 00 UTC origins for
development/holdout and supplied official origins for the leaderboard. Windows
with missing observations use the last past observed speed, per the user's
subsequent instruction. No future interpolation is used; filled-slot fractions
and genuinely observed peak support are reported. Strict complete-window
evaluation remains available through `event_metrics(..., missing_policy="strict")`. Peak
matching maximizes one-to-one matches within ±12h, then minimizes timing error.
The gate requires precision and recall >=0.60, matched peak-height MAE <=60
km/s, and at least ten unique observed peaks. Repeated origins count as forecast
decisions, while event support is counted by distinct actual peak timestamps.
Seven-day block bootstrap intervals address dependence between overlapping
windows. The old 24/48/72h high-speed classification diagnostics remain separate.

Inspect `run-state.json` and `run.log` for progress; `hpo/screen-progress.json`
contains completed trials. `masks/` contains 24 training-only mask overlays.
`frozen.json`, `holdout.json`, forecast Parquets, curve SVGs and `report.json` /
`report.md` preserve final comparisons. Budget exhaustion or incomplete stages
do not grant submission eligibility. RAM below 8 GiB available or GPU memory
above 85% pauses new assignments. Deadlines are checked between horizons; a
running bounded tree fit can finish slightly after its phase deadline.
