# Fixed-forecast residual audit

## Baseline and reproduction

The user-confirmed W is the current official **S** (`lead-timse-6h-v1`). Primary fixed single forecast: `research/drop035_v650_confirm_20261108/runs/drop035_v650_s42`, reproduced **97.492442080600**; 50% target **48.746221040300**. This replay does not retrain a model. The original A+B1 is a secondary scientific reference, not the denominator of the goal.

|Forecast|Total S|F1|F2|F3|
|---|---:|---:|---:|---:|
|Primary seed42|97.492442|119.90588|74.82322|60.66323|
|Same configuration seed43|97.873708|120.71515|74.39999|60.71181|
|Same configuration seed44|98.074513|120.99950|74.42036|60.59499|
|Official causal persistence|149.383639|171.68622|122.39621|107.62976|
|Input v_lag0 persistence|150.936705|173.56013|124.66306|108.73893|
|Input rec27 recurrence|142.949916|153.66820|135.08395|85.77368|

Each forecast has zero official prediction replacements. Same-configuration S mean **97.813555**, sample variance **0.087416**, SD **0.295661**, worst **98.074513**; prediction arrays are never averaged. Mean official lead-averaged RMSE **67.636085**; pooled valid-pair MAE **51.422303**. Pooled RMSE elsewhere in this diagnostic differs from the official lead-averaged RMSE by construction.

The unchanged root scorer defines and aligns the 1,052 six-hour origins and 72 leads. There are 52,619 scored origin/lead pairs. Exact resampling-kernel values agree with freshly computed official S/TIMSE/RMSE/TRMSE at atol 1e-8, rtol 1e-10. Outputs preserve full official by-lead values in `residual_official_scores.json`.

## Evidence-backed bottlenecks

### 1. Tail amplitude and a few events dominate the objective (high confidence descriptive evidence)

|Observed speed (km/s)|Scored pairs|Mean MAE|Mean residual bias (pred−truth)|Pooled RMSE|
|---|---:|---:|---:|---:|
|<400|17,316|51.80|+47.19|68.96|
|400–550|26,204|44.79|+16.96|58.79|
|550–700|8,465|62.69|−35.44|80.18|
|≥700|634|164.84|−157.28|193.59|

Residuals show compression toward moderate speeds. In 765 origins with ≥48 valid future observations, observed 72-hour range median is **168.10 km/s**, predicted range medians **89.74/89.37/89.45**, and median per-origin range ratios **0.521/0.519/0.525**. Predictions and observed ranges are compared on the same valid lead support. This is evidence of amplitude compression; it does not establish whether its cause is inadequate predictors, conditional uncertainty, loss/regularization, or faulty extreme observations.

Summing exact linear TIMSE squared-error contributions by unique target timestamp, the top 1/5/10/24 hours account for **9.13%/33.21%/46.77%/67.97%** of mean TIMSE. These are TIMSE fractions, not additive fractions of nonlinear S. The leading target is **2026-01-19 22:00 UTC, 1080.3 km/s**, contributing to 12 forecasts; the next major contributions are January 19–20. `residual_target_contributions.csv` preserves all unique targets and contribution weights. Repeated origin/lead pairs are intentionally retained by the official protocol; they are not independent events. The first decisive follow-up is to audit these high-impact observations against raw quality flags and independently calibrated records, without deleting or replacing official targets based on their errors.

### 2. Forecast horizon is an important information boundary (high confidence descriptive evidence)

|Lead hours|MAE|Pooled RMSE|Bias|
|---|---:|---:|---:|
|1–6|23.19|35.30|+4.60|
|7–24|40.10|55.67|+10.60|
|25–48|57.12|74.35|+17.50|
|49–72|61.28|78.44|+22.52|

Inference: long-horizon evolution is less constrained by the current inputs. This does not measure a Bayes error floor. Origin-speed/rising/falling strata, periods and ENLIL-pending proxy are in `residual_strata.json`; rising origins have MAE56.36 versus falling48.69, but the groups differ in event mix and this is not causal attribution.

### 3. Event timing varies, but a global timing offset is unsupported (high confidence for the tested shifts)

Median absolute forecast-peak timing discrepancy is **12h** for each seed. On common central lead support (13–60), fixed prediction shifts −12/−6/0/+6/+12h yield pooled RMSE **76.36/73.15/71.96/72.58/74.77**. All tested shifts worsen RMSE relative to zero. Local event-dependent timing errors remain possible. These diagnostic shifts are never used to produce calibrated forecasts or claimed as attainable improvements. Peak timing can also be unstable for flat/multiple peaks and incomplete truth.

### 4. Missingness comparisons expose support limits (high confidence)

All evaluated origins have ≥1 missing official observation in their preceding 24-hour window, leaving **no clean-window comparison**. Only 124 scored pairs occur when the latest valid official origin observation is older than 6h. No evaluated input has `img_age_h >24`; therefore image-staleness effect cannot be estimated with that threshold. Forward filling can remove NaNs while obscuring source age; absence of NaNs is not proof of timely observations. Pending ENLIL is an origin-known forecast proxy, not a CME/ICME truth label. Physical event classification remains unverified without an independently timestamped event catalog.

## Fixed controls, uncertainty and interpretation

Protocol was written before control scoring in `residual_strategy.md`. The controls have no fit parameters. `input_rec27` uses the exact parent input recipe, not an adapted recurrence period:

- `data-v2/notebooks/raw_to_input_df.ipynb`, cell14 source line44: `speed_bin.reindex(target_time - Timedelta(days=d))`, d=26,27,28; cell13 documents target-alignment and past availability for h≤72.
- `research/rerun_df_ffill_20261103/prepare.py:49`: forward fill independently along each fixed-h origin timeline. Thus this is the existing **filled input** recurrence; its source timestamps/quality are subject to the separate data audit.
- `leaderboard/scorer/competition.py:85`: official persistence uses the latest valid official observation at or before origin. It is a scoring control and does not prove operational publication-time availability.

2,000 period-stratified circular 168h origin-block bootstrap draws, RNG20261104, recompute empirical quantiles each draw and pair all models on identical indices. No bootstrap draw was rejected.

|Comparison: primary seed42 versus control|Relative S reduction|95% block CI|
|---|---:|---:|
|Official persistence|34.737%|26.687%–40.946%|
|Input persistence|35.408%|27.501%–41.758%|
|Input 27-day recurrence|31.800%|21.705%–46.682%|

Same-configuration seed-mean reductions are 34.522%,35.196%,31.575%, with CIs26.704–40.961%,27.467–41.746%,21.210–46.716%. These comparisons demonstrate skill against simple information controls, not a 50% gain against the already strong primary model.

Matched-seed primary-configuration versus original A+B1 reduction averages **1.620%**, 95% CI **−0.630% to +2.753%**. The interval includes no gain. This comparison is diagnostic because the configurations differ in retained factors; it does not independently validate a single-factor causal mechanism. Seed42's baseline S bootstrap CI is **70.397–112.256**, illustrating substantial event/sample uncertainty, especially in the tail-weighted score. It is not a confidence interval for an unknown model's possible performance.

F1/F2/F3 have been repeatedly exposed and the primary seed42 is selected. These intervals condition on this benchmark and fitted models, omit search/selection uncertainty, and must not be interpreted as independent generalization. A future blind, preregistered protocol is needed before deployment claims. No outcome here establishes either attainability or impossibility of 50% improvement.

## Reproduction and limits

Run from repository root:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python analysis/w50_audit_20261108/residual_diagnostics.py --bootstrap 2000
```

`residual_run.exit=0`; `residual_run.log` includes progress and completion. `residual_arrays.npz` preserves nine actual forecasts, truth, origin grid and labels; `residual_bootstrap.npz` preserves every resampled metric. This was CPU-only postprocessing, with no new training process, external publication or resource allocation for candidate experiments. `residual_provenance.json` records commit, environment and source byte/time identities. Requested cryptographic file identity collection was blocked by an automated older workspace No Hashes hook; the hook was not bypassed, so that reproducibility requirement remains incomplete.

Verification: three existing kernel tests pass (`test_analysis.py`, exact scorer equivalence with missing truth, multiplicity/quantile recomputation, and six-hour circular period-block structure). Real data kernel/scorer agreement assertions also passed for all nine forecast arrays.

Unknowns that change the next decision: whether the high-impact January19–20 tail is independently confirmed and correctly aligned; whether required observations were publicly available at each origin; event-dependent timing versus unpredictable amplitude after those validity checks; optimization/train-loss curves for the already fitted baseline. Current residual evidence cannot distinguish irreducible information limits from training optimization failure. No new model is justified purely by these benchmark residuals.
