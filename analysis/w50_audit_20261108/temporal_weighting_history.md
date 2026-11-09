# Temporal weighting: bounded failure-history audit

Native Explorer source lookup, followed by leader receipt checks. No model was
fitted and no ongoing experiment score was read. No exact completed or queued
condition retaining all current28features, all rows, tail4 and spectral was
established by this bounded search. This is not proof that every semantic
variant is absent, or permission to blindly repeat the old recipes.

The following completed seed42 reruns have train.exit0 and saved official
`lead-timse-6h-v1` receipts under research/rerun_df_ffill_20261103/runs:

|Stored recipe|Stored Total S|Intervention boundary|
|---|---:|---|
|e2000fs_feat30_transfer_recency|102.436297|LightGBM scalar heads; temporal weights replace unweighted L2, not today's retained loss|
|e2000fs_feat30_transfer_recent_window|105.453446|Keep only last365days; different support and loss/architecture|
|e2000fs_budget_covshift|103.939059|XGBoost vector heads, point4 × density weights; no retained spectral/current offset|
|Historical point4 reference|100.616914|Different parent; not the current frozen97.492442 single model|

These are historical stored scores, not fresh like-for-like comparisons or
current-SOTA claims. Metadata points to df_ffill_trainy.parquet and144h
maturity, but source/version/observation identity needs its own replay before
stronger numerical comparison. Generic target-policy text in some metadata
mentions spectral masking even where the executed LightGBM recipe has no
spectral objective; prefer the actual fit path over generic wording.

## Smooth recency versus removing rows

`research/rerun_df_ffill_20261103/advanced.py:27,110` defines
r=.1+.9*exp(−ln(2)*age_days/180), normalized to mean1. It retains eligible
rows and passes sample_weight to400tree depth6 per-horizon LightGBM fits on30
features. It does not multiply the current tail4 objective or retain spectral
coupling/frozen-offset pipeline. Its worse score does not isolate temporal
weighting on today's parent.

The same file's recent-window branch at112 keeps365days and fits unweighted
LightGBM. It tests removal of support and a different estimator, not smooth
forgetting with all rows/components retained.

## Covariate redistribution is a different intervention

`research/rerun_df_ffill_20261103/structured.py:30,231` uses900round XGBoost,
three24-lead heads and30features. A past-only classifier distinguishes recent
81days from pre-cutoff origins, with cross-fitting on27day blocks/72h embargo.
Ratios clipped[.25,4] multiply point4, then normalize within calendar-year ×
horizon × target-speed-band to preserve original mass. This is restricted
within-stratum covariate redistribution, not unrestricted calendar recency.
The fit lacks today's spectral and frozen-offset components.

The secondary summary at offset_roll_ablation3_20261105/literature/
06_nonstationarity_training_window/evidence.md:32 groups this with per-horizon
LightGBM families; the executed covshift code/metadata are XGBoost. The stored
negative result does not establish that forgetting old samples generally
fails, or that its failure was only a bias shift.

## Near-parent offset weighting is not gradient weighting

offset_roll_ablation2_20261105/strategy.md:47 and experiment.py:198 register
o2_offset_ewma: replace the flat final336-row calibration-residual mean with an
exponentially weighted estimate over three months, half-life336rows. This
weights offset residuals rather than training gradients. This bounded pass did
not establish that arm's completion/outcome receipts, and does not count it
as a failed or unexecuted full-training recency experiment.

## Decision boundary

The current saved-model training-fit probe precedes any new recency training
decision. A future justified test must retain original targets/rows/features,
threshold weights and spectral term, make any row multiplication explicit for
both gradients and Hessians, freeze age reference/normalization beforehand,
and distinguish distribution drift from missing physical information. The
legacy180day half-life is an independently existing recipe, not a physical
constant or an optimum inferred from current F1/F2/F3. No decay strength is
selected or tuned by this history audit.
