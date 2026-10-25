# E1908 — frozen TSAA direct-forecast submission

Source: Nochumsohn & Azencot, Data Augmentation Policy Search for Long-Term Forecasting,
arXiv:2405.00319. CPU tabular adaptation: 111 existing features, three direct horizon
heads, 900 XGBoost rounds, seeds 0/1/2, training targets strictly before 2026-05-29 UTC.
The augmentation policy was fixed using chronological internal validation before
external ROLL3. No official scores or labels were used to select or modify it.

The user explicitly requested this leaderboard entry after the preofficial gate
failure was disclosed. ROLL3 MSE improved 1.68%, all three folds improved and F1
rose from 0.375 to 0.409, but the 72h-block CI [-4.13%,+0.98%] crossed zero.
This is a submitted candidate, not an adopted champion. The accepted parent and
the failed gate are unchanged. Existing source availability receipts are inherited;
publication does not independently certify every historical source issue timestamp.

## Reproduce the submitted forecast

From the repository root:

    python scripts/seongeun/e1908_tsaa_policy_prediction_mean/4e583eee5ebd4be9efc5519464cbf48fbdd21cab62d3585768f5e7279d44e3c7.py --output predictions.parquet

Requires NumPy, pandas and pyarrow. The script verifies the configuration and the
three seed archives, then reproduces their predeclared equal prediction mean.
It contains no truth or observation data. This is fixed-grid forecast replay,
not end-to-end retraining or inference for new origins. The archived source
modules preserve the actual registered training and prediction implementation.

## Timing

The leaderboard timing is measured from actual predictions of nine loaded CPU
XGBoost heads, plus the equal mean of three seeds, over all 2,833 origins.
Data loading, model loading/initialization, training, feature preparation and
output writing are excluded. See inference-timing.json and benchmark_models.py.
The replay script's cached averaging time is recorded separately and is not
presented as model inference time. All nine official models used 900 rounds;
their repeated prediction exactly reproduced the official seed files.

Administrative execution: E1910; model identity remains E1908.
