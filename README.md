# Solar wind 72h — team leaderboard

Evaluation (**fixed 2026-09-30**): targets **2026-06-01 00:00 – 2026-09-30 00:00 UTC (exclusive)**, hourly origins, 72h windows:
2,833 origins × 72 = 203,976 rows. Origins run from `2026-05-31T23:00:00Z` to `2026-09-26T23:00:00Z`.
Data snapshot: NASA SOHO ACE/SWEPAM archive downloaded 2026-09-30 09:52 UTC (`data/raw/`); 19.8% of the eval targets are forward-filled.
Rank = overall MSE. Regime columns (trend / 27-day recurrence / forecastability × low/high) are diagnostic.

- Private data store: `tlabtlab/sunrun-lb-store` (needs membership in the `tlabtlab` HF org)
- Leaderboard page: https://huggingface.co/spaces/tlabtlab/sunrun-leaderboard (private static Space, org members)

## Submit
```bash
cd sunrun/leaderboard
uv venv .venv && uv pip install -p .venv/bin/python -r requirements.txt   # once
.venv/bin/hf auth login                                                    # once, token with write access to tlabtlab repos
.venv/bin/python submit.py my_predictions.parquet meta.yaml [--dry-run]
```
`--dry-run` validates and scores without uploading. Scoring runs on your machine using the truth/folds
downloaded from the private dataset; the result JSON and your raw file are stored in the dataset and the
leaderboard page is regenerated.

**predictions** (`.parquet` or `.csv`): one row per (origin, horizon), columns
`origin_last_input_utc` (e.g. `2026-06-01T05:00:00Z`, the last timestamp your model may use),
`horizon_hours` (1..72), `pred_kms`. Every origin/horizon in the evaluation set must be present, exactly once.
Origins are all hourly timestamps from `2026-05-31T23:00:00Z` to `2026-09-26T23:00:00Z`
(list them from `folds.parquet` in the dataset: column `origin_last_input_utc`; ignore its regime columns).

**meta.yaml**: see `meta.example.yaml`. `no_future_leakage: true` means only data with timestamp ≤ `origin_last_input_utc`
was used for that origin. The truth is public NASA/ACE data, so this is an honour statement, not a technical guarantee.

## Read the numbers
- `MSE` counts forward-filled targets (same policy as `analysis/naive_baselines`); `MSE obs.` only originally observed targets.
- `beats naive (95% CI)`: paired MSE difference vs Naive, bootstrapped over 72h blocks (folds overlap 71/72, so ~120 effective blocks, not 2,857 folds).
- Regime cells with < 3 non-overlapping 72h blocks are hidden. Do not tune on regime rows.
- The bar is **mean_reversion** (Naive → 648h mean, decay 48h; tuned only on data before June), not Naive: it is ~35% better.

## Tune before June
Evaluation uses every day from June on, so tune on 2025-08 .. 2026-05 only and submit finished models.

## Rebuild (maintainers; the evaluation is fixed, this is for reproducibility)
`python refresh_data.py` → `analysis/naive_baselines/.venv/bin/python extend_baselines.py` → `python build_store.py --out store`
rebuilds series / SES+ARIMA for origins after the original study / folds, truth, naive and baseline files.
Regime thresholds are frozen in `labeling/config.py` (medians over the final 2,833 folds: T 0.762, S 0.174, F 0.472).
Auto ARIMA fits mostly end with an optimizer precision warning (fit code 2; 81% in the study, 96/96 in the extension).
`app.py` is an optional self-hosted Gradio version (needs a paid HF plan or your own server); the default flow is `submit.py`.
