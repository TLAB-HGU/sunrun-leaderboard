# seongeun portable model inference

This package loads frozen fitted models from `<experiment>_seongeun.pkl` and predicts 72 hourly solar-wind speeds per origin. It does not train, download data, or import the original research repository. Python 3.12 is required; install the exact tested dependencies from `requirements.txt`.

All released models run on CPU. For recipes originally fitted on CUDA, the supplied inference helper preserves the original float32 intercept accumulation order without changing the learned trees. Use `load_model(...).predict(...)` so this arithmetic and the ensemble/analog state are applied together.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m model_release --help
```

Unzip the package and run from its directory (or add that directory to `PYTHONPATH`). Download the desired PKL separately from the release Drive folder. Verify the SHA-256 in the release manifest before loading; pickle files execute Python during loading and must be trusted.

```python
from model_release import load_model

inputs = {
    "raw_ace_dir": "data/raw-ace-snapshot",
    "speed_parquet": "data/ace.parquet",
    "ch_parquet": "data/ch-hourly.parquet",
    "suvi_parquet": "data/suvi-hourly-v2.parquet",
    "suvi_manifest": "suvi_availability.parquet",
    # Required for xgb_d76_lead_moe_enlil and xgb_d76_lead_moe_resid_dv_enlil:
    "donki_dir": "data/donki",
}
model = load_model("xgb_d76_lead_moe_seongeun.pkl")
forecast = model.predict(["2026-06-01T00:00:00Z"], inputs)
forecast.to_parquet("predictions.parquet", index=False)
```

Equivalent CLI:

```sh
.venv/bin/python -m model_release \
  --model xgb_d76_lead_moe_seongeun.pkl \
  --origin 2026-06-01T00:00:00Z \
  --raw-ace-dir data/raw-ace-snapshot \
  --speed-parquet data/ace.parquet \
  --ch-parquet data/ch-hourly.parquet \
  --suvi-parquet data/suvi-hourly-v2.parquet \
  --suvi-manifest suvi_availability.parquet \
  --output predictions.parquet
```

Repeat `--origin` for multiple origins. Add `--donki-dir data/donki` for the two Enlil models. Output columns are `origin_last_input_utc` (UTC ISO timestamp), `horizon_hours` (integers 1–72), and `pred_kms` (finite float). Duplicate, empty or non-hourly origins are rejected. Use origins on or after the fitted cutoff, 2026-05-29T00:00:00Z.

## Input contract

The release includes the SUVI availability metadata, not the numeric observation datasets. All raw and numeric input snapshots must be supplied by the user. The packaged availability manifest applies only to its pinned 36,015 SUVI slots; later observations require a matching numeric table and newly verified availability metadata. Supply the same numeric schemas used by the submitted models:

- `raw_ace_dir/year=*/part.parquet`: `timestamp_utc` plus `ace_speed_kms`, `ace_density_cm3`, `ace_temperature_k`, `ace_bt_nt`, `ace_bz_gsm_nt`, `ace_by_gsm_nt`, `ace_epam_e_channel1`, `ace_epam_e_channel2`, `ace_epam_p_channel1`, `ace_epam_p_channel3`, `ace_epam_p_channel5`, `ace_sis_gt10_flux`. Duplicate hours keep the last row in sorted file order. The source log transform, rolling windows and six-hour forward-fill limits are preserved.
- Speed parquet: `timestamp_utc`, `filled_speed_kms`. Provide hourly rows with the original filling policy; inference does not invent replacements. Ordinary models need at least 28 days of speed history (recurrence and 120-hour lags). Residual analog models additionally request features 12 hours before each origin and use 12-hour observed speed histories. Their frozen analog pool is already in the PKL. D46 event-branch analogs need at least 36 days of preceding speed and CH history; provide the complete original history for reproducibility.
- CH parquet: UTC `DatetimeIndex` (or timestamp column for base feature construction), original numeric columns whose names contain `meridian`, `equatorial`, or `_area` but not `lon`. Base features use the most recent row within six hours. For D46 use the original sorted UTC index and the `ch_t035_meridian30_equatorial_area`, `ch_t035_equatorial_area`, `ch_t035_meridian30_area`, `ch_t045_meridian30_equatorial_area`, `ch_t035_area`, `ch_t045_equatorial_area` columns.
- SUVI numeric parquet: UTC index and 12 longitude cells named `lon00_lat1_*` through `lon11_lat1_*`. Required suffixes are `log_median`, `dark_fraction`, `valid_fraction`; preserve the original additional `log_q10` columns, which participate in source availability checks. Numeric rows must exactly align with the sorted manifest slots.
- SUVI manifest: `slot`, `obs_end`, `s3_modified`, `status`, boolean `available`, and optional `verified_local_bytes`. The supplied `available` records the original local nonempty-file check at export time, not a new cryptographic/S3 audit. The original operational proxy is retained: status ok, slot at least one hour before origin, S3 modification no later than origin, finite equatorial numeric cells and nonempty local bytes; otherwise walk back to an earlier slot. Images older than six hours yield missing strips plus a stale flag. Original metadata enforces observation end no later than slot. No raw images are needed at inference.
- DONKI directory for Enlil models: original `WSAEnlilSimulations_*.json` arrays, with `modelCompletionTime`, `estimatedShockArrivalTime`, optional `kp_90`, `kp_135`, `kp_180`, and `isEarthGB`. Only runs completed by each origin enter features. No catalog refresh occurs.

Missing values within valid schemas retain source behavior: many numerical predictors accept NaN, stale images become NaN, and residual analog correction falls back to the base model when unavailable. Missing files, required columns and missing availability flags raise errors. Preserve full historical input files for exact reproduction; shortening history can change feature values or analog availability.

## Reproduction and tests

`identity-audit.json` records source/input identity findings. `suvi_availability.parquet` records source-environment availability. Release manifests and validation receipts alongside the PKLs record fitted-model prediction comparisons and SHA-256 hashes; source metadata is provenance, not a guarantee that arbitrary new inputs match the leaderboard score.

```sh
.venv/bin/python -m unittest model_release.test_release -v
```

Tests cover future-value poisoning with a positive control, completed-at-origin Enlil eligibility, missing inputs, absent-image fallbacks, origin validation, pickle/API dispatch and feature ordering. Release integration also compares numeric features and full official-window predictions with the original source and verifies pickle round trips. Production inference modules contain no training calls.
