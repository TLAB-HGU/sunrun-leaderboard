# Solar wind 72h — team leaderboard (submit client)

Evaluation (**fixed 2026-09-30**): targets **2026-06-01 00:00 – 2026-09-30 00:00 UTC (exclusive)**, hourly origins, 72h windows:
2,833 origins × 72 = 203,976 rows. Origins run from `2026-05-31T23:00:00Z` to `2026-09-26T23:00:00Z`.
Rank = overall MSE. Regime columns are diagnostic.

- Private data store (truth, folds, results): `tlabtlab/sunrun-lb-store`
- Leaderboard page: https://huggingface.co/spaces/tlabtlab/sunrun-leaderboard (private; needs an HF login of an org member).
  Without an account, `python submit.py --show` prints the same ranking in the terminal.

## Setup (once)
```bash
uv venv .venv && uv pip install -p .venv/bin/python -r requirements.txt
export HF_TOKEN=<token from the maintainer>   # no HF account / org membership needed; never commit or paste it
```

## Submit
```bash
.venv/bin/python submit.py my_predictions.parquet meta.yaml --dry-run   # validate + score, upload nothing
.venv/bin/python submit.py my_predictions.parquet meta.yaml             # real submission
.venv/bin/python submit.py --show                                       # current ranking
```
Scoring runs on your machine with truth/folds downloaded from the private dataset; the result JSON and your raw file
are stored in the dataset and the leaderboard page is regenerated.

**predictions** (`.parquet` or `.csv`): one row per (origin, horizon), columns
`origin_last_input_utc` (e.g. `2026-06-01T05:00:00Z`, the last timestamp your model may use),
`horizon_hours` (1..72), `pred_kms`. Every origin/horizon of the evaluation set must be present exactly once.
List the origins from the dataset:
```python
import pandas as pd
from huggingface_hub import hf_hub_download
folds = pd.read_parquet(hf_hub_download("tlabtlab/sunrun-lb-store", "folds.parquet", repo_type="dataset"))
origins = folds["origin_last_input_utc"]   # ignore the regime columns
```

**meta.yaml**: see `meta.example.yaml`. `no_future_leakage: true` means only data with timestamp ≤ `origin_last_input_utc`
was used for that origin. The truth is public NASA/ACE data, so this is an honour statement, not a technical guarantee.

Every new result is uniquely identified by `(team_member, experiment, config_sha256)`. `config_sha256` is computed by
`submit.py` from the canonical JSON form of `config.data` and `config.hyperparameters`; YAML key order does not affect it.
The same key cannot be submitted twice unless `--replace` is explicitly supplied.

`inference_seconds_per_fold` is the mean wall-clock time needed to produce one 72-hour forecast after data loading,
training/model initialization, feature preparation, and output writing have been excluded. Measure the inference section
with `time.perf_counter()` over multiple folds.

Each config must have one reproducible script at:
```text
scripts/<team_member>/<experiment>/<config_sha256>.py
```
`--dry-run` checks the local path. A real submission checks that the script is already available on this repository's
`main` branch, so push the script before uploading the result. Existing entries without config, timing, or code metadata
remain visible as `legacy` records.

To intentionally replace an existing result with the same three-part key:
```bash
.venv/bin/python submit.py my_predictions.parquet meta.yaml --replace
```

## Experiment scripts
- [`baseline / mean_reversion`](scripts/baseline/mean_reversion/095a1fce921eaa40ef3ab29a582af4617eb34dab8327cc7fdd8acdf32560ee66.py):
  648-hour rolling mean with 48-hour exponential decay. The script generates all 72-hour forecasts and reports its
  measured inference seconds per fold.

## Input data
Model inputs come from the shared Drive folder **sun-db**
(https://drive.google.com/drive/folders/1fOOjTHYUiwIGSLaWXeDr5uIUxttFJWw7): `timeseries/ace_solar/year=YYYY/part.parquet`
(ACE solar wind, hourly, one file per year) and `timeseries/suvi_hourly/year=YYYY/`.
- The scoring truth is the ACE SWEPAM bulk speed. It was checked against `ace_speed_kms` of the collector's `spaceweather/solar.csv`:
  identical on all 2,328 originally observed evaluation hours (max abs diff 0.0). The `sun-db` parquet files themselves were not compared.
- The files keep updating and contain hours **after** your origin. For origin `t` use only rows with `timestamp_utc <= t`
  (`point_in_time_vintage_verified` is 0 for every row: values are latest-vintage, not as-of).
- 19.8% of the evaluation targets are missing hours; the scoring truth forward-fills them with the last valid speed (`MSE obs.` ignores them).
  Predict as if every hour had a value; do not drop missing hours.

## Read the numbers
- `MSE` counts forward-filled targets (19.8% of the evaluation targets); `MSE obs.` only originally observed targets.
- `beats naive (95% CI)`: paired MSE difference vs Naive, bootstrapped over 72h blocks (folds overlap 71/72, so ~120 effective blocks, not 2,833 folds).
- Regime cells with < 3 non-overlapping 72h blocks are hidden. Do not tune on regime rows.
- The bar is **mean_reversion** (Naive → 648h mean, decay 48h, tuned only on data before June), not Naive: it is ~36% better.
- Evaluation covers every day from June on: tune on 2025-08 .. 2026-05 only and submit finished models.

Maintenance code (data refresh, fold/label building, baseline generation, Gradio app) is kept at git tag `full-2026-09-30`.
