# Solar wind 72h — team leaderboard (submit client)

Evaluation (**two periods since 2026-10-26**), hourly origins, 72h windows, all target times UTC, end exclusive:

| period | targets | origins | folds |
|---|---|---|---|
| **F1** | 2026-03-01 00:00 – 2026-06-01 00:00 | `2026-02-28T23:00:00Z` .. `2026-05-28T23:00:00Z` | 2,137 |
| **F2** | 2026-06-01 00:00 – 2026-09-30 00:00 | `2026-05-31T23:00:00Z` .. `2026-09-26T23:00:00Z` | 2,833 |

4,970 origins × 72 = 357,840 rows. Origins whose 72h window straddles the boundary (2026-05-29..05-31) are in neither period.
Rank = overall MSE over F1 + F2. The board also shows F1 MSE, F2 MSE, and the same three MSEs over targets whose true speed
is ≥ 550 km/s (`MSE ≥550`, `F1 ≥550`, `F2 ≥550`; forward-filled targets count, as in `MSE`). Regime columns are diagnostic.

**F1 training cutoff:** F1 predictions must come from a model trained (and tuned) only on data before
**2026-02-26 00:00 UTC** (F1 start − 72h). Declare it in `meta.yaml` as `f1_training_data_end_utc` (≤ that time);
`submit.py` rejects later values. Like `no_future_leakage`, this is an honour statement.

Results submitted before 2026-10-26 hold only F2 (their MSE is the F2 MSE). They are listed below the ranking as
`F2 only`, sorted by F2 MSE, until they are resubmitted with F1 + F2 predictions (`--replace` keeps the same key).

- [E2000FS DirMO-24 loss · F1+F2 월별 GIF](docs/visualizations/e2000fs_dirmo_24/README.md) — 제출 F1·F2 모델의 round별 train·validation MSE, 2026-03~09 전체 4,970개 공식 폴드.
- [월별 검증 폴드 GIF: E0472 / xgb_d76_lead_moe_resid_dv_enlil](docs/visualizations/xgb_d76_lead_moe_resid_dv_enlil/README.md) — 최근 7일 관측과 이후 72시간 실제값·예측, 전체 2,833개 공식 폴드 및 별도 10월 참고 추론 264개, 2배속.

- [Reproduce `e2000fs_feat30_pointweight4` and `e1604_hsspeak25` end to end](reproduce/README.md) — raw ACE parquet → features → training → predictions, verified against the published results.

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
`horizon_hours` (1..72), `pred_kms`. Every origin/horizon of the evaluation set (F1 and F2) must be present exactly once.
List the origins from the dataset:
```python
import pandas as pd
from huggingface_hub import hf_hub_download
folds = pd.read_parquet(hf_hub_download("tlabtlab/sunrun-lb-store", "folds.parquet", repo_type="dataset"))
origins = folds["origin_last_input_utc"]   # `period` is F1 or F2; ignore the regime columns
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
- [`baseline / naive`](scripts/baseline/naive/a73b16dda3144747a5589cd0a34ba56512df8b0e626125c65bdb1e940bc5cb02.py):
  repeats the last available speed for all 72 horizons.
- [`baseline / seasonal_naive_648`](scripts/baseline/seasonal_naive_648/e82281cb844fb18079acceb7579f554d639a65f8e9bba5fd1016b32713f72f77.py):
  uses the value from the same horizon in the previous 648-hour solar-rotation window.
- [`baseline / ses`](scripts/baseline/ses/1794ffdffc953cf81eaf273a8f84d0df154e58d2b9b2cb6a9fe6ecf69fa4778f.py):
  refits optimized simple exponential smoothing on the latest 648 hours at each fold.
- [`baseline / auto_arima`](scripts/baseline/auto_arima/e176bda8304400115ef8bd0f433c8a4794a646606feb497e075d84877f0fe60f.py):
  refits the published non-seasonal AutoARIMA configuration on the latest 648 hours at each fold.

SES and AutoARIMA were reproduced with the pinned environment in `requirements-baselines.txt`. Their scripts report
fit time separately and use only the post-fit 72-hour `predict()` call for `inference_seconds_per_fold`.
The locked baseline environment uses Python 3.12.3 (also recorded in `scripts/baseline/.python-version`):
```bash
uv venv --python 3.12.3 .venv-baselines
uv pip install -p .venv-baselines/bin/python -r requirements-baselines.txt
```

## Input data
Model inputs come from the shared Drive folder **sun-db**
(https://drive.google.com/drive/folders/1fOOjTHYUiwIGSLaWXeDr5uIUxttFJWw7): `timeseries/ace_solar/year=YYYY/part.parquet`
(ACE solar wind, hourly, one file per year) and `timeseries/suvi_hourly/year=YYYY/`.
- The scoring truth is the ACE SWEPAM bulk speed. It was checked against `ace_speed_kms` of the collector's `spaceweather/solar.csv`:
  identical on all 2,328 originally observed evaluation hours (max abs diff 0.0). The `sun-db` parquet files themselves were not compared.
- The files keep updating and contain hours **after** your origin. For origin `t` use only rows with `timestamp_utc <= t`
  (`point_in_time_vintage_verified` is 0 for every row: values are latest-vintage, not as-of).
- 19.8% of the F2 targets (21.3% of F1) are missing hours; the scoring truth forward-fills them with the last valid speed (`MSE obs.` ignores them).
  Predict as if every hour had a value; do not drop missing hours.

## Read the numbers
- `MSE` counts forward-filled targets; `MSE obs.` only originally observed targets.
- `≥550` columns use only targets with true speed ≥ 550 km/s (F1 24,480 pairs, F2 23,976): high-speed streams, where most of the squared error is.
- `CSI` and `timing MAE (h)` score HSS **events** in each 72h window, for truth and forecast alike: an event is a run of hours
  ≥ 550 km/s (quiet gaps ≤ 6h merged; runs touching hour 1 or 72 are cut by the window and not counted), timed at its peak.
  Forecast and true peaks are matched one-to-one within 12h. Summed over all windows, `CSI` (threat score) = TP / (TP + FP + FN):
  did the forecast catch the HSS at all. `timing MAE` = mean |forecast − true peak hour| of matched events: how many hours early or
  late a caught HSS was (the result JSON also holds the signed `timing_bias_h`, + = late, and POD/FAR/TP/FP/FN per period).
  Windows overlap, so one HSS counts once per window that contains it. Both are diagnostic; rank stays MSE.
- `beats naive (95% CI)`: paired MSE difference vs Naive, bootstrapped over 72h blocks (folds overlap 71/72, so ~120 effective blocks, not 2,833 folds).
- Regime cells are computed on F2 folds only (thresholds were frozen on them); cells with < 3 non-overlapping 72h blocks are hidden. Do not tune on regime rows.
- The bar is **mean_reversion** (Naive → 648h mean, decay 48h, tuned only on data before June), not Naive: it is ~36% better.
- Evaluation covers every day from March on. F1 models: train/tune before 2026-02-26 00:00 UTC. F2 models: tune on 2025-08 ..
  2026-05 only, as before. A model trained once before 2026-02-26 can produce both periods.
- The `mean_reversion` decay (48h) was tuned on 2025-09 .. 2026-05; retuned before 2026-02-26 it would be 24h, with dev MSE
  within 0.13% of 48h, so the baseline was kept unchanged for F1 (team decision 2026-10-26).

Maintenance code (data refresh, fold/label building, baseline generation, Gradio app) is kept at git tag `full-2026-09-30`;
the F1 + F2 store build is on branch `maint-f1f2` (its F2 folds/truth/naive are bit-identical to the 2026-09-30 store).
