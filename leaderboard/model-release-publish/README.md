# Solar wind 72h — team leaderboard (submit client)

Evaluation (**three quarterly periods since 2026-10-31**), hourly origins, 72h windows, all target times UTC, end exclusive:

| period | targets | origins | folds | training cutoff (`meta.yaml`) |
|---|---|---|---|---|
| **F1** | 2026-01-01 00:00 – 2026-04-01 00:00 | `2025-12-31T23:00:00Z` .. `2026-03-28T23:00:00Z` | 2,089 | `f1_training_data_end_utc` ≤ 2025-12-29 |
| **F2** | 2026-04-01 00:00 – 2026-07-01 00:00 | `2026-03-31T23:00:00Z` .. `2026-06-27T23:00:00Z` | 2,113 | `f2_training_data_end_utc` ≤ 2026-03-29 |
| **F3** | 2026-07-01 00:00 – 2026-09-30 00:00 | `2026-06-30T23:00:00Z` .. `2026-09-26T23:00:00Z` | 2,113 | `f3_training_data_end_utc` ≤ 2026-06-29 |

6,315 origins × 72 = 454,680 rows. Origins whose 72h window straddles a boundary (the 71 origins before each period start) are in no period.
Rank = overall MSE over F1 + F2 + F3. The board also shows F1, F2 and F3 MSE, and the same four MSEs over targets whose true speed
is ≥ 550 km/s (`MSE ≥550`, `F1 ≥550`, `F2 ≥550`, `F3 ≥550`; forward-filled targets count, as in `MSE`). Regime columns are diagnostic
and now cover F2 + F3 (the thresholds were frozen on the 2026-06-01..09-29 folds).

**Training cutoffs:** predictions for a period must come from a model trained (and tuned) only on data before
**period start − 72h**: 2025-12-29, 2026-03-29 and 2026-06-29 00:00 UTC for F1, F2 and F3. Declare each in `meta.yaml`
(`f1_/f2_/f3_training_data_end_utc`, ≤ that time); `submit.py` rejects later values. Like `no_future_leakage`, this is an honour statement.

**Earlier results (rescored 2026-10-31):** results submitted under the old windows (F1 = 03..05, F2 = 06..09) were rescored from their
stored predictions on the new folds; the old scores are kept in each record as `legacy_f1f2`. Those predictions start in March and skip the
origins of 2026-05-29..05-31, so their F1 is only about a third covered and F2 lacks those 3 days (`coverage_f1/f2/f3` in the record).
They are listed below the complete rows as `no F1`, sorted by pooled F2 + F3 MSE, with F1 and the overall MSE hidden, until they are
resubmitted with predictions for all 6,315 origins (`--replace` keeps the same key). A result that could not be rescored shows as `legacy`.

- [월별 검증 폴드 GIF: E0472 / xgb_d76_lead_moe_resid_dv_enlil](docs/visualizations/xgb_d76_lead_moe_resid_dv_enlil/README.md) — 최근 7일 관측과 이후 72시간 실제값·예측, 전체 2,833개 공식 폴드 및 별도 10월 참고 추론 264개, 2배속.
- [E1149 L72-COND / ACE+2시간 GIF](docs/visualizations/e1149_l72cond_ace2h/README.md) — 원 실험의 6~9월 저장 예측과 10월 참고 추론.
- [E1149 L72-COND / ACE 지연 없는 재학습 GIF](docs/visualizations/e1149_l72cond_ace0h/README.md) — 학습부터 +2시간을 제거한 미제출 비교, 6~10월.

- [월별 검증 폴드 GIF: xgb_d76_lead_moe_enlil](docs/visualizations/xgb_d76_lead_moe_enlil/README.md) — 최근 7일 관측과 이후 72시간 실제값·예측, 전체 2,833개 폴드.

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

Every new result is uniquely identified by `(team_member, experiment, submission key)`. `submit.py` derives the key
internally from the canonical JSON form of `config.data` and `config.hyperparameters`; YAML key order does not affect it,
and you never compute or check it yourself.
The same key cannot be submitted twice unless `--replace` is explicitly supplied.

`inference_seconds_per_fold` is the mean wall-clock time needed to produce one 72-hour forecast after data loading,
training/model initialization, feature preparation, and output writing have been excluded. Measure the inference section
with `time.perf_counter()` over multiple folds.

Each config must have one reproducible script at:
```text
scripts/<team_member>/<experiment>/<submission-key>.py
```
Get that path from `submit.replay_script_path(member, experiment, config)`.
`--dry-run` checks the local path. A real submission checks that the script is already available on this repository's
`main` branch, so push the script before uploading the result. Existing entries without config, timing, or code metadata
remain visible as `legacy` records.

**The board's `code` column links the architecture, not that replay script.** `meta.yaml` must also carry
`architecture_url`: the study source that defines the model, pinned to a commit, e.g.
`https://github.com/TLAB-HGU/sunrun-leaderboard/blob/6b85acc0b00d731bf31b6885e0b847c18ee395d0/research/e2000fs_f123-20261031/qxgb.py`
(optionally `#Lx-Ly`). Publish the study directory first, then submit: `submit.py` rejects a URL on a branch instead of a
commit, a directory, and the `scripts/<member>/<experiment>/<submission-key>.py` replay script itself. It records the link
in the result (`architecture_url`) and writes the matching entry to the Space's `architecture-code.json` in the same
commit as the page, so a new row is published as `architecture` and never needs hand curation:

```bash
python architecture_code.py check                  # ranking rows whose code cell is not an architecture link (exit 1)
python architecture_code.py add <member> <exp> <submission-key> <url>   # repair an older row
python architecture_code.py publish                # upload the mapping and rewrite the code cells
python architecture_code.py verify                 # every mapped URL answers 200
```

`lbstatus.py sync` reports any row that is still on a replay script, and `lbstatus.py` re-applies the mapping whenever it
rewrites the page.

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
- 15.9% of the F1 targets (23.1% of F2, 19.3% of F3) are missing hours; the scoring truth forward-fills them with the last valid speed (`MSE obs.` ignores them).
  Predict as if every hour had a value; do not drop missing hours.

## Read the numbers
- `MSE` counts forward-filled targets; `MSE obs.` only originally observed targets.
- `≥550` columns use only targets with true speed ≥ 550 km/s (F1 42,129 pairs, F2 23,156, F3 13,094): high-speed streams, where most of the squared error is.
- `CSI` and `timing MAE (h)` score HSS **events** in each 72h window, for truth and forecast alike: an event is a run of hours
  ≥ 550 km/s (quiet gaps ≤ 6h merged; runs touching hour 1 or 72 are cut by the window and not counted), timed at its peak.
  Forecast and true peaks are matched one-to-one within 12h. Summed over all windows, `CSI` (threat score) = TP / (TP + FP + FN):
  did the forecast catch the HSS at all. `timing MAE` = mean |forecast − true peak hour| of matched events: how many hours early or
  late a caught HSS was (the result JSON also holds the signed `timing_bias_h`, + = late, and POD/FAR/TP/FP/FN per period).
  Windows overlap, so one HSS counts once per window that contains it. Both are diagnostic; rank stays MSE.
- `beats naive (95% CI)`: paired MSE difference vs Naive, bootstrapped over 72h blocks (folds overlap 71/72, so ~88 effective blocks, not 6,315 folds).
- Regime cells with < 3 non-overlapping 72h blocks are hidden. Do not tune on regime rows.
- The bar is **mean_reversion** (Naive → 648h mean, decay 48h, tuned only on data before June), not Naive: it is ~36% better.
- Evaluation covers every day from June on: tune on 2025-08 .. 2026-05 only and submit finished models.

Maintenance code (data refresh, fold/label building, baseline generation, Gradio app) is kept at git tag `full-2026-09-30`.
