# Solar wind 72h — team leaderboard (submit client)

Evaluation (**fixed 2026-09-30**): targets **2026-06-01 00:00 – 2026-09-30 00:00 UTC (exclusive)**, hourly origins, 72h windows:
2,833 origins × 72 = 203,976 rows. Origins run from `2026-05-31T23:00:00Z` to `2026-09-26T23:00:00Z`.
Rank = overall MSE. Regime columns are diagnostic.

- Private data store (truth, folds, results): `tlabtlab/sunrun-lb-store` — needs membership in the `tlabtlab` HF org
- Leaderboard page: https://huggingface.co/spaces/tlabtlab/sunrun-leaderboard (private static Space, org members)

## Setup (once)
```bash
uv venv .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/hf auth login        # token with write access to tlabtlab repos
```

## Submit
```bash
.venv/bin/python submit.py my_predictions.parquet meta.yaml --dry-run   # validate + score, upload nothing
.venv/bin/python submit.py my_predictions.parquet meta.yaml             # real submission
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

## Read the numbers
- `MSE` counts forward-filled targets (19.8% of the evaluation targets); `MSE obs.` only originally observed targets.
- `beats naive (95% CI)`: paired MSE difference vs Naive, bootstrapped over 72h blocks (folds overlap 71/72, so ~120 effective blocks, not 2,833 folds).
- Regime cells with < 3 non-overlapping 72h blocks are hidden. Do not tune on regime rows.
- The bar is **mean_reversion** (Naive → 648h mean, decay 48h, tuned only on data before June), not Naive: it is ~36% better.
- Evaluation covers every day from June on: tune on 2025-08 .. 2026-05 only and submit finished models.

Maintenance code (data refresh, fold/label building, baseline generation, Gradio app) is kept at git tag `full-2026-09-30`.
