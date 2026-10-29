# Reproduce `e2000fs_feat30_pointweight4` and `e1604_hsspeak25`

Raw ACE parquet → features → training → predictions → score, for the two leaderboard entries. Both reruns were
checked against the published results: pointweight4 matches `releases/e2000fs_feat30_pointweight4/forecast.npz`
exactly (max abs diff 0.0) and e1604 matches the submitted predictions parquet exactly (max abs diff 0.0).

| | `e2000fs_feat30_pointweight4` | `e1604_hsspeak25` |
|---|---|---|
| Method | 72 independent LightGBM models (one per horizon) on the frozen Top-30 feature groups; training weight 4 where the target is ≥ 550 km/s, else 1 (mean-normalized) | Two `HistGradientBoostingRegressor` heads (peak speed, peak hour) add a 12 h Gaussian bump (0.15 × (peak − path min), from lead 25) to the frozen E0472 forecasts, 3 seeds averaged |
| Starts from | raw ACE tables + the three frozen 111-feature grids | raw ACE table + frozen E0472 base forecasts + HUXt Earth trajectory |
| Rebuilt here | lag-0 ACE/speed features, block design, 3 training stages, official scoring | 29 causal features (compared with the frozen matrices), both heads (full and F1), bump, predictions |
| Not rebuilt | SUVI / ENLIL / coronal-hole columns of the grids; the Top-30 selection and HPO | the E0472 base forecasts (HUXt, ENLIL, SUVI, GONG/PFSS/WSA) |
| Expected | combined MSE **3732.27** (< 550: 3365.97, ≥ 550: 6071.06), 357,840 rows | mean-seed MSE ≈ 3233.09 locally; public MSE 3222.883 |
| Run time | about 25 min on 2 threads (216 models) | under 1 min |

## Run

```bash
python -m venv .venv && . .venv/bin/activate          # Python 3.12
pip install -r reproduce/requirements.txt
export HF_TOKEN=...                                    # read access to the private datasets below
bash reproduce/run_all.sh            # or: bash reproduce/run_all.sh --smoke  (3 trees, about a minute)
python -m pytest reproduce/tests                       # data-free unit tests
```

`run_all.sh` calls, in order: `fetch_data.py` (downloads and sha256-verifies the inputs),
`e1604_hsspeak25/run.py`, `feat30_pointweight4/build_features.py`, `compare_cache.py` (the rebuilt cache must equal the
frozen cache array by array) and `feat30_pointweight4/train_predict.py`. Outputs go to `reproduce/outputs/`
(`pointweight4/predictions.parquet`, `e1604/predictions.parquet`, `results.json`, `receipt.json`).

Set `SUNRUN_REPRO_DATA` / `SUNRUN_REPRO_OUT` to move the input and output directories. Nothing else depends on the
location of the checkout.

## Data

Inputs are not stored in git. `data_manifest.json` lists every file with its sha256; `fetch_data.py` fetches them:

| Source | Content |
|---|---|
| this repository | `releases/e1604/prepared_inputs.npz`, `releases/f1-20261026/hsspeak_f1_inputs.npz`, `releases/e2000fs_feat30_pointweight4/forecast.npz` |
| HF dataset `tlabtlab/sunrun-repro-data` (private) | `store/ch-v1/ace.parquet`, `store/ch-breakthrough-v2/upload/raw-ace-snapshot/year=*/part.parquet` (raw ACE); `grids/*` (train / F1 / official 111-feature grids, `features_v3.json`); `e1604/huxt_traj_slim.npz`; `reference/*` (frozen cache and E0472-based predictions for verification) |
| HF dataset `tlabtlab/sunrun-lb-store` (private), revision `5c274be8…` | `truth.parquet`, `folds.parquet`, `naive.parquet` for the official score |

`e1604/huxt_traj_slim.npz` is the `traj` array of the 654 MB HUXt cache restricted to the 4,970 evaluation origins
(`e1604_hsspeak25/make_huxt_slim.py`).

## Input timing

E0472 (the e1604 base) and the frozen grids were produced under the old ACE +2h availability delay. Per the 2026-10-29
rule the pointweight4 cache is re-timed to the origin hour (`timestamp_utc <= t0`, no 2h shift); `build_features.py`
first proves that the same functions at lag 2h reproduce the frozen parquet columns (0 mismatches), then writes the lag-0
cache. e1604 keeps the original contract (ACE observation through the origin on its frozen E0472 base) and is labelled
ACE+2h when compared with new runs.

## Caveats

- The Top-30 feature selection used previously scored official periods, so this family is official-exposed post-hoc
  research, not a fresh holdout. pointweight4 failed the both-region acceptance gate; it is published as a record.
- HPO parameters, selected feature groups and the scorer (sha256 `9b5f36cc…`) are frozen in
  `feat30_pointweight4/config/frozen.json` and `scorer/`; the training code does not tune anything.
- `e1604_hsspeak25/provenance/` keeps the original `run.py`, `official.py`, `score.py` (leaderboard git `d55d9f6d6`),
  the F1 head script, the pre-registration and the registration-defect note. They are history only; `run.py` one level
  up is the self-contained reproduction.
- Results are deterministic here (fixed seeds, 2 threads). A different LightGBM version or thread count can change
  predictions in the last digits; `train_predict.py --reference` prints the maximum difference.
