# Reproduce five leaderboard entries from the raw ACE tables

Raw ACE parquet → features → training → predictions → official score, for:

| Entry | Method | Reproduced MSE (F1+F2) | vs published `forecast.npz` |
|---|---|---:|---|
| `e2000fs_feat30_pointweight4` | 72 independent LightGBM models (one per horizon) on the frozen Top-30 groups; weight 4 where this target is ≥ 550 km/s | 3732.2689 | max abs diff 0.0 |
| `e2000fs_feat30_lgbm_extreme4` | same models; weight 4 for every sample whose 24 h block has a ≥ 550 km/s value or a ≥ 100 km/s 6 h rise | 3728.7713 | max abs diff 0.0 |
| `e2000fs_dirmo_24_feat30` | **DirMO-24**: three 24-output XGBoost joint-tree blocks on the Top-30 groups (237 columns each) | 3249.8423 | max abs diff 0.0 |
| `e2000fs_dirrecmo_24` | **DirRecMO-24**: three 24-output XGBoost blocks on all 111 features; each block also sees the trajectory of the earlier blocks | 3580.4322 | max abs diff 0.0 |
| `e1604_hsspeak25` | two HGB heads add a 12 h Gaussian bump (0.15 × (peak − path min), from lead 25) to the frozen E0472 forecasts, 3 seeds averaged | 3233.09 (mean seed, local); public 3222.883 | max abs diff 0.0 (vs submitted parquet) |

Each rerun starts from the raw ACE tables and the frozen grids and checks itself against the published result.
Run times on 24 cores: LightGBM arms about 25 min each (2 threads, 216 models), DirMO about 20 min, DirRecMO about 20 min,
e1604 under a minute, feature cache about 40 s.

| | Rebuilt here | Not rebuilt (frozen inputs) |
|---|---|---|
| pointweight4 / extreme4 / DirMO / DirRecMO | lag-0 ACE and speed features, block design, all training stages, official scoring, the Top-30 selection (DirMO) | SUVI / ENLIL / coronal-hole columns of the three 111-feature grids; the Top-30 importance table; HPO for the LightGBM arms |
| e1604 | 29 causal features (compared with the frozen matrices), both heads (full and F1), bump, predictions | E0472 base forecasts (HUXt, ENLIL, SUVI, GONG/PFSS/WSA) |

## Run

```bash
python -m venv .venv && . .venv/bin/activate          # Python 3.12
pip install -r reproduce/requirements.txt
export HF_TOKEN=...                                    # read access to the private datasets below
bash reproduce/run_all.sh            # or: bash reproduce/run_all.sh --smoke  (3 trees, about a minute)
python -m pytest reproduce/tests                       # data-free unit tests
```

`run_all.sh` calls, in order: `fetch_data.py` (downloads and sha256-verifies the inputs), `e1604_hsspeak25/run.py`,
`feat30_pointweight4/build_features.py`, `compare_cache.py` (the rebuilt cache must equal the frozen cache array by
array), `feat30_pointweight4/train_predict.py` (`--arm pointweight4`, `--arm extreme4`), `e2000fs_xgb/dirmo_24_feat30.py`
and `e2000fs_xgb/dirrecmo_24.py`. Every training script takes `--smoke` and `--reference <published forecast.npz>`.
Outputs go to `reproduce/outputs/<entry>/` (`predictions.parquet`, `forecast.npz`, `result.json`).

Set `SUNRUN_REPRO_DATA` / `SUNRUN_REPRO_OUT` to move the input and output directories. Nothing else depends on the
location of the checkout.

## Data

Inputs are not stored in git. `data_manifest.json` lists every file with its sha256; `fetch_data.py` fetches them:

| Source | Content |
|---|---|
| this repository | `releases/e1604/prepared_inputs.npz`, `releases/f1-20261026/hsspeak_f1_inputs.npz`, and the published `forecast.npz` of the four trained entries (used as references) |
| HF dataset `tlabtlab/sunrun-repro-data` (private) | `store/ch-v1/ace.parquet`, `store/ch-breakthrough-v2/upload/raw-ace-snapshot/year=*/part.parquet` (raw ACE); `grids/*` (train / F1 / official 111-feature grids, `features_v3.json`); `e1604/huxt_traj_slim.npz`; `inputs/feature_importance.csv` (Top-30 selection); `reference/*` (frozen cache and E0472-based predictions for verification) |
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

- The Top-30 feature selection used previously scored official periods, so the feat30 family (pointweight4, extreme4,
  DirMO-24 feat30) is official-exposed post-hoc research, not a fresh holdout. pointweight4 failed the both-region
  acceptance gate; it is published as a record.
- `e2000fs_dirrecmo_24` and the XGBoost recipe (900 rounds, depth 3, lr 0.03, seed 42, `multi_output_tree`) live in
  `e2000fs_xgb/config/xgb.json`; the original `experiments/e2000/model.py` that defined it is no longer available.
- HPO parameters, selected feature groups and the scorer (sha256 `9b5f36cc…`) are frozen in
  `feat30_pointweight4/config/frozen.json` and `scorer/`; the training code does not tune anything.
- `e1604_hsspeak25/provenance/` keeps the original `run.py`, `official.py`, `score.py` (leaderboard git `d55d9f6d6`),
  the F1 head script, the pre-registration and the registration-defect note. They are history only; `run.py` one level
  up is the self-contained reproduction.
- Results are deterministic here (fixed seeds, 2 threads). A different LightGBM version or thread count can change
  predictions in the last digits; `train_predict.py --reference` prints the maximum difference.
