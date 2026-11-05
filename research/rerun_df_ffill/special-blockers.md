# E0472-derived experiments and block0 snapshot recovery

## Implemented: xgb_d76_lead_moe_resid_dv_enlil

`special.py` is a standalone CPU runner. The source dependencies were recovered locally at:

- `leaderboard/lb-f1f2/experiments/d148_enlil_live/lean_enlil.py`
- `leaderboard/lb-f1f2/experiments/d147_e0421_lean/lean.py`
- `leaderboard/lb-f1f2/experiments/d143_axisA_analog/a2b/{resid_lib_a2b,run_official_a2b,spec_a2b}.py`
- `leaderboard/experiments/f1_resubmit_20261026/e0472_f1.py`

The runner retrains seven scalar models for each of W0, W1, W2 and the full fit. The seed-cell-0 short expert seeds are 0/1/2, long expert seed 1. The registered per-lead mixture weights are unchanged. The 111 rowwise inputs retain original order: horizon, recurrence, speed, ACE, CH, strips, image metadata, ENLIL. CH names map to new `suvi_*` columns; image staleness is reconstructed from the new raw table.

All query paths, 12-hour hindcasts, and hourly candidate paths are computed by these fresh models. The K=15 analog correction retains the 96-dimensional scenario, variance scaling, 654.48-hour exclusion, one-year lookback and 0..1 step-.05 OOF blend search. Residual target outcomes are explicitly separated from the causal input-speed timeline. They come only from the prepared training rows (including the user's requested causal target ffill); official query `y` never enters fitting. Remaining target NaNs are excluded.

The historical May-29 cutoff/calendar moves to each campaign cutoff (Dec-29, Mar-29, Jun-29). W0/W1 retain complete calendar months, W2 starts three calendar months before the official period. OOF origins stop at least 144 hours before the campaign cutoff; final fits use the supplied period training mask. These changes and corrected new-data availability are explicitly recorded. Original ACE+2h feature timing is not recreated.

CLI: `--experiment xgb_d76_lead_moe_resid_dv_enlil --data PREPARED --raw-data RAW --out DIR --period F1 --threads N`; `--smoke` reduces fit rounds/origins/candidates, `--merge` assembles all three full periods. Fitted models, loss histories, OOF outputs, forecasts, actual calendars, input identity, libraries, and inference time are saved.

Seven focused tests pass: original mixture arithmetic, finite residual replay parity, seven fitted-model parity, 29-feature HSSPEAK formula parity, calendar bounds, a complete 28-model smoke, and query-target poisoning exclusion. Full production training is parent-owned and has not been run by this agent.

## Blocked: e1604_hsspeak25

The source `source_archive/e1604_hsspeak25.py` only consumes historically prepared matrices. Their construction is recovered in `leaderboard/experiments/f1_resubmit_20261026/hsspeak_f1.py`: `features()` requires a 72-hour **HUXt physical trajectory** in addition to E0472 paths and historical ACE. Six of its 29 columns are HUXt maximum, argmax and lead-1/24/48/72 values.

The new `df.parquet` schema has no HUXt trajectory columns. Historical `leaderboard/experiments/solar_lhx72_omni2_20261018/cache/huxt_earth.npz` exists, but using that old forecast cache would violate the no-old-prediction-reuse scope. Raw SUVI permission does not supply the HUXt simulation boundary conditions or a new simulation pipeline. A faithful HSSPEAK rerun requires fresh HUXt trajectories covering the training roll6 folds and current query grids, plus fresh three-seed E0472 roll6 paths. The pure 29-feature and Gaussian-bump functions are ported and parity-tested; they do not constitute a runnable HSSPEAK experiment. No generic substitute was made.

## Blocked: e2000fs_blk0prev

Pinned source recovered with `git show 52feefa58:scripts/seongeun/e2000fs_blk0prev/96cca83e4ef2e6aec127bb3105ee0003435d12802eed40ca79128b649f9285a2.py`.

It does not merely shift a complete feature row by 12h. It rebuilds strips from the SUVI cell table at `origin-12h` while keeping other features at the current origin. The script has undefined `main`, missing recipe helpers, and a strip loop that overwrites all five output channels on each speed iteration; literal execution would leave all channels based on the final speed. Consequently intended versus actually submitted strip semantics cannot be established from this damaged standalone source alone. A simple 12h row shift or generic current strips would change the registered algorithm. No runnable replacement was introduced.

## HSSPEAK dependency recovery: exact physical boundary blocker

A second recovery pass traced the old trajectory all the way to its physical inputs. This is an input-content blocker, not just an absent high-level column or missing training script.

1. `git show d55d9f6d6:experiments/solar_lhx72_omni2_20261018/huxt_extract.py` recovers the cache generator. It opens `research/expos/runs/E1466/tdwsahuxt-state/td-state-*/lead_fields.h5`, takes the outer-radius 128-longitude ring for every lead, and indexes each ring at `earth_lon_index`. Its output `traj` is the source of all six HSSPEAK HUXt features. It does not compute these from ACE or SUVI.
2. `leaderboard/lb-f1f2/docs/visualizations/e1466_td_wsa_huxt72/epoch_repro_seed0/code/tdwsahuxt_state.py:44` loads a causal history of 36-bin `v_wsa_kms` profiles and original magnetic-map availability records. Availability is `max(observation+2h, HTTP Last-Modified)`. Latest per-longitude observations are retained across gaps. `_new_model()` at line 129 initializes HUXt with that boundary at 21.5 solar radii and propagates to 215 solar radii; `_boundary_matrix()` supplies its time-varying longitude boundary.
3. `git show d55d9f6d6:experiments/solar_pivot_tdwsahuxt72_20261016/prepare_sources.py` identifies the required raw source as GONG **SW-012 6-hour magnetograms**, historically under `store/v2/gong-mrzqs`, plus their observation and fetch/publication receipts. These are a separate data source from SUVI imagery.
4. `git show d55d9f6d6:experiments/solar_inc_pfss_wsa_20261014/extract_pfss.py` provides the profile equation and upstream calculation. A 180×360 magnetic map is averaged to 18×36, mean-subtracted, passed through a PFSS solver, then traced at source-surface longitude and photospheric seed locations. This produces flux expansion `fs` and angular distance `theta_b` to the nearest closed-field footpoint. The WSA law is:

   `v = clip(267.5 + 410 * clip(fs,1,300)^(-0.4) * (1 - 0.8*exp(-(clip(theta_b,0,30)/2)^2))^3, 240, 800)` km/s.

The current new raw `data-v2/input/df.parquet` schema has no `gong`, `magnet`, `pfss`, `wsa`, `expansion`, `footpoint`, or `boundary` columns. Its SUVI intensity cells and coronal-hole areas do not specify the signed photospheric magnetic boundary, and the magnetic topology cannot be recovered uniquely from those image summaries or a one-dimensional ACE series. Computing a different ACE/CH-based boundary would replace the original experiment.

The active `.venv` also lacks `huxt`, `pfsspy` and `sunpy` imports (`astropy` is present); a local vendored PFSS package exists. This dependency state is secondary: even a solver alone does not supply the missing magnetic maps/availability history. No package installation was attempted.

Within the assigned constraint of no extra data source and no reuse of old forecast caches, a faithful fresh HSSPEAK trajectory is therefore unavailable. The recovery path requires separately authorized GONG maps with causal availability receipts, a functioning original PFSS/HUXt runtime, a regenerated trajectory field, then fresh three-seed roll6 E0472 paths and HSSPEAK heads. No hashes, original validators, full physics simulation, or old forecast reuse were executed in this investigation. No `special_hsspeak.py` runner was created because a runnable interface without those inputs would falsely imply support.

## HSSPEAK recovery after authorization of original GONG data

The user subsequently authorized GONG originals. The former missing-source authorization blocker is superseded. `special_hsspeak.py` now provides a complete standalone statistical pipeline, consuming **fresh** `input/huxt_features.parquet` from the independent GONG/PFSS/WSA/HUXt task. Its contract is unique UTC `origin` and `huxt_max`, one-based `huxt_argmax`, `huxt_h1`, `huxt_h24`, `huxt_h48`, `huxt_h72`. Required coverage is the eligible roll6 training origins and current official query origins; unsupported initial physics spin-up origins are not used by the peak heads.

Further source recovery corrected an important shorthand in the earlier description: the head-training controls in `solar_hsspeak25_20261020/run.py` are **E1508/E1529**, the E1149 **L72-COND augmented analog** pipeline. They are not plain E0472 predictions. The new runner therefore rebuilds the chronological level bank from 2023, causal 18-feature Ridge level heads, 112-feature augmented seven-expert mixtures, and original 14-hour-hindcast analog replay with official calibration-window rules. The six historical folds are retained, restricted to training labels available under the current period's 144-hour maturity mask. Head features use the **three-seed mean** control path, matching the recovered source.

Official query features use individually retrained three-seed **E0472 release** paths, matching `f1_resubmit_20261026/e0472_f1.py` / `hsspeak_f1.py`. The two HistGradientBoosting heads retain 500 iterations, learning rate .04, depth 4, leaf minimum 50, and random state 0. The Gaussian correction retains width 12, shrink .15 and lead-25 onset; corrected per-seed paths are averaged. Every model, Ridge head, OOF component forecast, head-training matrix, and final prediction is newly computed. No old forecast caches are loaded.

CLI: `--experiment e1604_hsspeak25 --data PREPARED_TRAINY_FFILL --raw-data RAW --huxt-features FRESH_PHYSICS --out DIR --period F3 --threads N`; `--smoke` uses two boosting/head rounds and small grids; `--merge` requires full three-period outputs. Full training is deliberately parent-owned. The nested dependency graph is substantial: the complete synthetic three-seed/six-fold smoke fits 1,050 XGBoost boosters, 60 Ridge models and the two final heads, producing 18 fold/seed forecasts. Initial full smoke passed in 40.6 seconds. Pure-source parity covers the 29-feature ordering, Gaussian bump, analog replay and 18-feature level summaries; query-target poisoning confirms query `y` is excluded from training inputs and outcome series.

The physics task owns producing and validating the fresh GONG-derived artifact. Statistical pipeline readiness does not claim that physics or full production training/publication has completed.

Final statistical-runner verification: eight tests pass (seven source/causality/merge tests plus the complete three-seed/six-fold smoke). Restoring the original LevelBank terminal date of 2026-05-29 avoids an unused extra bank interval: the final smoke fits **1,008 boosters, 57 Ridge models, two peak heads** in 31.2 seconds. Production retains 900 boosting rounds and 500 head iterations maximum; smoke only uses two. Original `e0472_ablation_hpo/recipe.py` explicitly sets `device='cpu'`, so no CUDA backend was introduced. Per-engine models are reused by exact cutoff/augmentation state within a fixed seed/input context, and per-model origin-level values use the original cache semantics. Cross-period merge rejects differing training/raw/physics snapshot identities. Ruff and Python compilation pass. Full production physics, training, and submission remain parent-owned.
