# e2000fs_calib_offset_roll_rcdials_noimg24_rawdf (jsy301)

Rank-1 recipe `e2000fs_calib_offset_roll` on the team raw-only df with the DirMO lab input fill, **with leads 1-24 trained without image columns and two regime dials** chosen per period on the pre-cutoff calibration quarter: the tail weight (all three blocks) and a recurrence pull on leads 25-72.

## Base: rank 1 on our data

The recipe is the team's rank-1 entry `e2000fs_calib_offset_roll`. Its functions `design`, `spectral_objective`,
`prior_split`, `native_head`, `estimate_state`, `apply_state` and `subset`, and its `BASE_NAMES` / `FEATURES` /
`PARAMS` / `OFFSET_*` constants, are copied into `experiment.py` unchanged. Per period F1 / F2 / F3:

- **Training rows:** the df's `fit_P` flags (origin + 144 h <= cutoff). The query rows are read without any y column.
- **Inputs:** feat30, 21 constant + 24 x 9 lead-varying columns per 24-lead block (237), team column order.
- **Training y:** the fit rows' own targets forward-filled along target time (last observed hourly speed; team rule).
- **Model:** three multi_output_tree XGBoost 2.1.4 blocks (depth 3, lr 0.03, 900 rounds, PARAMS seed 42 only),
  spectral loss (weight on observed y >= 550 km/s, normalised per lead, + 0.1 x pseudo-Huber term on the FFT of
  the 24-lead error).
- **Prior model and offset:** the same recipe on the rows with origin + 144 h <= boundary (boundary = last training
  origin - 3 months) predicts the calibration quarter (origins >= boundary). The frozen offset per lead is 0.25 x
  the mean residual of that prior over the last 336 calibration rows. It is added to the full model's query forecast,
  then clip 1-3000.

**The one difference from rank 1 in every entry of this family is the input fill.** We use the team raw-only df
with the DirMO lab gap fill, as in our DS entry `dirmo_24_feat30_pw4_d12_stack25_rawdf`: causal last value along the
hourly origin timeline, solar-wind observations without limit, image columns (suvi_*, strip_*, img_age_h) for at most
24 h with img_age_h grown by the hours carried, ENLIL columns not filled. Rank 1 fills every column without limit.

## Block 1 without image columns

Block 1 (leads 1-24) of both the prior and the full model is trained without the image-derived columns: the 7
suvi_* areas, img_age_h and the 5 strip_* inputs. That leaves 109 inputs instead of 237. Blocks 2-3 (leads 25-72)
keep the rank-1 inputs. The offset of every lead comes from its own block's prior.

## The dials (chosen inside the entry, per period)

| dial | grid | where it acts |
|---|---|---|
| tail weight t | 2, 4, 6 | weight t instead of 4 on y >= 550 km/s in the spectral loss, prior and full model (all three blocks) |
| recurrence pull w | 0, 0.1, 0.2, 0.3 | leads 25-72 after the offset: forecast + w (rec27 - forecast); no pull where rec27 is missing; clip 1-3000 |

`rec27` is the df's lead-varying rec27 input of the same origin, after the fill.

**Selection rule.** For each t the prior model (weight t) predicts the calibration quarter and gets its own offset.
For each w the pull is applied. S is the board metric over all 72 leads (official per-lead formula). It is computed
on the calibration quarter's 00/06/12/18 UTC origins against the raw, unfilled training speed, with missing hours
left out. (t*, w*) is the argmin; ties go to the smaller t, then the smaller w. The full model is trained with t*,
and the query forecast gets offset_{t*} and pull w*.

**Selection of (t, w) uses only pre-cutoff training data (the prior model's calibration quarter).** No query truth
is read, and no placebo or control arm is part of the entry. Each run stores the 3 x 4 S table and the choice in
`artifacts/<P>/selection.json`.

## Why it is submitted, and how to read the board score

- **Pre-registered** in jsy301 DirMO lab `plans/post_freeze/01_night_run.md` (combination of rounds RC and NI), before any number of
  this run.
- **Submitted because its team-board dry-run Total S was below 101 (user rule 2026-11-08); lab gate result:
  fail for both components (round RC and round NI gates failed, see the two single entries); the combination is not a formal lab arm and is submitted under the user rule only.** The dry-run read is a selection on board truth. This is disclosed here and in the description.
- **No other board number chose anything** in this entry.
- **Seed 42 only.** Rank 1's own seed-42 board point sits inside a seed spread of about 0.8 S (public board records of the same recipe with seeds 42 / 43 / 44: 98.763 / 99.932 / 100.213, sd 0.77; see `e2000fs_calib_offset_roll_abl2_baseline_s43/s44`), so a single-seed
  difference below that is not a measured effect.
- **Lab gains have not transferred well** to the board so far (rounds MW, HW1, D12, DS, TC2, BT).

## Artifacts of a run

Each run writes `artifacts/<P>/` per period. It holds `frozen_calibration_t<t>.json` (offset state per tail weight),
`prior_predictions_t<t>.npz`, `selection.json`, `timing.json` (fit seconds per block) and the model files.
`python experiment.py --df <df.parquet | split dir> --out predictions.parquet` reproduces the run. `--smoke` runs
2 rounds on subsampled rows and is a code test only.

## Folder layout in this repository (2026-11-09)

- `experiment.py` — the submitted script, unchanged (single file; data → training → inference).
- `results/predictions.parquet`, `results/score.json` — the stored leaderboard prediction and its official score record (`tlabtlab/sunrun-lb-store`).
- `logs/leaderboard.jsonl` — the stored submission record as JSON lines.
- `weights/` — empty. The trained models and run logs of this entry exist only on jsy301's machine (none on this host or golab2); a rerun of `experiment.py` recreates them.
