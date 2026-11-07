# User-requested combined negative-mean ablation

Frozen before training on 2026-11-07. Parent: original e2000fs_calib_offset_roll A, no B1 blend. The user explicitly requests combining every negative-mean item shown at 17:08 KST and publishing the result. This is an exploratory combination selected after viewing official ablations; the same official periods are not independent confirmatory evidence. Some ingredient effects were incomplete or seed-inconsistent. No independent contribution or guaranteed cumulative gain is claimed.

## Fixed hypothesis and intervention

Combining the eight source-feature removals, alpha=1 offset and Hessian-only spectral control may improve A jointly. Remove exactly: ch_ch_t035_equatorial_area, strip_v450_logmed, rec27, strip_v650_dark, ch_ch_t055_meridian15_area, ace_density_cm3_last, strip_v750_logmed, ace_density_cm3_m6. Retain the other 22 original source features. Keep tail weight 4, offset enabled/horizon scope/336h window, block24, both clipping operations and 900 boosting rounds. Use spectral=True, spectral_hessian_only=True, alpha=1.0. Seeds 42/43/44 are fixed; their predictions are averaged with weights 1/3 without fitting weights or selecting seeds.

## Protocol and interpretation

Train only before official F1/F2/F3 cutoffs, 144h label maturity/purge and training-only frozen calibration. No internal validation periods. Official observations only score fitted predictions. Use the same official scorer and observation revision as the full ablation campaign. Report mean3 against original A and the previously posted single-removal mean3. Improvement requires lower official S without material deterioration of companion metrics; otherwise report failure and retain the prior baseline. User explicitly requests publication regardless of whether this exploratory combination improves; do not tune/revert ingredients based on its score.

Existing evidence/reference source: ../offset_roll_complete_ablation_20261106/strategy.md and its frozen source, official per-seed results. No new external mechanism or literature claim is introduced.

## Execution and completion

GBM CPU only, CUDA_VISIBLE_DEVICES='', four threads/job. One new job per host, at most six actual global CPU training processes and 24 declared CPU threads, require 11GiB MemAvailable before a new job to reserve 6GiB after estimated 5GiB allocation. Existing experiments are preserved. golab: seed42; golab2: seeds43 then44. Each run has independent tmux/logs and full source/config/data provenance. Completion requires exit0, consistent 454680-row official prediction grids, metadata, score and validation. Collect remote results locally; publish fixed mean3 with architecture/replay and full visualization; only then campaign-complete.json.
