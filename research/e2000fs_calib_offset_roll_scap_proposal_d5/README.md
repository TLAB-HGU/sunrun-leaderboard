# S-cap proposal depth 5 — E1

Fixed baseline: `e2000fs_calib_offset_roll_scap`, seeds 42/43/44, official F1/F2/F3. The sole intervention is continuation proposal `max_depth=3 → 5`, for both prior and full models. The 54 stored A900 depth3 reference arrays are reused directly, with byte-array equality, source/key provenance, serialized tree-depth, output-width, feature-count and round-count checks. No A900 refitting occurs in the full experiment. Smoke runs train two-round references and explicitly do not certify equality to A900.

## Strategy

Hypothesis: deeper shared-tree proposals can express tail-specific descent directions within the unchanged per-lead MSE cap. This tests this depth intervention, not the whole capacity hypothesis or an irreducible uncertainty floor. The frozen preregistration is [the experiment specification](../../analysis/amplitude_controlled_experiment_spec.md), with an external pre-execution snapshot in `~/sunrun-scap-proposal-depth/frozen-experiment-spec.md`.

Retained inputs: all 23 constant features (including `v_lag24/48/72`) and all eight varying features from the fixed baseline; expanded input width 215. No replacement or new feature. Retained training: filled labels, original cutoff and 144-hour maturity, 24-lead blocks, depth3 A900 references, point4+spectral baseline, 180 proposal rounds in six 30-round stages, stage seed increment, learning rate .03, subsample .8, colsample .7, min-child-weight 20, lambda 5, Fourier coefficient .1/scale 50, cap 1.01 and the original Armijo grid. Calibration algorithm stays fixed (last 336 training calibration origins, coefficient .25); its offset values are recomputed because prior predictions change.

Original observed labels are read only on the fitted training grid; their finite values must equal filled labels at the same positions. Prior/full call order and smoke subsampling mirror the shared runner and are checked against the exact fitted target matrix. Stage logs separate original-observed bins, imputed/observed cap consumption, raw proposal/accepted movement, directional derivative, feasible lead bounds, hypothetical unconstrained alpha and spectral terms. The hypothetical unconstrained alpha is never applied. Evaluation labels do not enter the fitter.

## Prespecified decisions

Activation requires at least 9 of 27 full blocks to have one stage with alpha > 1e-8 and accepted RMS >= 1 km/s. Mechanistic support requires observed training >=700 km/s raw RMSE improvement >=5% in at least two periods (seed mean), at least two concordant seeds in those periods, and no worsening of absolute tail bias. If inactive, do not reject the capacity hypothesis. If active without this endpoint, reject only this intervention. Offset-only gains do not support the proposed raw-capacity mechanism.

Practical candidate gate: mean3 official S <= 94.54311921240927, all matched seeds improve S, pooled MAE/official RMSE and each period S worsen by at most 1%. Report paired 7-day circular-block bootstrap (2,000 draws, RNG 20261113) separately from seed variability. Do not count gap-fragmented observations as independent events or claim independent physical validation. No automatic baseline promotion, depth sweep, coefficient fitting or publication.

## Execution and artifacts

CPU only: `CUDA_VISIBLE_DEVICES='' .venv/bin/python research/e2000fs_calib_offset_roll_scap_proposal_d5/experiment.py --seeds 42 --periods F1 F2 F3 --device cpu --threads 6 --observations PINNED_FILE --out CAMPAIGN_RUN`.

The shared `_module/gbm.py` runner owns calibration, replay, scoring, metadata and standard `result`/weights layout. Campaign orchestration, logs and validation scripts live outside the repository under `~/sunrun-scap-proposal-depth/`. `continuation.diagnostics` is written in each `fit` JSONL record; outer period, seed stream, prior/full stage and block identify the fitted unit. Full-run completion requires all nine seed-period units and the frozen mean3 comparison; launching is not completion.

## Evidence and limits

References: [XGBoost multi-output](https://xgboost.readthedocs.io/en/release_2.1.0/tutorials/multioutput.html), [tree parameters](https://xgboost.readthedocs.io/en/release_2.1.0/parameter.html). These document the shared tree partition and depth control; neither establishes that depth5 improves this dataset. Original diagnosis and competing H1/H2/H3/H4 mechanisms remain in the frozen specification.

Implementation verification: synthetic old/new default-depth predictions, UBJSON stage arrays and stage records agree exactly; enabling a no-op callback preserves them. Invalid proposal depths are rejected. Full experiment outcomes remain pending until the campaign completion report.

## Completed E1 results

2026-11-13T05:33:25.981422+00:00 UTC. Local validated fixed mean3 results.

| Model | Total S | TIMSE | RMSE |
|---|---:|---:|---:|
| e2000fs_calib_offset_roll_scap | 95.498100 | 19174.020267 | 67.720438 |
| e2000fs_calib_offset_roll_scap_proposal_d5 | 95.586285 | 19177.452202 | 67.865001 |

Mechanism: training_supported_official_tail_not_improved; activated full blocks=27/27; candidate gate=False. No automatic baseline promotion. Full report in research/result/e2000fs_calib_offset_roll_scap_proposal_d5/logs/e1_report.json.
