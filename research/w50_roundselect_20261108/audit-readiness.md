# Independent audit readiness — F1 only

2026-11-09 KST. **Ready to continue the frozen protocol; not ready for a final
campaign integrity verdict or performance claim.** No new protocol blocker was
found in this bounded review. All three local `train.exit` receipts were absent
at inspection. This is an expected incomplete state, not a failed experiment.

This review reads receipts, source and preregistration only. It does not rerun
model inference, read treatment local_score files, inspect official prediction
performance, open query truth, run the goal evaluator, or modify active fits,
runtime, models, evaluator or another author's files. The reviewer previously
authored the evaluator and its tests; this readiness note is not an independent
validation of that author's own implementation. The eventual scientific audit
must ground its assertions in fresh run evidence.

## F1 integrity already supported

Each `runs/roundselect_s{42,43,44}/F1_model-replay.json` exactly matches its entry
in `verification-partial.json`. The latter correctly remains `status: partial`
and `goal_achieved: false`.

| Seed | Selected full-model head counts | Prior head counts | Maximum replay differences |
| --- | --- | --- | --- |
| 42 | 315 / 152 / 135 | 900 / 900 / 900 | 0 for every checked prediction/offset comparison |
| 43 | 282 / 159 / 133 | 900 / 900 / 900 | 0 for every checked prediction/offset comparison |
| 44 | 304 / 150 / 108 | 900 / 900 / 900 | 0 for every checked prediction/offset comparison |

The zero differences cover frozen-parent 900-round predictions, selected prior
predictions, offsets, full raw forecasts and final clipped forecasts. Every F1
receipt reports 1,706 stopping rows, 336 offset rows, verified fitting subsets
and purge, and `query_truth_read: false`. Selected native-RMSE replay differs
from the logged value by at most 2.803e-8 across the nine heads.

Read source `verify_selected_replay.py` implements actual saved-booster round
and 212-column checks, first-minimum selection, independent offset arithmetic,
array replay and original-label/origin equality. These findings establish what
the existing successful replay receipts support; this readiness review did not
load the boosters again. The final reviewer should rerun or independently
reproduce those checks against the final collected artifacts.

Direct current-versus-executed-snapshot byte equality passes for experiment.py,
the parent's core.py and baseline.py, and the official scorer. No hash was
computed. The executed strategy snapshot differs from the current strategy
only by appended runtime/F1 evidence; the frozen rule and gates are unchanged.
The executed-source manifest was captured at 15:27:23 UTC, after launch, so its
capture time must not be mislabeled as prelaunch. Relevant per-run numerical
versions were recorded at start; the fuller environment snapshot was captured
at 15:27:48 UTC.

## Evidence required for the final audit

First collect all three successful exits, completion receipts and all nine
seed-period artifact sets without changing their contents. Then complete the
following checks, using the evaluator's exact `REQUIRED_CHECKS` names:

| Check | Evidence needed before marking passed |
| --- | --- |
| `causal_fitting` | Actual training/query loader contract; all nine cutoff/maturity checks; original reference/calibration subsets; stopping counts 1706/1682/1730 and inclusive 144-hour separation; final 336 rows excluded from count selection; no query target used in fitting. Distinguish computation chronology from historical public availability. |
| `fresh_training` | Launch commands, source paths and logs showing fresh prior/full fits for every head; metadata forbidding reuse; all exits zero and matching completion seeds; 54 saved boosters across nine periods, with 27 unchanged 900-round priors and 27 selected-count full models. |
| `single_factor_diff` | Frozen parent/source comparison and actual executed source; only effective tree-count policy changed, no age repair, objective change, offset-policy change or hidden feature replacement. Preserve normal per-subset base_score and custom-objective weights. |
| `feature_identity` | Actual matrices and ordered feature receipts for all periods: original 28 source/212 expanded columns, original img_age_h, same two exclusions, training rows/ranges and model parameters. Metadata equality alone is weaker than input equality. |
| `official_score_replay` | After every run completes, independently reassemble NPZ forecasts and verify parquet equality, official scorer version and frozen observation identity, zero scorer replacements, and fresh total/F1/F2/F3 S/RMSE/TRMSE/TIMSE/MAE parity. Reproduce the fixed seed-42 parent S anchor. |
| `model_replay` | Repeat F1-quality checks for F2/F3 and bind all nine replays to final artifact identities. Load all saved models to verify actual rounds and features; verify complete 900-value finite RMSE histories and first-minimum counts; reproduce selected-prefix predictions, prior900 controls, independent offsets and final predictions. |
| `provenance_complete` | Executed source snapshots and semantic diff, explicit source-commit versus actual-host-commit distinction, dataset/store revision and byte/time receipts, launch-time numerical versions plus separately timed environment snapshot, and documented upstream availability/corroboration evidence. Do not silently clear known unresolved provenance requirements. |
| `device_policy` | All launch commands use CPU/four threads; existing live-process receipt records empty CUDA visibility for all three PIDs. Preserve resource admission and host ownership evidence, and ensure the completed artifacts correspond to these runs. |
| `preregistration` | Executed prelaunch strategy/source/feature lock, local and remote synthetic preflight evidence, frozen controls and protocol review, and proof that later appended diagnostics did not change selection, gates or parameters. |

All-seed analysis also needs `results.json` and `bootstrap.npz`: 2,000 paired,
period-stratified circular 168-hour origin-block draws, RNG 20261104, empirical
TIMSE quantiles recomputed, saved draw/CI consistency and positive lower 95%
gain bound. Verify matched mean gain >=1%, improvement at every seed, MAE/RMSE
mean deterioration <=1%, and each period mean-S deterioration <=2%. Separately
enforce every candidate S <=96.5175176597937 for promotion. The stronger goal
remains every single-model S <=48.746221040299844; neither a seed ensemble nor
the best seed substitutes for this. Preserve a dated SOTA snapshot separately
from the fixed-best comparator. Conditional bootstrap is not unseen-future
confirmation and does not erase past benchmark selection.

## Exact evaluator audit contract and identity inventory

For numerical goal success to become final PASS,
`verification-independent.json` must contain `status: "pass"`,
`integrity_pass: true`, a nonempty `reviewer`, explicit
`unresolved_required_gaps: []`, and no nonempty `gaps`. Every required check
above needs `passed: true` and a nonempty list of existing evidence files.
Do not manufacture a pass-shaped receipt to satisfy the evaluator.

`inputs` must cover the **168 unique paths** currently returned by
`integrity_files(root, runs)`, with exact absolute `path`, `bytes` and `mtime_ns`
receipts; no duplicate paths or hashes:

- Six global files: results.json, bootstrap.npz, strategy.md, experiment.py,
  fixed official scorer and fixed observations.
- Six per seed: train.exit, complete.json, predictions.parquet, provenance.json,
  resolved_config.json, and its matched parent's predictions.parquet.
- Sixteen per seed-period: final predictions NPZ, metadata, parent metadata,
  campaign frozen parent prior_predictions NPZ; control.json,
  tree_selection.json, prior900_predictions.npz, prior_predictions.npz,
  fitting_subsets.npz, frozen_calibration.json; and all six prior/full boosters.

The inventory is the minimum required set. Include additional reviewed replay
reports, raw predictions used for replay, launch/process receipts, source
snapshots, shared source files, analysis/replay scripts and preflight records
in `inputs` as needed to bind actual evidence to its reviewed contents. The
evaluator allows this superset. It checks model-file presence and metadata;
the independent `model_replay` audit must establish actual model behavior.

Freeze documentation before creating audit identities: appending to strategy.md
afterward correctly invalidates its receipt. If artifacts are collected again
or materially changed, refresh the affected substantive checks before producing
new receipts; merely refreshing timestamps is not re-verification.

## Issues and limits to carry forward

- The pre-offset diagnostics added during fitting are descriptive checks on
  existing fitting subsets. The strategy explicitly disclaims independent
  validation and no fitted rule changed. No protocol violation is established.
  These observations cannot justify adapting the current campaign or claiming
  official improvement.
- All three run provenance files record `actual_git_commit: null` and
  `git_commit: null`; `source_commit` is a separate supplied source reference.
  Executed snapshots and direct equality improve reproducibility but do not
  turn a non-Git training workspace into a verified Git checkout. The final
  audit must state exactly what is proven, using the preserved snapshot chain.
- Historical upstream issue-time availability/corroboration and the previously
  recorded hash-evidence restriction remain explicit. Do not compute forbidden
  hashes or infer historical availability from current bytes. If an applicable
  required provenance claim remains unproven, record a required gap and keep
  overall integrity PARTIAL even if implementation checks pass.
- The evaluator intentionally returns FAIL immediately when complete, freshly
  scored forecasts miss the 50% threshold. That verdict is a numerical goal
  failure, not an independent scientific audit. The planned final audit and
  failure diagnosis still need completion; they must not be skipped because
  the early FAIL branch does not inspect them.

Stop condition for this note is met: F1 receipt consistency and source continuity
are checked, the final audit requirements are concrete, and incomplete work is
explicit. No whole-campaign completion, performance improvement, publication
eligibility or final goal success is asserted.
