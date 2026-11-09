# Research: training-only early stopping on the frozen 900-round parent

## Request Type

Bounded original-paper and version-specific implementation-reference lookup.
Retrieved 2026-11-08. No fitting, evaluation-truth access, model/data edits,
package installation, or new experiment execution was performed.

## Direct Answer

Training-only round selection is a defensible **unverified regularization
hypothesis**, not an established improvement or a route guaranteed to halve
error. Preserve the unchanged 900-round baseline and all its features, custom
objective, calibration formula, cutoffs, purge, seeds and scoring policy.
The proposed single changed component is the rule choosing boosting rounds
from the already-existing pre-cutoff prior/calibration procedure. Do not adopt
the failed age repair, replace features, select 300 rounds from the diagnostic
table, add internal experiment-selection folds, or tune against F1/F2/F3 truth.

The leader must preregister the monitored metric, patience, maximum rounds,
per-head selection rule, and whether selected rounds affect prior and full
fits before implementation. A coherent round-transfer test selects a count
on the prior fit, uses that prefix for its existing offset calculation, and
freshly trains the full fit for that count. This changes the round policy;
it does not justify additional calibration, objective or feature changes.
Full-refit transfer itself remains an empirical hypothesis, not an API promise.

## Local Evidence and Its Limits

The supplied [mechanism audit](../../research/w50_age_retrain_20261108/age_repair_mechanism.md)
reports prior-fit RMSE falling from approximately 56.7 to 47.4 over 300–900
rounds while existing calibration RMSE increases from 90.5–90.8 to 95.1–95.6
for all three seeds. The prior fit ends 2025-09-17; calibration origins span
2025-09-23 to 2025-12-23, with 2,185 rows and the unchanged 144-hour purge.
The >=700 target-pair fraction rises from 1.54% to 10.31%. This is consistent
with overfitting and distribution shift together; it does not isolate causes.
The full fit includes calibration rows, so prior-stage deterioration is not
proof that the final 900-round fit overfits its official forecast period.

The completed [age-repair results](../../research/w50_age_retrain_20261108/results.json)
supersede older “pending” wording in the mechanism/history documents:
three-seed mean official S is 97.8135546 for control versus 97.90254 for repair;
the paired relative-reduction 95% interval is [-0.00327912, 0.00055820], and
`adopted=false`. Preserve the fixed best-single reference S=97.49244208059969;
do not confuse it with the control mean or with a live SOTA measurement.
[Failure history](state_failure_history.md) identifies no inspected current-parent
experiment isolating early stopping. Older-parent routing failures do not test
this mechanism. These are supplied audit findings, not a new repository audit.

## Original-Paper Evidence

- **Zhang and Yu (2005), “Boosting with early stopping: convergence and
  consistency,” Annals of Statistics 33(4), 1538–1579.**
  [Original author reprint](https://arxiv.org/pdf/math/0508276),
  [DOI](https://doi.org/10.1214/009053605000000255).
  Studies greedy additive boosting, numerical optimization and statistical
  consistency as distinct questions. Early stopping restricts the fitted
  boosting path and acts as regularization. Its consistency results require
  i.i.d. observations and specified loss/search/step-size conditions. It supports
  the mechanism, not a theorem for this dependent, temporally shifted,
  24-output custom-objective XGBoost problem. No improvement magnitude transfers.
- **Kuznetsov and Mohri (2015), “Learning Theory and Algorithms for Forecasting
  Non-Stationary Time Series,” NeurIPS.**
  [Original proceedings paper](https://proceedings.neurips.cc/paper_files/paper/2015/file/41f1f19176d383480afa65d325c06ed0-Paper.pdf).
  Studies general non-stationary, non-mixing forecasting with bounded loss;
  its guarantees contain both sequential complexity and a sample-to-target
  discrepancy term. Reducing complexity does not eliminate distribution
  discrepancy. It supplies the temporal-shift caveat, not validation of a
  particular patience or round-transfer recipe. Its bounded-loss assumptions
  are not automatically satisfied by unrestricted squared error here.

Inference for this task: stopping can reduce unnecessary fitting to older
conditions, but may also remove useful learning and worsen extreme-target
attenuation. Regularization cannot create missing physical observations such
as transient eruption kinematics. No cited paper establishes a 50% gain.

## Official Docs Evidence

- [XGBoost 2.1 multi-output tutorial](https://xgboost.readthedocs.io/en/release_2.1.0/tutorials/multioutput.html)
  describes vector leaves through `tree_method="hist"` and
  `multi_strategy="multi_output_tree"`; support is experimental and incomplete.
- [Official multi-output example](https://xgboost.readthedocs.io/en/release_2.1.0/python/examples/multioutput_regression.html)
  combines a Python custom objective, explicit `num_target`, `xgb.train`,
  `evals_result`, custom metric and `inplace_predict`. It demonstrates two
  outputs, not this exact 24-output spectral objective or early stopping.
- [Prediction guide](https://xgboost.readthedocs.io/en/release_2.1.0/prediction.html#early-stopping)
  explains that native Booster prediction uses the full model by default;
  `best_iteration` and an explicit iteration range are needed for a best prefix.

## Version Note

Target runtime is XGBoost **2.1.4**, as reported by the supplied model audit.
No runtime import/version verification was performed in this research slice.
GitHub's `v2.1.4` tag resolved during retrieval to full commit
`62e7923619352c4079b24303b367134486b1c84f`.
Exa's extraction of the nominal 2.1.0 Python API page included 3.1.0 additions;
that extraction is version-contaminated and is not used to prove 2.1 behavior.
The exact-version upstream sources below resolve API details instead.

## Source-Reference Evidence

All references are `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f`:

- [`demo/guide-python/multioutput_regression.py:L68-L135`](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/demo/guide-python/multioutput_regression.py#L68-L135)
  explicitly exercises the custom-objective vector-leaf path; set
  `num_target=24` for 24 outputs and preserve gradient/Hessian shape semantics.
- [`python-package/xgboost/training.py:L52-L100`](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/python-package/xgboost/training.py#L52-L100)
  exposes `obj`, `evals`, `early_stopping_rounds`, `evals_result` and
  `custom_metric`. The last eval dataset and last metric control stopping.
  Put only the existing training-only calibration set in that controlling
  position; fitting-row diagnostics can be logged first. Use an explicit
  minimize direction for a loss metric.
- [`python-package/xgboost/training.py:L155-L192`](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/python-package/xgboost/training.py#L155-L192)
  passes raw margins to a custom metric when a custom objective is present,
  updates the supplied evaluation-history dictionary and returns the last
  model. Preserve the objective's output/link semantics in any monitor.
- [`python-package/xgboost/callback.py:L349-L355`](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/python-package/xgboost/callback.py#L349-L355)
  defaults to `save_best=False`;
  [`L417-L492`](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/python-package/xgboost/callback.py#L417-L492)
  stores the best epoch, retains later patience trees by default and implements
  `save_best=True` by slicing. Do not assume vector-tree slicing works without
  testing it. Save the unsliced model plus selected count if using prefix inference.
- [`python-package/xgboost/core.py:L2397-L2470`](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/python-package/xgboost/core.py#L2397-L2470)
  forwards `inplace_predict(iteration_range=...)` to native begin/end bounds;
  [`L2706-L2719`](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/python-package/xgboost/core.py#L2706-L2719)
  exposes `best_iteration` as a model attribute.

## Supporting Example and Required Compatibility Check

Illustrative API contract only: `selected_rounds = prior.best_iteration + 1`;
`prior.inplace_predict(X, iteration_range=(0, selected_rounds))` uses the
zero-based best iteration with an exclusive upper bound. A **fresh** full fit
uses `num_boost_round=selected_rounds` and no `xgb_model=prior` continuation.
Continuation adds trees to the prior model and is not the same estimator as
refitting on all eligible rows. Save the selected count separately: a full fit
without early stopping need not have `best_iteration`.

**Compatibility status: upstream component paths verified; exact combined
24-target early-stopping path not executed. Tiny synthetic smoke required.**
The implementation owner should run synthetic data only, CPU and
`CUDA_VISIBLE_DEVICES=""`, using the installed 2.1.4 build, hist/vector trees,
24 outputs, the unchanged custom objective, existing monitor convention and
a fixed small patience. Check gradient/Hessian shapes, finite `(n,24)` output,
history length, `0 <= best_iteration < num_boosted_rounds`, explicit prefix
prediction agreement before/after JSON save-load, and exact selected-round
count for a separate fresh fit. Include a deliberate synthetic plateau so
patience trees and off-by-one handling are exercised. This verifies software
behavior, not scientific transfer or performance. Do not use real evaluation
truth or add an internal selection fold for the smoke.

Do **not** call native `Booster.get_score`: the supplied audit records a
vector-leaf diagnostic segmentation fault. JSON `save_model` serialization
already worked there; it is distinct from `dump_model`/`get_dump` introspection.

## Caveats / Ambiguity Flags

- Choosing rounds on a calibration block then estimating offsets on the same
  block reuses training labels. It can be part of the declared fitting procedure,
  but that block is no longer an unbiased performance estimate. Report official
  F1/F2/F3 scores only after fitting; do not choose between stopping variants
  from their repeatedly exposed leaderboard outcomes.
- RMSE-based stopping need not minimize the unchanged tail/spectral objective
  or official S. Metric choice must be explicit, frozen and justified, rather
  than silently substituted. The papers provide no universal choice.
- The fuller, more recent training distribution can need a different number
  of rounds; reusing a prior count does not establish an optimal full-fit count.
- The 144-hour purge addresses the declared overlap/maturity policy, not every
  observation's publication-time availability. The audit leaves that incomplete.

## Reusable Takeaway

Keep the 900-round parent unchanged. A preregistered training-only round rule
is a supported hypothesis about regularization, with uncertain temporal and
full-refit transfer. Complete the tiny compatibility smoke before implementation;
no performance improvement, component adoption or 50% goal completion follows
from this literature review.
