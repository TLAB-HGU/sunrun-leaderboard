# Mechanism checks while training continues

These checks do not use official target arrays or change any fitted setting.
No new score or adoption conclusion is available yet.

F1 seeds42/43: six900-round models each reproduce full query predictions,
prior-calibration predictions and training-only offsets with max differences0.
The2,185calibration rows and144h maturity/purge are independently verified.
Thus feature omission, stored-output inconsistency and calibration arithmetic
failure are not supported for these completed periods. This is not proof of
optimal convergence or validity of every observation's publication time.

Serialized F1 seed43 full trees contain520parent img_age_h splits and515after
repair(2.776% vs2.751% of branch occurrences across three heads). Every28source
feature is retained and actually used in both configurations. Counts are not
gain, causal attribution or evidence of improved skill. XGBoost2.1.4's
get_score diagnostic crashed a separate inspection child(return−11) on this
vector-leaf model; it was abandoned. Supported JSON model serialization worked.
The training processes were unaffected. See inspection-failure.json and
runs/age_retrain_s43/F1_split-inspection.json.

The input-only post-hoc event exposure check finds63changed F1 query origins
out of2,089, but **zero** changed inputs among96origins January18–21. Their
maximum repaired age is0.02058h. This repair therefore adds no direct new
query information in that dominant previously diagnosed CME window. Fresh
training may still change predictions globally, so this is neither a score
result nor proof of no benefit. The window was selected after earlier residual
analysis; it is not an untouched causal test. Reproduction and exact counts:
../../analysis/w50_audit_20261108/age_repair_event_exposure.py and adjacent JSON.

This limits the physical reach of the age hypothesis: repairing stale-image
metadata cannot by itself supply missing transient eruption kinematics.
Whether added versioned eruption information helps remains untested. Complete
the preregistered multi-seed experiment and diagnose its scored outcome before
choosing the next change. Do not replace or remove any baseline feature based
on these split counts.

F1 seed43 saved-tree prefix diagnostic on the original fitting rows:

|Trees retained|Training RMSE|Training bias at target>=700|
|---:|---:|---:|
|1|97.20|−308.31|
|100|67.38|−135.23|
|300|60.47|−89.51|
|600|54.96|−70.06|
|900|51.00|−58.41|

The model learns substantially, while high-speed attenuation persists even
on fitting data. Training weighted MSE continues to decrease between600and900
(3401.85→2845.09); this does not establish that more rounds would generalize,
that the full spectral objective is unconverged, or that optimization failure
is confirmed. No rounds or parameters are selected/changed. Original fitting
rows are not a new internal validation fold. Exact inputs/checkpoints are in
training_fit_diagnostic.py and runs/age_retrain_s43/F1_training-fit.json.

Separate restricted residual-energy calculations show that perfect removal of
only the previously diagnosed Jan19 event errors, with all others fixed, leaves
baseline seed42 S76.98. Removing only all>=700km/s error terms leaves61.47;
removing only lead49–72 terms leaves61.89. All exceed the50% target48.75. These
are mathematical sensitivity scenarios, not model outputs, achievable bounds,
or feature/parameter fits. They motivate addressing broader error structure
without treating one extreme event as the entire problem. Protocol and all
seed outcomes: ../../analysis/w50_audit_20261108/residual_energy_budget.md.

The existing training-only **prior-stage** offset-calibration procedure gives
a different curve. All times below precede the F1 cutoff. The prior model was
fitted through2025-09-17 and its existing2,185calibration origins run from
2025-09-23 through2025-12-23; the144h purge is unchanged.

|Seed|Prior fitting RMSE300→900|Existing calibration RMSE300→900|
|---:|---:|---:|
|42|56.66→47.38|90.72→95.63|
|43|56.74→47.45|90.81→95.45|
|44|56.66→47.51|90.52→95.13|

The split's>=700target-pair fraction changes from1.54% in the fitting rows to
10.31% in calibration. These are repeated origin/lead pairs, not independent
event counts. Transfer deterioration is reproducible across seeds and coexists
with a large temporal state shift. This supports investigating prior-stage
overfit/distribution shift rather than assuming a simple lack of training.
It does not causally separate those explanations. The final full model also
fits the calibration rows, so these prior-stage curves do not prove that the
final900-round model overfits the official query period or that300is an optimal
final round count. No epoch/hyperparameter was selected or changed.

Reproduction: prior_fit_diagnostic.py on each saved seed's F1 models; exact
fixed checkpoints and ranges are in runs/age_retrain_s{42,43,44}/F1_prior-fit.json.
This is an audit of the original fitting/calibration procedure, not an added
internal experiment-selection fold or a new performance comparison.
