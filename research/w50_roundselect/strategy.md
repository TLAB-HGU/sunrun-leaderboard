# Training-only tree-count selection: preregistration

2026-11-08, before implementation or fresh fits. Previous image-age replacement
finished: control meanS97.8135546, treatment97.9025353, mean paired reduction
−0.09144%,168h block CI[−0.32791%,+0.05582%]. Independent implementation checks
passed; the component is not adopted. See ../w50_age_retrain_20261108/results.json
and verification-independent.md. This is the next diagnostic-driven experiment,
not a guarantee of improvement or50% attainability.

## Frozen A, retained components, single B

A is drop035_v650_confirm_20261108, matched seeds42/43/44. Fixed best single
S97.49244208059969, mean97.81355461658598; fixed50% threshold48.746221040299844.
Use original28source features/212expanded columns in the same order, including
the original img_age_h values. The failed age repair is not accumulated.
Keep original imputed training targets, all training rows, depth3,lr.03,
subsample.8,colsample.7,min_child_weight20,lambda5,tail4@550,spectral.1/scale50,
three24h vector-leaf heads,seed,CPU4threads,component/final clips[1,3000],
training-only horizon offsetalpha.25,last336calibration rows. No B1.

B changes only the **effective boosting tree-count rule**. For every period
and24h head, freshly fit the unchanged prior model for900rounds and record
native unweighted RMSE after each round on the stopping subset defined below.
Choose the first minimum over rounds1…900, with no patience parameter or other
search. Use that exact prefix for prior predictions feeding the original
offset fit; freshly train the full model on all original training rows for
the selected count. The prior900-round model is saved intact. Its default
prediction is never mistaken for its selected prefix. This is train-only
tree-prefix/round selection regularization; it does not promise shorter prior
training, select a fixed300rounds from the diagnostic, or change the objective.
Full refit transfers the selected count without rescaling for sample size.

## Fitting-only temporal subsets

Keep original baseline.prior_split: last-three-month calibration,144h reference
purge. Keep its final336rows exclusively for the unchanged offset calculation.
Stopping rows come from the existing calibration origins satisfying
origin+144h<=first_offset_origin. This leaves a maturity gap between stopping
and offset subsets. No query target is loaded, no F0/internal experiment score
is created, no threshold/metric/window is selected using F1/F2/F3 outcomes.
The stopping subset is part of model fitting, as permitted for training-only
epoch selection. Full fitting still uses every original fit_F{n} row.

Expected stopping counts under unchanged official grids:1706/1682/1730 for
F1/F2/F3; record and verify exact ranges instead of silently accepting mismatch.
Offset rows remain336. Cutoffs2025-12-29,2026-03-29,2026-06-29 and144h maturity
are unchanged. Previous periods may enter later official training as already
allowed; this does not make later periods independent blind tests.

## Evidence, mechanism and falsification

Fact: all three prior-stage F1 fits decrease fitting RMSE300→900 from~56.7to47.4
while the existing training-only calibration RMSE rises~90.5–90.8to95.1–95.6.
The>=700target-pair share changes1.54%→10.31%. This is consistent with overfit
and temporal regime shift; it does not separate them. The full model fits
calibration rows too, so successful round transfer is an empirical hypothesis.

Mechanism: limiting fitted-path complexity may improve transfer under changing
solar-wind state distributions and reduce broad moderate-state errors. It adds
no physical observations, cannot reveal an unobserved eruption, and may worsen
tail attenuation because stopping monitors RMSE while training retains tail/
spectral loss. Needed information is the same origin-causal inputs and mature
pre-cutoff targets. Applicability requires the stopping interval to contain
useful transferable structure; this is not guaranteed under further shifts.

Refutation: with valid controls, selected shorter paths do not pass the frozen
official multi-seed improvement gate, or RMSE improves while officialS/tail
performance fails it. If all selected counts remain900, report a null policy
intervention. Do not respond by changing metric, maximum, window, patience or
features against the exposed benchmark. Diagnose the failure before another
hypothesis. Existing older-parent cluster/routing failures do not test this
same component; see ../../analysis/w50_audit_20261108/state_failure_history.md.

Original/official evidence, retrieved via Exa and pinned2.1.4 upstream source:
../../analysis/w50_audit_20261108/early_stopping_evidence.md and sources.json;
Zhang&Yu2005 https://doi.org/10.1214/009053605000000255 ;
Kuznetsov&Mohri2015 https://proceedings.neurips.cc/paper_files/paper/2015/file/41f1f19176d383480afa65d325c06ed0-Paper.pdf ;
https://xgboost.readthedocs.io/en/release_2.1.0/python/examples/multioutput_regression.html .
No theorem there establishes benefit for this dependent, shifted custom loss.

## Preflight and control checks

Before actual fits, run a tiny CPU synthetic24-output spectral-objective smoke
covering evals_result, all900-style logging semantics at reduced smoke budget,
first-minimum selection and inplace_predict(iteration_range). Verify adding an
evaluation matrix alone leaves final model predictions identical to fitting
without it. No synthetic score is a scientific result.

Record semantic/source diff and actual retained feature lists before launch.
Compare each newly trained prior900 endpoint on all original calibration rows
with the frozen parent's saved prior_predictions.npz (max difference<=1e-6).
This verifies unchanged training path/environment before applying selection.
A failure is a control/protocol diagnosis, not a performance conclusion; do not
silently relax this check. Saved full models must have selected_rounds trees;
saved prior models900trees, with inference prefix recorded. Independently replay
calibration offsets and final forecasts after fitting.

## Official gates and uncertainty

Only official F1/F2/F3 and unchanged leaderboard/scorer/competition.py plus the
same frozen observations. Freeze all period forecasts before scoring a run;
matched analysis waits for all three seeds. Report S,RMSE,TRMSE,TIMSE,MAE per
period/total and per-seed mean,variance,SD,worst. Never assemble a seed ensemble
or choose the best treatment seed. Report fixed-best and current-SOTA comparisons
separately, with an immutable board snapshot.

Candidate gate: mean matched relative reduction>=1%, every seed improves,
paired period-stratified circular168h origin-block bootstrap2000 lower95% gain
bound>0 (RNG20261104; empirical TIMSE quantiles recomputed), mean MAE/RMSE
worsening<=1%, each period meanS worsening<=2%. Fifty-percent goal remains the
stronger separate fixed threshold48.746221040299844 for all three single models
with verified uncertainty and integrity. Failed components are not promoted.

Before any promotion/submission, additionally enforce the AGENTS existing-best
single-model1% criterion: **every candidate seed S<=96.5175176597937**, which is
.99times the frozen best single97.49244208059969. A1% gain against the scalar
three-seed control mean alone is insufficient. This conservative all-seed bound
prevents choosing a lucky single seed; all matched/CI/auxiliary gates still
apply. This clarification is fixed before implementation or any new fit.

Repeated leaderboard exposure/selection bias remains; conditional bootstrap
does not provide unseen-future confirmation. Blind validation remains a separate
proposed protocol. Public-availability/observation-corroboration and previous
automatic hash-approval gaps remain explicit. Do not alter existing evaluation
policy, raw/scorer truth, or publish a failed/unverified candidate.

## Resources, artifacts and continuation

Check actual local/golab2 jobs,CPU and MemAvailable before admission. GBM CPU
only,CUDA_VISIBLE_DEVICES='',--device cpu,--threads4. At most3of these already
required seed replications per host, totalhost<=6jobs/24threads, require11GiB
available before each admission,reserve6GiB. No duplicate hypothesis jobs or
resource placeholders. Preserve others' campaigns. All fresh prior/full fits;
no fitted model or forecast reuse apart from frozen-reference comparisons.

Own tmux/log per run; registry/runtime/manifest or remote assignment/scheduler
layout, launch.json before execution, JSON period/block/unit/units progress,
all F1/F2/F3 predictions/metadata, trained boosters, tree-selection curves and
ranges, complete.json,publish:not_required until gates,train.exit. Verify sunrun
running/completed display. Save commit/environment/seeds/source snapshots,
data/store revisions and byte/time identities; do not bypass rejected hashes.
Completion requires exits/artifacts/official parity, not command submission.

Research goal remains active after an experiment fails. Diagnose and preregister
the next justified change; never reinterpret a small gain as50% or guarantee it.

## Outcome

Pending implementation/preflight; no actual fit or result asserted.

### Runtime evidence, not a performance outcome

Protocol review passed after clarifying the fixed-best1% bound. Local runner
tests11/11 and evaluator tests10/10 passed. The actual training host's synthetic
212-input/24-output spectral/native-RMSE smoke also passed, with eval-matrix
endpoint invariance and saved-prefix replay both exactly0. This is compatibility
evidence only. Source diff and feature lock preceded the first scientific fit.

Fresh seeds42/43/44 started2026-11-08 15:08:52/54/56UTC (November9KST) on golab2,
fourCPUthreads each,CUDA invisible, within the measured total6job/24thread limit
including three preserved other-campaign jobs. Source/protocol/control arrays
were staged before admission. Exact launch/resources are in runs/*/launch.json.

Independent first-head F1 checks on all three newly fitted900-round priors
reproduce the frozen parent's calibration predictions with max difference0;
query truth was not read. Receipts:
runs/roundselect_s{42,43,44}/F1_prior900_b0_control_replay.json. This verifies
one head only; complete three-head controls must still pass before selection.
No treatment period score, promotion or50% achievement is asserted.

Paired-analysis code is prepared and gated on all three completed runs, with
fresh direct prior900-array checks before official truth access. Primary168h/
2000draw uncertainty and all fixed-best/matched/auxiliary gates remain as above.
SOTA comparison uses a separate dated published snapshot; unavailable snapshots
are reported as unavailable rather than conflated with the frozen control.

### First completed-period evidence

All3F1 prior900 full-head endpoints exactly match the frozen parent controls
(difference0). The training-only first-minimum rule selected head counts
315/152/135(seed42),282/159/133(seed43),304/150/108(seed44). Each full model was
newly fitted for those counts; no prior model was sliced and passed off as a
new full fit. All three F1 predictions, selected-prior predictions and offsets
independently replay with max difference0. Recorded selected native-RMSE values
recompute within3e-8, and stopping/offset indices and144h gaps match exactly.
Receipts: runs/roundselect_s{42,43,44}/F1_model-replay.json.

An additional descriptive diagnostic of existing **fitting** subsets was added
while the scientific fits were running, before any treatment official score
was consumed. It changes no model, fitting criterion, gate or evaluation policy.
On the later336offset rows unused for tree-count selection, raw-prior RMSE is
88.05/88.75/89.39 for900trees and85.01/84.88/85.46 for selected prefixes, but
>=700bias worsens from−92.82/−92.38/−96.49 to−96.95/−98.44/−101.43. These rows
are used subsequently to fit the original offsets; these are pre-offset raw
fitting diagnostics, not independent official performance evidence. This
supports retaining the preregistered RMSE-versus-S uncertainty rather than
interpreting a selection-metric gain as forecast improvement. No criterion or
parameter was adjusted. All3runs proceeded to F2 under the fixed procedure.

Actual executed workspace source copies, the remote prelaunch strategy and
precise source byte/time receipts are preserved in code_snapshot/executed_workspace/.
Experiment/shared-core direct byte comparisons to local sources pass. Full
training-host distribution/environment snapshot is preserved separately with
its capture time; relevant numerical versions are also recorded at run start.

### Completion-validator correction during active fitting

The scheduler artifact validator reused `count` for selected trees and later
checked forecast shape against that number instead of official origin count.
The inner variable is renamed `selected_count`; no scientific fit or decision
rule changes. A synthetic completed-artifact regression failed8/passed2 before
and passes10/10 after. See scheduler-validation-fix.diff/json and
test_host_scheduler.py. The already-loaded remote scheduler remains untouched
while models fit; preserve any raw misclassification and revalidate completed
artifacts with the corrected validator after all model processes exit.

### F2 replay and F3 control progress

All three F2 completed selected-model replays pass with exact-zero parent900,
prior prefix, full raw, offset and final forecast differences. Selected counts
are42:[892,98,68],43:[867,99,74],44:[888,121,76]. All F3 first-head prior900
replays also match the parent exactly. Remaining F3 fits are live; no official
performance outcome has been inspected or selected. Fitting-only descriptive
RMSE/tail discrepancies are preserved in fitting-diagnostics-interim.md/json.
The terminal-only reconciliation controller has passed its actual-host guard
(rejected before any mutation while training remains live), plus lint and
compilation; it will be invoked after original scheduler and fit termination.

### Frozen postfit diagnosis scope, before any treatment performance reading

After all three complete successfully, compare matched parent/treatment final
and existing no-offset artifacts independently. This separates reported raw
full-model differences from the arithmetic effect of the unchanged offset
policy without fitting a correction or proposing an ensemble. It is descriptive,
not a new selectable candidate. Preserve all original forecasts.

Reuse the original residual audit fixed strata (lead, target/origin speed,
origin24h trend, official observation gap, original image age and ENLIL pending)
and all five fixed timing shifts on identical central leads. Report per-seed
errors and additive MSE/TIMSE energy changes with original official weights;
these are not additive regional S contributions or causal event effects.
Require completed_runs before opening official truth. Diagnostic code may be
prepared and synthetically tested during fitting, but scientific execution
waits for all three exits. No thresholds, features, counts or gates are changed.

### Completed scientific outcome (frozen)

All three fresh runs exited0; all54savedboosters independently replayed with
maxprediction/offsetdifference0. Original28features/212columns and actual values
were independently reconstructed and match. Official freshscores reproduce
allstoredmetrics exactly withoutfallback. NewS42/43/44 are
98.94406819690332/99.5352074794667/99.72224550591471; parent values remain
97.49244208059969/97.87370833197167/98.07451343718657.
MeanS97.8135546166→99.4005070608; meanpairedreduction−1.622213%,
168h/2000circularperiodbootstrap95CI[−3.019933%,+1.537568%].
Everyseedworse; fixedbest1%,matchedgain,auxiliary,periodand50%gatesfail.
No adoption or publication. CIincludes0; no statisticallyconfirmedharmclaim.
Goal evaluatorfreshscoreFAILexit1; researchgoalremainsactive.

Diagnosis (predeclared fixed strata) finds F2dominantdegradation, longlead and
low/highstateamplitudefailures. No-offsetS also worsens, so frozenoffsetalone
doesnotexplainfailure. Offsethelpsbotharms buthelpsnewmodelless. Do notretune
monitor/window/patienceagainsttheseofficialoutcomes. New mechanisticresearch
mustprecedeanyfurthercontrolledfreshtraining. See report.md/diagnosis.json.

Raw schedulerterminalclassification ispreserved in scheduler_initial_terminal;
correctedterminal-onlyvalidator recheckedallartifactssuccessfully withnofits
orforecastchanges. Firstdiagnosticlayoutfailure log/exit ispreserved in
failed_diagnostics/attempt01; corrected metadata-state receipt path passed
sixsynthetictests andactualdiagnosticexit0. NumPyimport2.4.6 isconfirmed;
2.2.6 isanotherVenv's listeddistribution,notusedtrainingmodule. No environment
orpackagechanges. Provenance/publicavailability/calibration/hash gapsremain
explicit in independentaudit, whichmustnotforceanoverallintegrityPASS.
