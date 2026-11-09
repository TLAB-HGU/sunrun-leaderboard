# Fresh training: causal SUVI age repair

Preregistered 2026-11-08 before any fresh fit. User requires new model training
and an active research campaign until evidence establishes the 50% goal.
W means official S. This is a replacement/bugfix ablation, not an additive claim.

## Fixed comparison and single change

A is the current best single configuration from
`../drop035_v650_confirm_20261108`, seed-matched 42/43/44. Best single official
S=97.49244208059969; immutable 50% target=48.746221040299844. Mean reference
S=97.8135546166. Report current external SOTA separately with snapshot date;
the previously audited current SOTA equals this best single, not an older row.

B replaces only `img_age_h` with origin minus the last real observed SUVI frame
time, computed causally from raw origin minus raw age and forward-carried frame
time. Apply consistently to training, training-only calibration and query.
Preserve all other matrix values, column names/order and targets. Keep exactly
28 source features and 212 expanded columns per 24h block. Do not repair
img_stale, reproject strip features, alter targets, or add features in this test.
Trees and training-only offsets must be freshly fitted, with no model reuse.

Keep original depth3, learning_rate .03, rounds900, subsample .8, colsample .7,
min_child_weight20, lambda5, tail weight4 at550, spectral coefficient .1,
scale50, three24h multioutput heads, horizon offset alpha .25 over last336
training-calibration rows and clipping[1,3000]. Same seeds42/43/44.

## Mechanism, evidence and falsification

Confirmed: forward filling raw missing age makes a days-old image appear less
than one hour old. See `../../analysis/w50_audit_20261108/data_audit.md`.
Previous frozen-tree intervention had mean S gain .0186%, CI including zero;
it did not test learning corrected ages outside the previous training support.
Inference: correct ages may let trees distinguish stale image information.
Necessary observation is actual frame observation time, available here through
raw nonmissing age; actual publication receipts remain unverified. A null fresh
fit with invariant checks passing rejects a material benefit for this frozen
learning configuration, not the possibility of other missing observations.

References: SWPC RTSW quality/update contract
https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/README ;
Rotter2015 https://arxiv.org/abs/1501.06697 ; SIML-HSS
https://ccmc.gsfc.nasa.gov/models/SIML-HSS~1/ ; official XGBoost multioutput
https://xgboost.readthedocs.io/en/release_2.1.0/tutorials/multioutput.html .
These establish data/physical/implementation context, not a predicted score gain.
Detailed checked literature: `../../analysis/w50_audit_20261108/literature.md`.

## Evaluation and gates

Only official F1/F2/F3; unchanged cutoffs2025-12-29,2026-03-29,2026-06-29 and
144h training-label maturity. No new selection folds. Training-only holdout
for existing offset fit remains. Freeze all predictions before reading official
truth for scoring. Official scorer `leaderboard/scorer/competition.py`, same
frozen observations as A. No evaluation-target coefficient/parameter fitting.

Report official period/total S, MAE, RMSE, TRMSE, TIMSE for each seed, sample
mean/variance/SD/worst, and paired relative reduction. Reuse preregistered
period-stratified circular168h origin bootstrap2000, RNG20261104, recomputing
empirical TIMSE quantiles. Candidate gate: mean relative gain>=1%, all seeds
improve, 95% lower gain CI>0, mean MAE/RMSE deterioration<=1%, each period mean
S deterioration<=2%. No seed cherry-picking. No automatic external publication.
50% is a separate stronger gate: all three single-model scores<=48.746221040299844
and verified fitting, artifacts, auxiliary metrics and uncertainty evidence.
Failure to reach50% keeps the research goal active, never redefines the target.

Repeated F1/F2/F3 exposure and original baseline feature selection cause
selection bias; these intervals do not certify unseen future generalization.
Unexposed future/external blind validation remains a separate proposed protocol;
do not silently change existing official evaluation policy.

## Execution and stopping

CPU GBM only, CUDA_VISIBLE_DEVICES='', four threads. Local currently6jobs/24threads:
no admission. golab2 currently0jobs and~30GiB available: admit at most two of
these runs concurrently, then third; total host<=6jobs/24threads and reserve6GiB,
require11GiB MemAvailable per admission. Recheck actual host workload each time.
Independent tmux/session/log per run, standard launch and registry/assignment,
runtime/scheduler-state/manifest, JSON progress, full boosters and period
predictions/metadata, train.exit and complete.json. Confirm sunrun running and
completed displays. Source diff and actual feature receipts precede training.
Preserve errors and existing experiments. No placeholder resource reservations.

Preserve commit/environment/seed/data identities/features/cutoffs/commands.
Previous hash request was automatically rejected under an older No Hashes rule;
do not bypass it; disclose byte/time identities and unresolved digest gap.

After results, diagnose hypothesis vs implementation vs protocol/resource
failure before choosing another experiment. Do not preselect a model roster or
expand scale without a physical/information diagnosis. Continue campaign through
the active Codex goal and durable performance-goal ledger until validated target,
user cancellation, or an actual externally blocked condition. Training completion
alone does not complete the research objective. Do not guarantee attainability.

## Outcomes

Pending new training. No new learning result is asserted.

### Execution incident and recovery

First detached remote scheduler and seeds42/43 ran at13:11:42UTC, then vanished
after13:12:27 with no train.exit. This is a resource/lifecycle failure, not a
hypothesis result. Partial outputs and logs are preserved under
failed_attempts/remote_session_01 on both hosts; no partial models reused.
Remote pytest was unavailable during a prior preflight; local8tests and actual
real-data smoke passed, and the remote config/raw-input preflight passed.

At13:14:47UTC clean fresh fits were restarted under a foreground Windows
OpenSSH→WSL connection running the actual host scheduler. The corresponding
local tmux is srw50-remote-scheduler-controller-20261108. This is a scheduling
service doing real admission/completion checks, not a sleep reservation.
Investigate disappearance independently; do not claim its root cause proven.
The local registry is empty to avoid presenting remote mirrors as interrupted
local jobs; remote assignment/scheduler-state is the authoritative TUI layout.

### First completed-period verification

Seeds42/43 both completed F1 and entered F2 training. Separate replay used an
independent latest-real-frame lookup (searchsorted), not the training repair
function, and never loaded query targets. Each saved period has six900-round
boosters; all2,089query origins reproduce stored forecasts with max difference0.
Source feature names/order28 and expanded columns212 match the frozen parent.
Receipts: runs/age_retrain_s{42,43}/F1_model-replay.json, generated on the training
host by verify_model_replay.py. This verifies completed-period artifacts only;
all-period exits, official scores, multi-seed CI and final audit remain pending.

Parallel read-only data work recovered41months of training-era official issued
CME notifications:1,174messages,882unambiguous single-event kinematic summaries.
See ../../analysis/w50_audit_20261108/notification_archive_probe.md. No new
feature/model was adopted and this experiment's inputs were not changed.

### Resource-only amendment before seed44 admission

2026-11-08, before seed44 starts and before any treatment official score is
available: permit the three already-registered seeds to run concurrently on
golab2 instead of limiting this cohort to two. Measured host: two4-thread fits,
28.07GiB MemAvailable, load7.01 on24logical CPUs. Keep the host limits6jobs/
24threads, four threads per fit, require11GiB before admission and reserve6GiB;
three conservative5GiB job budgets plus6GiB reserve fit the31GiB host.
No additional hypothesis, fit, seed, feature, hyperparameter, cutoff or scoring
change. This reduces waiting for the already-required seed replication.

The running scheduler has its old two-admission limit loaded. An exact-owned
one-time admission controller may briefly stop ONLY that scheduler while
checking ownership/resources and launching seed44, and must resume it in a
finally block. Existing fits continue. Afterwards the original scheduler tracks
all three existing launch receipts; it does not create additional experiments.
Preserve the admission command, resource snapshot and amendment in launch.json.

### Continuing verification

Seed44 fresh fit started13:42:45UTC after the recorded resource-only amendment.
Actual live processes for all three report deviceCPU,threads4,CUDA invisible.
Seeds42/43 completed F2 and entered F3; F2 full forecasts, prior-calibration
forecasts and training-only offsets independently replay with differences0,
with2,161calibration rows and unchanged maturity/purge. F1/F2 receipts do not
constitute all-period completion or an official performance improvement.

Further diagnostic evidence is summarized in age_repair_mechanism.md. The
training curve is descending but cannot establish optimization optimality.
Restricted residual-energy calculations demonstrate that isolated correction
of the prior Jan19event or all>=700targets, keeping other errors fixed, is
insufficient for50%; these are not prediction outputs or attainable bounds.
Official calibrated-observation coverage remains unresolved: ACE science L2
event coverage is absent in checked public archives; DSCOVR operational L2
outputs carry proton-quality warnings and cannot settle ACE correctness.
No data, loss, features, rounds or evaluation policy changed in response.

### Final completed outcome

All3fresh fits exited0, all54boosters saved, all9period full/prior/offset replays
exact(max difference0). Independent review reproduced actual matrices, source
snapshots, scores and sampled bootstrap draws. Analysis exited0; owned scheduler
and collector also exited0. Full report: report.md; independent review:
verification-independent.md/json. No candidate was adopted or published.

|Seed|Fixed-parent S|Fresh repaired-age S|
|---:|---:|---:|
|42|97.4924420806|97.8250391694|
|43|97.8737083320|97.9226666911|
|44|98.0745134372|97.9598999339|

Control mean97.8135546166; treatment97.9025352648, samplevariance.0048508122,
SD.0696477723,worst97.9598999339. Mean paired relative reduction−.0914366%,
95%168h block bootstrap2000 CI[−.3279122%,+.0558201%]. CI includes0: statistically
significant harm is not established, and the registered1% improvement is not
supported. All-seeds/mean/CI gates fail;50%target48.7462210403 also fails.

The preregistered matched-mean1% gate is not identical to AGENTS's existing-best
single1% submission rule. Later protocol review made the stronger fixed-best
bound96.5175176598 explicit for future experiments. This completed experiment
fails both; no retrospective rule change creates its failure or any promotion.

Classification: performance hypothesis not supported under this configuration;
implementation/score failure not supported. Repairing age provides no direct
new input at the diagnosed Jan18–21origins; fitting-only transfer diagnostics
also reveal temporal shift/overfit. These are explanations with limits, not a
claim that all correct-age representations or future observations are useless.
The failed component is not accumulated. Next preregistered investigation is
../w50_roundselect, which changes only the train-only tree-count rule.
All benchmark selection/provenance/public-availability/calibration gaps remain.
