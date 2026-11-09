# Current leaderboard winner plus lag24/48/72: exploratory joint addition

Preregistered before fresh fitting, 2026-11-08 KST. User explicitly requests
adding all three listed features to the leaderboard first-place model, fresh
training, and leaderboard publication. This authorizes publishing all three
prespecified seeds even if the exploratory performance gate fails. It does not
authorize replacing the winner's existing features or picking a lucky seed.

## Frozen parent and single joint hypothesis

Live official store audit read all207results at revision
f756e1623af7f0cbd9fd9c0fc181902f542580bf. Best overall and best single model is
`e2000fs_calib_offset_roll_drop_t035_v650_s42`, S97.49244208059969. Exact record
and configuration are pinned in parent-lock.json. The corresponding scientific
configuration D has28source features (20constant/8varying),212expanded inputs;
matched controls are existing D seeds42/43/44, mean S97.81355461658598.

The only intervention is D union [`v_lag24`,`v_lag48`,`v_lag72`]. All are
origin-constant historical speed inputs, none already belongs to D. Their
source-column names and values come from the existing prepared input; no new
feature calculation or label reconstruction. Retain all28parent inputs and
their expanded212-column prefix; append3constants =>31source/215expanded.
Do not restore the two removed features, change the SUVI age input, add B1,
blend predictions, or import modifications from the concurrent age-repair run.

Hypothesis: jointly including one/two/three-day speed history adds temporal
information beyond D's current-speed and recurrence features. Correlated
features may be redundant; single-column gains need not sum and interactions
can reverse a gain. This is one joint feature-group addition, not independent
contribution attribution for each lag. Single additions remain independently
tracked by the existing exhaustive feature campaign; do not duplicate them here.

The selection evidence is exploratory: original single-column seed42 S values
were97.00551251655173 (lag24),97.09593267259113 (lag48),97.38345083613505
(lag72), versus97.49244208059969. Seeds43/44 had not confirmed these gains when
the user selected the combination. Repeated F1/F2/F3 exposure means this fresh
fit is not independent blind generalization evidence.

## Implementation and physical context

Keep XGBoost2.1.4 CPU hist, depth3, learning rate.03,900rounds, subsample.8,
colsample.7,min_child_weight20,lambda5,multi_output_tree, three24h blocks,
tail4at550, full spectral coefficient.1/scale50, training-only horizon offset
alpha.25 over last336calibration rows, component and final clip[1,3000]. Seeds
42/43/44 are all fitted afresh; no model or prediction reuse, no best-seed fit.

Previous evidence/reference sources remain applicable:
- `../feature_addition_all_20261108/strategy.md` and inventory.json: complete
  training-only candidate audit, original407ledger, lag input classification,
  timestamp causality and prior results. All three candidates passed the audit.
- Temmer et al.(2018), https://www.swsc-journal.org/articles/swsc/full_html/2018/01/swsc170088/swsc170088.html :
  persistence and coronal-hole information are useful forecasting mechanisms,
  but added information did not necessarily improve speed values over its
  persistence control. This motivates a measured complementarity test and
  does not predict a numerical gain here.
- https://xgboost.readthedocs.io/en/release_2.1.0/tutorials/multioutput.html :
  retain the existing supported vector-leaf implementation; no new dependencies.

## Evaluation, reporting and publication

Only official F1/F2/F3, cutoffs2025-12-29/2026-03-29/2026-06-29,
origin+144h maturity, unchanged purged training-only calibration split and
prepared-label policy. Targets enter scoring after fitting, never feature or
coefficient fit. Freeze the exact parent observation revision and official scorer.

Report matched-seed S deltas, mean/sampleSD/worst, MAE/RMSE/TRMSE/TIMSE and each
period's metrics. Separate comparison with parent three-seed mean from the
external best single score. A material candidate requires mean S gain>=1%, all
three seeds improve, paired168h block-bootstrap2000 CIupper<0, meanMAE/RMSE
deterioration<=1%, each period meanS deterioration<=2%. A single predeclared
joint hypothesis requires no additional within-campaign multiplicity correction;
selection from the broader campaign is explicitly disclosed. SOTA2% target is
95.54259323898769, separately reported; call official S by its actual name.

After normal exits, complete forecasts/metadata, exact model replay, source
union/order verification and official scorer parity, publish all3individual
seed models with pinned architecture and self-contained replay sources. No
ensemble is requested or created. Preserve the exploratory annotation even if
the measured score improves. Verify storedTotal/F1/F2/F3scores and visibleSpace
rows/architecture links; submission alone is not completion.

## Resources, lifecycle and completion

Preserve running experiments on both hosts. GBM CPU only,CUDA_VISIBLE_DEVICES='',
--device cpu,--threads4, global host<=6CPU workloads/24declared threads,
MemAvailable>=11GiB per admission (estimated5GiB plus6GiB reserve), real CPUload
and disk checked, shared admission lock. Local6jobs are currently occupied;
golab2 currently has3age-repair jobs and room for up to3new jobs if measurements
still agree at launch. Its repeated WSL lifecycle failure means the actual
remote scheduler must remain foregrounded on a durable SSH/WSL connection,
following the validated age-repair recovery pattern, never a placeholder sleep.

Independent run tmux/log/output folders, standard sunrun assignment/registry,
runtime/manifest, launch/config/source receipts, JSONperiod/block/unit/units,
Fperiodpredictions/metadata,train.exit and validated completion receipts.
Record Git revision/dirty state, versions, byte identities, features/ranges and
parameters; no file hashes. Preserve failed-attempt logs; distinguish resource,
implementation and protocol failures from a valid negative hypothesis result.

Stop only after all3fresh fits and verified publications, or an evidenced
external blocker. An active scheduler or queued registration is not a completed
fit or published score. Existing all-feature campaign remains report-only.
