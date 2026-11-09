# Eight predeclared additive feature axes after the three-lag combination

Frozen2026-11-09 before training. User requests continuing priorities1–8 in
tmux. This campaign performs three fresh seeds42/43/44 per independent axis,
preserving every adopted-parent feature, objective and calibration setting.
It is report-only; no new leaderboard publication is inferred from this request.

## Parent policy fixed before outcomes

Wait for the complete matched-seed report of lag24_48_72_confirm_20261109.
Use D+lag24/48/72 only if its predeclared full improvement gate passes. Otherwise
use the unchanged28-source D: e2000fs_calib_offset_roll_drop_t035_v650.
Do not choose by the best seed or silently adopt a subthreshold feature group.
Freeze parent-lock.json and baseline_runs before any new axis training.
No axis feeds into another axis; no score-driven feature unions or extra arms.

## Fixed axes and readiness

1. Daily shape: add v24−v48, v48−v72, v0−2v24+v48 as three constants.
   v0−v24 is excluded because speed_d24 already exists in the raw inventory.
2. Recurrence agreement: add leadwise populationSD(rec26,rec27,rec28), strict
   missing propagation. Existing recurrence inputs remain unchanged.
3. Current/recurrence disagreement: add rec_mean−v0 and rec_mean−v24 leadwise.
4. Direct morphology: add exactly the existing eight causal72h shape statistics
   slope_norm/delta_norm/curv/peak_pos/trough_pos/frac_inc/ac1/r2_lin, computed
   in12six-hour bins, unchanged definitions from the previous GMM experiment.
   This tests direct statistics, not another partition/expert/posterior model.
5. Add four reprojected SUVI strips, without changing original strip columns.
   Carry a whole last available raw image/frame, recompute its age and fixed
   longitude overlap for each horizon using the existing0.55deg/hour projection.
   Speeds/statistics350logmed,450logmed,550dark,750logmed match existing strips.
6. ACE minute variability: populationSD,linear-IQR,range of finite positive
   status0speed in (origin−60min,origin−4min]. Existing hourly inputs retained.
7. ACE minute trend: linear slope(km/s/min), recent15min mean minus prior45min
   mean in the same available window. At least2valid samples for slope and
   one valid sample per subwindow for difference; otherwiseNaN, no backfill.
8. Issued CME kinematics: speed,half-angle,longitude,latitude,issue-age from a
   unique valid single-event issued summary in the preceding72h. Latest event
   pages, prime flags, model completion and observed arrivals are forbidden.

Axes1–4 use existing input channels; their transformations add no observations.
Axis5 raw32,987frames reconstruct original strips within7.58e-7 and identical
NaN masks, then fixes coherent frame/lead projection additively. Its raw-origin
timestamp causality is verified; actual historical publication receipts are not.
Axes6/7 have all1,458required daily files; cache presence is not valid-sample
coverage. The4min release delay is a frozen research assumption from the ACE
study, not proven individual historical receipt time. Record finite coverage.

Axis8 has an independent prerequisite gate. Missing2026Jan–Sep archive months
were retrieved into this campaign only:9months,302messages,244summary records,
zero retrieval failures. This closes a data-presence gap but does not prove
immutable as-issued body history. Keep all3axis8 runs blocked_source until
that evidence is established. Never silently train latest retrospective values
or call a blocked/unexecuted arm a negative result. Source retrieval is not fit.

## Scientific implementation invariants

The fixed parent XGBoost2.1.4 CPU hist/vector-leaf recipe is unchanged: depth3,
lr.03,900rounds,subsample.8,colsample.7,min_child_weight20,lambda5, three24h
heads,tail4at550,full spectral.1/scale50,horizon/recent336offsetalpha.25,
component/final clipping[1,3000]. Model and prediction reuse is forbidden for
treatments. Parent forecasts are explicitly historical matched controls.

Create feature caches without reading y; append columns to immutable prepared
input rather than replacing anything. Preserve original row groups and(origin,h)
keys. Reread every output row group and compare all original features, labels,
metadata, missing masks and schema exactly. No labels are reconstructed and no
new derived column can collide with an original name. Generated inputs are
separate per axis; source/expanded lists and source identities are recorded.

Official F1/F2/F3 only, cutoffs2025-12-29/2026-03-29/2026-06-29,
origin+144h label maturity, unchanged training-only calibration purge. No
experiment-selection folds or benchmark-truth fitting. Report officialS,
RMSE/TRMSE/TIMSE/MAE,matched-seed deltas,mean/sampleSD/worst and period results.
Paired period-stratified168h block bootstrap2000,RNG20261109; same draws across
seeds/axes. BHfamily includes all8hypotheses with p1for unfinished/blocked arms.
No final adoption before all8complete. Candidate gate: meanS>=1%reduction,
all3seeds improve, CIupper<0,BHq<.05,meanMAE/RMSEdeterioration<=1%, eachperiod
meanSdeterioration<=2%. Repeated official-period selection remains exploratory.

## Prior evidence and literature

The original407raw additions are a different campaign. Existing speed_d24,
generic freshness/source-travel features, SIR, GMMposterior and quantile-width
must not be advertised as globally untried. SUVIage replacement fresh training
completed with meanS97.813555→97.902535; it is not adopted. Axis5 changes strip
geometry rather than repeating age-only replacement. Earlier MoE fragmentation
and two-stage failures do not establish direct morphology feature effects.

References and limits:
- Bussy-Virat & Ridley2014, https://doi.org/10.1002/2014SW001051 : current
  speed/slope and rotation recurrence motivate conditional temporal features;
  no numerical gain transfers to this dataset/metric.
- https://ccmc.gsfc.nasa.gov/models/SIML-HSS~1/ :26/27/28day recurrence and
  coronal-hole spatial inputs; no real-time availability or CME-skill claim.
- NOAA SUVI guide https://www.ncei.noaa.gov/sites/default/files/2024-12/SUVI%20data%20guide%20for%20SDO_AIA%20users.pdf :
  image acquisition/quality context, not evidence that reprojection improvesS.
- ACE RTSW README https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/README :
  status0and update convention; historical completed files are not receipt logs.
- DONKI notification/version guide https://ccmc.gsfc.nasa.gov/DONKI/guide/notifications/ :
  issued-message semantics. Local audit cme_information_followup.md documents
  completion/publication differences and retrospective enrichment hazards.

## Execution and completion

Local controller prepares label-free caches under one-thread admission, waits
for parent selection, stages ready axes on golab2, and runs the actual remote
scheduler foregrounded overSSH/WSL. No placeholder reserves resources.
GBM CUDA_VISIBLE_DEVICES='',CPU4threads; global host<=6workloads/24threads,
MemAvailable>=11GiB,6GiBreserve,15GiBdiskheadroom, shared admission lock and
other-campaign load counted. Preserve existing experiments. Each run has its
own tmux/log/output, launch identity, source/config snapshots, JSONprogress,
Fperiod predictions/metadata,normalexit,official parity and saved-model replay.
The standard sunrun layout is used without monitor-specific modifications.

Source/augmentation/scientific failures are not hypothesis failures. Preserve
failed attempts and receipts; resume only frozen matching artifacts. Record Git
revision/dirty state, versions, byte/mtime identities and input ranges. No file
hashes. Every completed fit says publish:not_required.21completed ready fits is
not all24completed: ready-axes-complete.json must retain blocked axis8 and the
provisional report. End-of-phase status must never claim an unexecuted axis
was tested or that the full eight-axis study is complete.
