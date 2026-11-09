# S 50% reduction: audit-first preregistration

2026-11-08. The user explicitly confirmed that W means the current official S.
Research goal: relative S reduction of at least 50% against the current best
single model, independently of the comparison with SOTA. Achievement and
attainability must be judged from evidence, without a guaranteed outcome.

## Frozen references and scope

Current best single: `drop035_v650_s42`, official S=97.49244208059969;
target S<=48.746221040299844. Its configuration replications are seeds42/43/44,
not a prediction ensemble. Keep the adopted A+B1 incumbent separate. The live
board revision and reference replay are in `../../analysis/w50_audit_20261108/`.

Before any new fit, audit raw quality flags, time semantics, publication delay,
cutoff/target maturity, exact input matrices, scorer and prior failures.
Official F1/F2/F3 only; evaluation truth is read only after predictions freeze.
Existing running campaigns are preserved. No score-based tuning or publication.

## First decisive control: stale image-age repair

Audit found carried-forward missing SUVI frames keep an age below one hour,
although the actual frame can be days old. This is a concrete input defect.

A: frozen 28-source-feature, 3x24h XGBoost models from
`../drop035_v650_confirm_20261108/runs/drop035_v650_s{42,43,44}`.
First reproduce their existing forecasts from stored full boosters and offsets.
Require absolute max prediction difference <=1e-6 before any treatment scores.

B: replace only `img_age_h` with the age of the last actual frame: for raw
nonmissing frame rows infer `frame_time=origin-img_age_h`, forward carry that
time causally, then calculate `origin-frame_time`. No backward fill. A+B retains
every other input, all fitted trees, offsets, clipping and source feature names
and order. This is explicitly a **replacement/bugfix inference ablation**, not
an additive feature experiment or a retrained model.

Physical mechanism: distinguish stale remote-sensing information from genuinely
recent coronal observations. Required observation is raw image age/frame time.
Falsification: no material forecast change or no score improvement under this
fixed-model intervention. A negative result does not falsify a corrected
training pipeline: repaired ages extend beyond the original training support.

Successful audit requires exact A reproduction, unchanged non-age matrices,
finite complete predictions, exit0 and official scorer results without fallback.
Potential candidate requires mean S reduction>=1%, every seed improves,
paired168h bootstrap 95% improvement CI lower>0, mean MAE/RMSE worsening<=1%,
no period mean S worsening>2%. Even if passed, no automatic adoption/publication:
this inference diagnostic lacks corrected-training evidence. A 50% achievement
requires the same candidate's valid score<=48.746221040299844; report mean/SD/
variance/worst across seeds separately, never choose a new best seed.

Temporal uncertainty: paired circular 168h origin blocks stratified by official
period, 2000 replicates, RNG20261104, empirical TIMSE quantiles recomputed per
draw. Intervals condition on the fitted seeds and exposed benchmark; they do
not remove repeated-leaderboard selection bias.

## Execution and preservation

CPU inference only, CUDA_VISIBLE_DEVICES='', --device cpu --threads1; no fits.
One immutable single-architecture experiment.py and independent tmux per run,
registry/runtime/manifest, launch.json before execution, JSON train.log progress,
F1/F2/F3 predictions and metadata, train.exit and complete.json with
publish:not_required. Check sunrun running and recent output without editing TUI.
Keep frozen originals unchanged; fetch remote boosters into this study only.
Record source/input byte/time identities, commit, environment, feature lists,
cutoffs, seeds and official scores. User-requested hashes were blocked by an
automatic approval hook citing an older No Hashes rule; preserve that gap.

## Literature basis before the intervention

SWPC RTSW official README distinguishes preliminary RTSW from final science
data, status0 nominal/1-8 bad/9 absent, hourly lists update at :10:
https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/README
Rotter et al.2015 measures coronal-hole/HSS transport lag4.02+/-0.5days and
arrival-time uncertainty; it does not establish benefit of this repair:
https://arxiv.org/abs/1501.06697
NASA SIML-HSS documents coronal imagery plus26/27/28day recurrence and excludes
CME forecasts: https://ccmc.gsfc.nasa.gov/models/SIML-HSS~1/

No model family or additional experiment inventory is fixed in advance.
If repair is null/negative, diagnose before proposing corrected-training work.
