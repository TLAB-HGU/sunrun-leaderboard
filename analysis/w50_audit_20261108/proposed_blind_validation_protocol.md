# Proposed independent future/blind verification

Status: PROPOSAL ONLY. Not activated; no future observations, forecasts or external
blind labels have been collected under this document. It does not change the
official F1/F2/F3 policy, create a new experiment ID for a validation-policy change,
alter a fitted candidate, or replace the existing50% goal/evaluator.

## Questions and distinct estimands

The official goal remains S<=48.746221040299844 against frozen best single
S=97.49244208059969 on the unchanged official benchmark, with its existing
multi-seed, auxiliary-metric, uncertainty and integrity requirements. Repeated
exposure to those periods is exploratory; temporal bootstrap does not remove
selection bias. This proposal would measure generalization and issue-time
causality on independently unexposed data. Its scores cannot substitute for the
official score or be compared numerically to a historic SOTA score from a
different observation window.

Primary future reporting estimand: the mean of the three matched per-seed relative
S reductions, with each arm evaluated on the identical origins/lead-wise truth
mask. Report the ratio of mean S separately. Report any50% future reduction
separately; neither a small positive future effect nor completion of collection
constitutes official-goal achievement. No new easier goal/PASS threshold is
introduced here.

## Freeze before activation

After the current official experiment's decision and before any future labels
are exposed, name exactly one candidate and the frozen A comparator, artifact
paths, source content commits/bundles, all fitted boosters/offsets, seeds42/43/44,
feature names/order, fitting populations/cutoffs and software environments. No
future-best-seed selection or forecast ensemble is permitted. Retain the
documented per-seed Python versions, CPU4/OMP4/BLAS1 and CUDA-invisible GBM policy.

The narrow default proposal tests the already fitted F3 A and candidate models
without retraining or updating calibration. Both retain the common official
F3 cutoff2026-06-29 and144h maturity. This estimates frozen-model performance
after that cutoff; it is not evidence about an updated operational retraining
policy. A future rolling-refit study would need its own complete predeclared
cutoffs and identical refit schedule in both arms, and is not activated here.

Create a separate inference-only adapter that uses the identical causal feature
construction and preserved fitted state. Do not modify the official scorer,
current experiment runner, original data or predictions. Prove the adapter's
feature/prediction parity against frozen existing examples before activation,
without using those examples to tune any parameter. If the frozen feature
construction needs information unavailable at issue time, record that failure;
do not silently repair timing, replace features or use retrospectively revised
inputs inside this validation study.

## Proposed prospective window and observation contract

Before collection, record an actual UTC start T0, beginning at00:00UTC after
activation. The primary origin interval is [T0,T0+90consecutive calendar days),
with360scheduled6h origins and all72leads per origin. Retain complete target
windows for the final origins: the last target valid time is T0+92days18h.
Collection and truth release extend through that target and the fixed
publication/revision delay; do not truncate terminal horizons. Ninety days
spans roughly three solar rotations; it is a physical coverage choice, not a
demonstrated power calculation. Do not extend or shorten the primary origin
window based on storms, candidate scores or significance. Insufficient event
coverage yields an inconclusive estimate; a later extension is a separately
frozen study.

Plan every6h UTC origin and all1–72h leads, matching the official lead/origin
convention. Before activation, resolve the producer's timestamp/averaging window,
physical location and speed definition, quality flags, label product/revision
policy, acceptable publication delay, and the distinction between observation
time and first availability. Primary truth should preserve the official RTSW
speed target/quality convention rather than silently substituting a different
calibrated product. Retain raw observations, quality flags and official correction
materials. A secondary calibrated or independent-instrument endpoint must be
predeclared and reported separately, with physical/time alignment established
from documentation/geometry rather than by fitting a lag or scale to test errors.

If these timing or product prerequisites remain unresolved, activation readiness
is incomplete. A reconstructed monthly archive does not alone certify what was
publicly available at an earlier origin.

## Issue-time evidence and delayed truth release

For every input packet preserve producer observation time, producer issue/public
time where available, local first-received time, revision/product ID, source URL,
raw quality flags, transformation version and actual forecast issue time. The
same packet feeds A and the candidate. Before activation, freeze each issue
deadline relative to its scheduled origin and state how actual issue latency
relates to origin-based lead time; record late forecasts as deadline failures.
Input availability must precede that deadline and satisfy the frozen causal
feature recipe; no later backfill, revised image, ENLIL issue or corrected
measurement may rewrite a past input packet. Missing-data handling follows the
frozen causal recipe and is logged. An issue-time feature outage is not a reason
to remove a difficult origin after seeing its truth.

Preserve complete forecast packets for every planned origin, all72 leads and all
three individual seeds, plus launch/process/exit, device/thread, source/environment,
cutoff and input receipts. Each forecast packet must be sealed before access to
its verifying targets. The frozen inference lane may ingest observations as
they become causally available for later origins; earlier forecast packets stay
immutable. A custodian or technically separated collection/scoring lane may
accumulate verifying truth internally, but model developers receive no primary
scoring labels, residuals, ranks or scores until every planned packet is frozen
and the entire cohort's last target plus the fixed publication/revision delay
is complete. Record access separation and any exposure deviations. Public raw
observations prevent claiming absolute raw-label secrecy unless actual access
controls support that claim. Do not send material to any outside custodian
without a separately arranged authorized workflow; this document sends nothing.

Keep the existing No Hashes restriction: do not generate observational
cryptographic digests. Preserve version IDs, raw files, size/time receipts and
explicit stream byte-copy comparisons instead. Source Git commits/bundles remain
allowed. This evidence has stated limits and must not be described as proof of
historical authenticity or a cryptographically sealed observation archive.

## Missingness, scoring and uncertainty

Require forecast coverage for every planned origin/lead in each arm. Preserve
missed deadlines, failed processes and absent/invalid forecasts; do not silently
drop them or fit/impute replacement forecasts from truth. Incomplete coverage
prevents a complete primary skill claim; available-case summaries, if shown, are
explicitly descriptive and selection-limited. Invalid/missing truth is excluded
only by the same predeclared product/quality mask in both arms. Report scheduled,
issued, valid, scored and missing counts, including reasons and time clusters.

Use the immutable official metric mathematics, lead-wise
S=mean_h((MSE_h*TIMSE_h)^(1/4)), with the empirical TIMSE quantiles/counts recomputed
on the declared future population. Since the official competition wrapper has
fixed F1/F2/F3 dates, use a separate date adapter with synthetic and frozen-example
math parity tests; never rename the future window F1/F2/F3 or overwrite official
results. Require zero undisclosed prediction replacements.

Report each individual seed's S/RMSE/MAE/TIMSE/TRMSE and their across-seed
mean/sample variance/SD/worst, plus coverage, lead and fixed
physical-event/state/missingness strata. Truth-defined strata are
reporting only and never become features, coefficients or hyperparameter choices.
Retain lead×fixed-speed MSE/TIMSE energy budgets if error attribution is requested;
pooled marginal energies cannot reconstruct nonlinear official mean-lead S.

Predeclare the existing2000replicate circular168h origin-block bootstrap,
RNG20261104, common draws across paired arms/seeds and recomputed TIMSE quantiles.
The future window is one prospectively defined population, not new F1/F2/F3
selection folds. Report that168h blocks may miss longer solar-rotation dependence,
three fitted seeds do not measure all training uncertainty, and sparse severe
events may leave wide intervals. No adaptive block-length/significance search,
interim stopping, seed cherry-picking or retrospective coefficient fitting.

## Alternative external blind route

A genuinely independent custodian could instead supply a fixed unexposed cohort
under the same observation, availability, feature, cutoff and scoring contract,
with labels hidden until complete forecast submission. Establish actual lack of
prior developer exposure and exclude targets/cases used in fitting, calibration,
feature/coefficient/hyperparameter selection or earlier exposed official/EDA
evaluation. An old public period already inspected in this project is not blind.
Declare the cohort's origin/target interval and verify non-overlap with those
uses and temporal eligibility after the frozenF3 cutoff before release. Legitimate
prior observations used only as causal input context are distinguished from
using held-out targets for fitting or selection. Predeclare case inclusion,
deadline failures, product revisions, physical/time alignment, uncertainty method
and release policy before any labels are seen. Do not use this alternative to
choose whichever test yields the better score. This is an alternative proposal,
not an additional active evaluation.

## Decisions and reproducibility

An independent completion review must reconcile actual forecasts, input timing,
raw flags/corrections, all process exits, source/environment/cutoff receipts,
scorer math, complete coverage, uncertainty and any protocol deviations. Preserve
invalid runs and unsuccessful outcomes. Keep metric evidence, protocol validity,
causal validity, adoption and official-goal achievement separate. Report remaining
uncertainty without interpreting outcomes toward50%.

Existing official F1/F2/F3 evaluation continues unchanged while this proposal
remains inactive. No automatic model, training-cutoff, metric, leaderboard
submission, external communication or policy change follows from this document.
