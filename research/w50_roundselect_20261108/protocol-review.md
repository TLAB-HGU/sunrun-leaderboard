# Independent protocol review before fitting

2026-11-08. Scope: scientific design and executable preregistration. No model
fitting, query-target access, source/data/model/runtime mutation, hashing or
new experiment execution. Only this review artifact is authored by this reviewer.

**OKAY — fixed-best submission-gate amendment verified in strategy.md.**
No unresolved scientific-design blocker was found. This approves the protocol,
not an uninspected implementation or an unperformed preflight.

## Gate issue found and resolved

The initial strategy required at least 1% mean matched-seed improvement, positive
paired uncertainty, and every seed improving. That does not imply the root
AGENTS requirement of at least 1% improvement over the existing best single
model. For illustration, 0.99 times the parent mean is 96.83541907042012,
whereas 0.99 times the fixed best single is **96.51751765979368**. The actual
matched gate averages relative gains, but this distinction still holds.

The leader explicitly accepted the following addition before fitting:

> Retain every matched-seed, bootstrap and auxiliary gate. Also require all
> three treatment single-model total S values to be at most
> 0.99 * 97.49244208059969 = 96.51751765979368 before promotion/submission.
> Do not choose a passing best seed. Keep the 50% goal gate separate.

The amended strategy.md now contains this rule; the eventual gate implementation
must enforce it. It resolves the definite requirements gap and adds no training
hyperparameter.
The previous age experiment already failed its original matched gate and this
fixed-best bound, so no historical verdict changes. Record the earlier gate
interpretation gap rather than claiming that the stricter gate was historically
preregistered. No age-treatment promotion/publication followed its failed gate.

## Evidence and representative execution paths

1. **A versus A+B is coherent.** Parent baseline.py:56-65 fixes the three-month
   prior split and 144-hour purge. Parent core.py:109-133 trains each prior/full
   head for 900 rounds with the same custom objective and data-specific
   base_score. core.py:144-162 takes the last 336 clipped prior residual rows
   and multiplies their mean by 0.25. Changing the selected round count for
   both prior inference and full refit is one declared estimator-policy change.
   The resulting offset values necessarily change; the offset formula/window
   do not. This does not isolate the separate contributions of prior-prefix
   calibration and full-fit complexity, and results must not claim that it does.
   Metadata confirms 28 source/212 expanded features, the two established
   feature exclusions, no B1, 900 rounds and the stated cutoffs. Failed age
   repair is correctly excluded. Recompute base_score and objective weights
   on each unchanged prior/full training subset just as the parent does.

2. **Stopping and offset chronology can be implemented without guessing.**
   Applying the declared inclusive maturity condition to the parent's saved
   seed-42 metadata yields these hourly-grid expectations:

   | Period | Stopping origins, UTC | Count | Offset origins, UTC |
   | --- | --- | ---: | --- |
   | F1 | 2025-09-23 00:00 through 2025-12-03 01:00 | 1706 | 2025-12-09 01:00 through 2025-12-23 00:00 |
   | F2 | 2025-12-23 00:00 through 2026-03-03 01:00 | 1682 | 2026-03-09 01:00 through 2026-03-23 00:00 |
   | F3 | 2026-03-23 00:00 through 2026-06-03 01:00 | 1730 | 2026-06-09 01:00 through 2026-06-23 00:00 |

   These counts were inferred from existing metadata, not revalidated by loading
   training arrays; actual-grid assertions remain required. Each offset block
   stays 336 rows. Stopping labels mature before its first offset origin;
   offset labels are not used to choose counts. The final model still fits
   stopping and offset rows as part of the original full training set. Thus
   “exclusively” means reserved from round selection, not excluded from full
   fitting. Later official training includes earlier periods under the existing
   policy, so there is no new blind-period claim or independent internal score.

3. **Prefix selection, control and replay have distinct roles.** Record all 900
   finite native unweighted RMSE values per prior head; first argmin plus one
   gives the exclusive upper bound for iteration_range=(0, selected_rounds).
   Preserve the prior 900-round model, freshly fit the full model for the
   selected count, and save explicit counts. Do not continue training the prior
   booster or slice vector trees implicitly. The fresh 900-round endpoint must
   match the parent's stored raw prior calibration predictions within 1e-6,
   after verifying matching origins/shapes. A failure is a control defect, not
   evidence against the hypothesis. The planned synthetic eval/no-eval equality
   check tests whether monitoring changes the training path. Independent saved
   model, selected-prefix, offset and final-forecast replay tests serialization
   and inference, rather than merely checking that files exist.

## Acceptable uncertainties, not blockers

- **Mechanistic evidence is limited appropriately.** age_repair_mechanism.md
  reports lower prior fitting RMSE but higher calibration RMSE at 900 versus
  300, together with a large change in high-speed target frequency. This
  supports investigating complexity/shift, not proving overfit alone. Full-fit
  training includes newer calibration data and can need more rounds; count
  transfer without rescaling is explicitly the hypothesis. Older-parent
  routing failures in state_failure_history.md do not test it.
- **RMSE is not official S.** competition.py computes leadwise geometric means
  of RMSE and tail RMSE, then averages leads on six-hour origins with unfilled
  observations. The stopping RMSE pools a 24-output head on original fitting
  targets. Their population, weighting and objective differ. The strategy
  freezes this choice and retains tail/spectral training, so this is a meaningful
  regularization test, not a metric-parity claim. More tail attenuation is a
  credible failure mode; report TRMSE/TIMSE alongside the frozen score gates.
- Selecting among 900 correlated prefixes can overfit the stopping interval.
  Separation from the last-336 offset labels helps role separation, but does
  not establish independent future performance or remove regime shift.
- All-900 selection is a null policy intervention; shorter counts failing the
  gates are a negative result for this policy, not evidence against every
  early-stopping design. A control mismatch, nonfinite curve, invalid grid or
  replay failure is implementation/protocol failure. No outcome licenses metric,
  window, patience or maximum-round retuning against these exposed periods.
- The fixed best is an observed single-model comparator, not a newly proved
  population optimum. Its earlier campaign itself reported selection exposure
  and an interval including zero. This does not invalidate paired comparison,
  but the new experiment cannot erase historical selection or provenance gaps.

The literature evidence and sources.json were read as bounded research receipts;
this review does not independently re-fetch external papers or certify upstream
runtime compatibility. The exact local 24-output synthetic smoke remains the
declared implementation gate. No theorem or audit guarantees 50% attainability.

## Preconditions and stop condition

Before real fits, the implementation owner must demonstrate the registered
synthetic selection/prefix/save-load/full-refit checks, feature/configuration
parity and range assertions. The leader owns actual host CPU/RAM admission,
CPU-only CUDA masking, unique tmux/output ownership, source/environment/data
identity receipts and monitor visibility. Three already-required seeds are
replications, not three new hypotheses. Prior endpoint checks occur as their
fresh fits complete and must fail closed before accepting treatment results.

Read evidence: strategy.md; the parent's baseline.py, core.py, strategy.md and
seed-42 F1/F2/F3 metadata; the previous age experiment's experiment.py,
age_repair_mechanism.md, results.json and verification-independent.md;
analysis/w50_audit_20261108/{early_stopping_evidence.md,sources.json,
state_failure_history.md}; leaderboard/scorer/competition.py. Existing result
JSON confirms the age treatment's mean paired gain -0.0914366%, interval
[-0.3279122%, +0.0558201%], and adopted=false.

- Clarity: pass; intervention and subset roles are explicit.
- Verifiability: pass; meaningful control and independent replay are specified.
- Completeness: pass; the accepted fixed-best gate is incorporated.
- Big picture: pass; one diagnostic-supported change, no age-repair accumulation.
- Principle/option consistency: pass; retaining 900 rounds is the frozen control,
  fixed-300 selection and post-score retuning are explicitly rejected.
- Risk/verification rigor: pass at protocol level; implementation and runtime
  evidence are pending, with failure classification specified.
- Deliberate-review additions: not separately requested; plausible failure
  mechanisms and their checks are nevertheless covered above.

Review stop condition is met: the only definite gate gap has an explicit
accepted fix, executable paths match the parent contract, and remaining
uncertainties are testable scientific risks. No source, model or runtime file
was changed by this reviewer.
