# Independent scientific-integrity verification

2026-11-09 KST. Reviewer: native verifier `/root/roundselect_final_integrity`.
Scope: the completed training-only tree-count policy versus matched parent
`drop035_v650_confirm_20261108`, seeds 42–44, official F1/F2/F3 only.

## Verdict

**PARTIAL scientific integrity; PASS computational implementation/replay;
FAIL performance-candidate and 50% goal gates.** Required historical provenance
and observation-calibration evidence remains incomplete. Successful numerical
checks do not discharge those requirements. No promotion or publication is
supported by these results.

| Seed | Parent Total S | Treatment Total S |
| --- | ---: | ---: |
| 42 | 97.49244208059969 | 98.94406819690332 |
| 43 | 97.87370833197167 | 99.53520747946670 |
| 44 | 98.07451343718657 | 99.72224550591471 |

Parent/treatment scalar means are 97.81355461658598/99.40050706076158.
Every seed worsens. Mean matched relative reduction is **−1.6222131684%**;
the fixed paired 95% interval is **[−3.0199325886%, +1.5375682544%]**.
Treatment sample variance/SD/worst S are
0.1649981331645574/0.4061996223097178/99.72224550591471.
Mean RMSE worsens 1.515951%, pooled MAE 3.137735%, and F2 S 5.863805%,
so the auxiliary gates also fail. All candidates exceed the fixed-best 1%
bound 96.5175176597937 and the separate 50% goal 48.746221040299844.
The dated SOTA snapshot is reported separately; it is not a new guarantee
about the current board or an independently verified generalization optimum.

## Evidence

- **Source and single factor:** current experiment.py, parent core.py and
  baseline.py, and official competition.py match executed-workspace snapshots
  byte for byte. Current shared sources also match all three parent-run source
  snapshots. The captured strategy is a prefix of the reviewed current strategy:
  appended runtime/diagnostic records do not replace the frozen rule. The
  runner calls the original loader, feature construction, custom objective,
  component clipping and offset functions. It preserves tail4 at550,
  spectral coefficient0.1/scale50 and normal per-subset base_score. Only the
  effective prior-inference/full-refit tree-count rule changes. Changing prior
  prefixes necessarily changes offset values under the unchanged formula.
- **Actual input identity:** independently selected/sorted parquet columns and
  manually concatenated all 18 training/query head matrices. They match the
  loader/design output exactly, including NaNs, original img_age_h, 28 source
  features and 212 expanded columns. Training origins number
  28297/30457/32665; query origins 2089/2113/2113. All nine saved subset arrays
  match reconstructed reference/calibration/stopping/offset indices. Stopping
  counts are1706/1682/1730, each last stopping origin is exactly144hours before
  its first offset origin, and each offset subset contains336rows. All fitting
  labels satisfy the unchanged cutoff/maturity rule. Query loading never reads
  y. These are computational chronology checks, not public-availability proof.
- **Fresh fitting and device:** all three train.exit files contain0, completion
  receipts identify the correct seed, and each train.log contains nine prior900
  and nine full_selected fit completions. Launch commands specify CPU/four
  threads; live-process-verification.json records empty CUDA visibility for
  all three actual training PIDs. Admissions show adequate memory and at most
  six total jobs/24threads after the third launch. The scheduler's original
  count-shadowing misclassification is preserved in scheduler_initial_terminal;
  scheduler-reconciliation.json records terminal-only artifact revalidation,
  no new fits and no changed forecasts. A supervisor validation defect does
  not change the model's scientific intervention.
- **Independent saved-model replay:** this reviewer freshly loaded all54local
  boosters with CPU1 and CUDA hidden, checked actual212columns and actual
  round counts, and replayed every seed/period. All27priors contain900rounds;
  all27full models contain their selected first-minimum counts. Every900-value
  RMSE curve is finite. Native selected-RMSE recomputation differs by at most
  2.803e-8. All parent900 endpoint, saved900 endpoint, selected-prior, full-raw,
  independently computed offset and final-clipped forecast differences are
  **exactly0**. Parameters and base_score match each actual prior/full subset.
  Original calibration origins/targets and query origins match exactly.
  Detailed head counts and differences are in verification-independent.json.
- **Fresh official score replay:** after the leader confirmed all three actual
  fits finished and all artifacts were mirrored, independently reconstructed
  combined forecast tables from nine NPZ files. They equal the three parquet
  tables exactly. Official lead-timse-6h-v1 total/F1/F2/F3 S/RMSE/TRMSE/TIMSE
  values match all stored scores exactly, with no replacements; pooled MAE,
  per-seed means, variance, SD and worst summaries match results.json.
  The frozen seed42parent anchor reproduces exactly. Frozen observations equal
  the pinned cached official-store observation dataframe. That identity check
  does not establish independent calibration of the observations.
- **Uncertainty and gates:** inspected the period-stratified circular168hour
  block implementation and its empirical-quantile TIMSE recomputation.
  All2000saved reduction/delta draws are finite; both complete saved95%
  quantile intervals reproduce. Independently rebuilt the first10deterministic
  RNG20261104 draws with separate sampling code and direct tail-conditional
  squared-error means, without using the campaign bootstrap weight helper.
  All ten match saved reductions within1e-12 and delta S within1e-10.
  Recomputed matched, fixed-best, auxiliary and50% gate decisions agree.
  This is not a second full2000draw run.
- **Chronology:** each trainer automatically scores only after its own F1/F2/F3
  forecasts freeze; faster seeds can therefore write scores before siblings
  finish. The analyst and this audit waited for all three fits. No official
  target was used for fitting or selection in the reviewed path. Historical
  earlier official periods can enter later training under the existing policy.
  That does not create a new independent blind test.

Checks executed through read-only inline Python under
`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1`.
Only this report and its JSON were written. No scientific training, forecast
transformation, evaluation-policy edit, process/runtime mutation, external
publication or digest computation was performed. Reproduction uses the frozen
loader/design and the saved inference paths in verify_selected_replay.py; this
audit additionally reconstructed matrices, offsets and bootstrap tail means
independently. It never used Booster.get_score().

## Gaps

1. The requested file hashes remain unavailable following the previously
   documented automatic approval rejection. No digest was computed or bypass
   attempted. Byte/time receipts and direct source comparisons are the bounded
   evidence available; they are not cryptographic historical identities.
2. Historical issue-time availability and version history of all ACE/SUVI/DONKI
   inputs remain incomplete. The data audit documents removed availability-delay
   processing and latest-download inputs. Passing origin/cutoff checks does not
   establish that each historical input was publicly available at issue time.
3. Independent science-calibrated corroboration of high-impact RTSW observations
   remains unresolved. calibration_crosscheck.md records unavailable2026ACE
   scienceL2 coverage; dscovr_crosscheck.md finds zero clear-quality comparison
   records. These are missing validation sources, not proof the official truth
   is wrong. No truth correction or achievable error floor follows from them.
4. All three actual training-host Git commits are null. The supplied
   source_commit d64c40339d770803cf46415bd383971dd30f00cd names a dirty source
   reference, not an actual verified remote checkout. Executed-source capture
   occurred2026-11-08T15:27:23.503435Z, after15:08:52–56Zlaunches; the broader
   environment snapshot was captured15:27:48Z. Neither is labeled prelaunch.
   Run-start imported numerical versions and retained source/data byte/time
   receipts improve reproducibility but do not prove immutable historical state.

The apparent NumPy version conflict is resolved, not silently ignored:
run-start imports report2.4.6. imported-runtime-versions.json shows the actual
module from the dedicated training environment and also a second2.2.6dist-info
under power_forecast. The flattened distribution listing overwrote metadata
with that other entry. No dependency was changed during this audit.

## Risks

Repeated leaderboard exposure and the parent's exploratory feature selection
remain. Conditional block bootstrap across three fitted seeds does not establish
unseen-future generalization or erase prior selection. The declared intervention
jointly changes full-model complexity and the selected-prior inputs to the
unchanged offset policy; it cannot identify their separate causal effects.

Numerical failure is established despite these provenance limits: the frozen
policy does not meet its declared improvement gates on the evaluated artifacts.
A different rule or revised stopping metric would be a new hypothesis requiring
its own preregistration, not a permitted rescue of this failed campaign.

The leader froze strategy, results, bootstrap and report/diagnosis artifacts in
documentation-freeze.json. Their current identities were independently checked
before the final receipts. The JSON binds all168required paths plus supporting
evidence, keeps provenance_complete false, and retains overall status partial.

The frozen report.md was also checked against fresh independent reconstruction
of both arms’ no-offset/final forecasts and official scores, low/high-speed and
long-lead bias/MAE, all fixed timing-shift RMSEs, peak/range summaries and
missingness support. No-offset mean S reproduces98.9532028006→100.2129411043;
all1052official origins have at least one missing past24hour observation, so
there is no complete-history contrast. No material report correction was needed.

Stop condition: the bounded computational, result, report and artifact checks
are complete. Remaining required historical proof sources are explicitly
unavailable; no further fitting or benchmark adaptation was performed.
