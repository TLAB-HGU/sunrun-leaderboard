# Fixed common-support and weight-concentration diagnostic

Frozen before scientific execution. This is one read-only follow-up to the
completed full-parent training-fit probe, not new fitting, an internal model
selection fold, or a proposal to tune a forgetting rate.

Use saved seed42 parent900-round full models and their unchanged mature
training rows for F1/F2/F3. Compare only the existing reference and calibration
subsets. They are all in-sample for these full models. Use originally observed
targets for residual comparisons, keeping imputed training labels/model state
unchanged. No foreign per-period query targets or benchmark score enter.

Three nested input-only views: fixed origin speed and rec_mean speed bins
<400/400–550/550–700/>=700 (missing separate), then CH meridian15 equatorial
area zero/positive/missing, then raw origin speed/image missingness flags.
Keep heads separate. Common support requires at least30distinct origins in
both original groups; report excluded coverage, overlapping-pair counts and
origin counts. This is a prespecified diagnostic flag, not a fitted threshold.

Within common support separately normalize old/recent pair masses. Check
exactly: total bias difference = sum w_recent*(b_recent−b_old) +
sum(w_recent−w_old)*b_old. This only standardizes the stated coarse proxies;
it does not identify the full conditional law or exclude unmeasured geometry,
input quality, target noise and estimator misspecification. Do not create a
significance test, new forecast or score-selected correction.

Audit the existing legacy recipe only: raw_r=.1+.9*exp(−ln(2)*age_days/180),
age from the latest eligible origin separately within reference/full fitting
stages, r=raw_r/mean(raw_r). No cutoff or candidate strength is selected from
results. Preserve the original per-lead tail4 normalization. Report row ESS,
flattened pair ESS (neither independent samples), tail/old/observed/imputed
mass and mean Hessian scale under a whole-origin multiplier of the retained
parent objective. No temporal weights are used to refit a model in this probe.

Evidence for consistent within-proxy recent drift plus adequate support can
justify considering a separately preregistered controlled time-weight test.
Mixed/absent drift or quality/composition explanations weaken that mechanism;
they must be reported rather than interpreted to fit a desired experiment.
Concentration warns of variance and rare-event loss despite positive weights.
No diagnostic result establishes improved future S or50% attainability.

CPU1, CUDA invisible, source/data/model byte-time receipts, own tmux/log/exit;
preserve the existing hash/publication-availability limitations. Do not change
any parent or already sealed experiment artifact.
