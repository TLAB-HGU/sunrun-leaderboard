# Prior loss/metric evidence: bounded factual audit

Read-only specialist lookup2026-11-09KST, preserved by the leader. No current
round-selection score/prediction was read, no fitting or new experiment was
performed, and no model family/quota is selected here.

## Conclusions and comparability boundary

Native RMSE and officialS reward different behavior. No clean test on today's
28-feature parent retaining the other adopted components was established by
this lookup. DirectS-aware objectives are **not wholly untested**: two older
30-feature A+B1 replacement experiments failed their improvement gates. The
legacy SERA run and incomplete conditional-distribution campaign are different
interventions. These facts do not license either repeating rejected methods
without a new diagnosis or declaring every metric-aware method refuted.

Official competition.py computes S per lead and averages leads. TotalS is not
generally the fourth root of the product of reported aggregate RMSE² and TIMSE.

## Completed internal2025 loss comparison: not officialF1/F2/F3

research/noise_loss_axes_20261103/strategy.md and results/*/seed*/metrics.json
document three exit0seeds for each of two controls:

|Arm|Internal W seeds42/43/44|RMSE range|TIMSE range|
|---|---|---:|---:|
|PlainL2|84.7816/84.5339/84.7883|64.14–64.30|13285.6–13451.9|
|SERAL2|81.0609/80.9060/80.6291|66.05–66.64|10359.0–10390.8|

This is historical2025Q3 internal evaluation with cutoff2025-06-29,356origins,
30source features/237columns,300fixed rounds and the November3filled input.
It illustrates RMSE/S-like disagreement; it is not official2026performance,
SOTA evidence, or a current experiment-selection fold. Its internal validation
policy predates and does not satisfy today's official-period-only policy.

The same campaign implemented Gaussian/lognormal/K2lognormal mixture NLL with
input-conditioned location/scale/probabilities, training-only initialization,
y/500scale,sigma floor.02,ceiling-failure2,step bound.5 and a positive diagonal
Fisher curvature approximation. It replaces weightedL2 and omits tail4,
spectral and frozen offset. Arithmetic conditional means are the predictions.
objective_reference.py:43 and experiment.py:370 contain the paths. status.json
has only six completed seeds (SERA/plainL2); mixture42has two blocks without
normal completion, remaining distribution runs lack complete receipts, and
status says retrying_transport. These are not validated full-run hypothesis
failures and are not cumulative controls against today's parent.

## Legacy serarounds2: executed ramp differs from stale descriptions

research/e2000fs_calib11g_serarounds2/experiment.py:151 uses per-lead
w=max((y−min(y))/(max(y)−min(y)),.001), median-normalized. The executed objective
at:90 is grad=w*error,hess=w. timse_weights() helpers exist but phase_weights()
does not call them. Executed anchor is0(level targets), with no tail4,
spectral or prediction-based rho fit, despite conflicting headers/meta fields.
Thirtysource features/237columns and rounds300/300/1200 are used; counts were
selected by prior-quarterS, not nativeRMSE early stopping. Metadata reports
legacy officialS99.093 but contradicts code on weights and72vs144h maturity.
run_log.txt records nine fits/merged outputs; this lookup did not establish an
independent exit0receipt. Do not infer direct official-S training from its name.

## Rawdf port: completed but confounded by input/target changes

research/e2000fs_calib11g_rawdf/strategy.md,experiment.py:141 and
results/local_score.json document nine blocks exit0,complete grids,no fallback:
TotalS102.596166,F1/F2/F3=124.647/77.851/64.707,RMSE70.2215,TIMSE23093.783.
It fails its registered threshold101.469. It retains a median-normalized SERA
recipe, maps seven renamed/recomputed SUVI features, uses raw status0missing
targets with zero gradient/Hessian and removes all-missing block rows. Counts
are selected on prior quarters from200/300/450/600/900/1200, giving300/300/1200.
Official cutoffs and144h maturity are present, but filled-target/feature values
and origin grid differ from the current parent. Legacy99.093→102.596 is not an
isolated loss comparison. The same-input point4reference102.494 also differs
in loss/round recipe. No current validation policy is adopted from this history.

## Direct official-S-gradient replacements on older A+B1

research/offset_roll_ablation6_20261105/experiment.py:143 computes training-label
TIMSE coefficients a and refreshes per-leadM,T,S each boosting round:
k=S*[1/(nM)+a/T], globally mean-normalized; grad=k*error, approximatehess=k.
It removes point4 **and** spectral, retaining30features,900rounds,originalffill,
144h maturity,offset andB1. The spectral-offs0control separately compares
point4→S weights. Completed strategy.md:170 gives:

- S101.689/101.653/101.521,mean101.621 versusA+B1mean99.426.
- MeanΔS+2.195,CI[−.758,+4.496]: improvement gate fails; formal result inconclusive.
- RMSE+3.14,bias+4.4,TRMSE roughlyunchanged. Versuss0meanΔS+1.71,CIcrosses0.
- In-sample coefficient drift/excess extreme-tail weighting is a recorded
  trace-supported diagnosis, not a universal failure theorem.

research/offset_roll_ablation7_20261105/experiment.py:143 freezes M/T coefficients
from matching parent prior predictions on its training-only three-month
holdout. It uses no official scoring truth for coefficient fitting. Same
point4/spectral replacement,900rounds,offset/B1. Completed/rescored evidence
at strategy.md:130 gives all3exit0,score replay0,fallback0:

- S102.559/102.887/102.223,mean102.556.
- ΔS+3.130,CI[+.904,+4.584]: registered rejection.
- RMSE+.66,TRMSE+7.72,bias−4.4; documented weaker550–700pressure than point4.

This is strong negative evidence about those particular replacements. It is
not evidence that an unchanged-tail4/spectral training-only complexity selector,
or every conditional-tail method, has already failed. Current round selection
remains under its preregistered procedure until official multi-seed scoring.
