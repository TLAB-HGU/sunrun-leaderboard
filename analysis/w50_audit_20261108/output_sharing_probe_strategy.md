# Preregistered training-only split compatibility diagnostic

Status: specified before candidate-gain inspection; not executed. This is an
inference/statistics diagnostic, not a fitted candidate, new experiment ID,
change of parent, or forecast-policy change. Full campaign goal remains a
verified50% official-S reduction from97.49244208059969 to48.746221040299844.

## Question and evidence

The completed fixed180 temporal multiplier was not adopted. It improved some
moderate-tail/F3 behavior while increasing low-speed and ordinary errors; the
>=700 extreme bias persisted. Separately, the frozen parent training probe has
high-negative/low-positive bias in all9heads, with useful300to900fit progress.
These facts do not distinguish missing predictive information from finite
representation, objective/regularization or noise.

Version-pinned upstream evidence in output_sharing_mechanism_evidence.md/json
shows that each24-output vector tree sums per-output split gains. Output signs
do not cancel across outputs. The three heads share no partitions with one
another; long-head error cannot prove it dominates short-head split selection.
The possible mechanism is incompatible preferred cuts within one head, not a
new physical propagation law. No extra observation is required for this test;
it uses the same available training inputs. More partitions cannot manufacture
missing eruption or calibrated observation information.

## Fixed parent, domains and inputs

Use only D=drop035_v650_confirm_20261108 seed42 saved FULL models, allF1/F2/F3,
all three original24lead heads, fixed prefixes300and900:18counterfactual root
contexts. No prefix, period or feature is selected using diagnostic gains.
The domain of each context is every original eligible mature training origin.
Use original filled targets and original20constant+24×8varying=212columns,
original values/order,144h maturity and originalcutoffs. Verify parent source,
metadata/data byte-time identities and exact feature list before inference.
No official observation or treatment predictions/scores are opened. Previous
period observations legitimately contained in later mature training remain
training data under the existing rolling policy, not a new evaluation fold.

Read/save fit-time metadata parameters and exact parent source: expected depth3,
lambda5,min_child_weight20,learning_rate.03,subsample.8,colsample_bytree.7.
Verify unspecified alpha/max_delta_step/gamma against the pinned version and an
explicitly reconstructed zero-tree runtime configuration if needed; do not treat
a reloaded model runtime configuration as proof of historical training defaults.
The parent metadata field is parameters, distinct from its experiment config. Preserve full-domain tail normalization and full24
spectral coupling when computing the existing callback gradient/Hessian.
Do not renormalize on rows, bins or outputs. No fit or forecast is emitted.
Use inplace_predict prefixes; never call Booster.get_score().

## Frozen candidates and calculations

For each of the212columns in original order, take all distinct finite-value
quantiles at levels1/32,...,31/32 from that context's training X. Keep valid
internal thresholds and both missing directions; duplicate-equivalent cuts
can be recorded once with deterministic order. Use X<threshold semantics.
Include the no-split option. Do not tune resolution or thresholds after gains.
This is a declared coarse candidate grid, not a claim to reconstruct XGBoost's
actual256-bin sketch or historical feature/row sampling masks.

Evaluate full original callback before candidate/node aggregation. Sum float32
callback derivatives using float64 accumulators and the pinned2.1.4 gain rule
Q_j=-u_j*soft_threshold(G_j), with actual lambda/alpha/minimum-Hessian and
max-delta semantics. The original parent metadata/config must match the rule.
The primary common feasible set requires both child Hessians>=min_child_weight
for every output, so scalar-specific feasibility is not mixed with sharing.
Retain separately counted infeasible candidates and directions; do not silently
turn invalid gains into evidence of agreement.

For each common candidate c, d_j(c)=Q_j(left)+Q_j(right)-Q_j(parent).
Compute shared winner argmax_c sum_j d_j(c), individual maxima and regrets,
R=sum_j max_c d_j(c)-max_c sum_j d_j(c). Report each lead's signed contribution,
preferred cut, winner agreement, gain concentration and leave-one-output-out
winner sensitivity separately. Concentrated gain alone is not conflict.
Tie ordering is original feature order, increasing threshold, missing-left
before missing-right, with no-split preferred for numerically zero gain.
Numerical equality tolerance is1e-9+1e-8*max(1,sum_j max_c d_j(c)). Report raw
values as well as normalized R/sum maxima when denominator is positive.
If all gains are zero, classify the context as uninformative, not useful aligned
predictive structure. Unexpected negative regret beyond tolerance is an
implementation failure and halts interpretation.

## Falsification and interpretation

A common maximizing cut with positive available gain and R approximately0
locally refutes split conflict on this frozen grid/context. Consistent negligible
regret weakens material conflict here; finite-grid absence cannot globally rule
out later-node/history-specific conflict. Positive R shows only a local
counterfactual partition tradeoff and is not a generalization improvement,
capacity lower bound or evidence that50% is attainable. Noise can also create
incompatible preferred cuts. Do not choose a posthoc materiality threshold.

This is a new correction-root diagnostic at frozen fitted predictions. It is
not historical split replay: original pre-tree gradients, subsampling masks,
histogram cuts, node memberships and tie decisions were not preserved. No
claim about the original training trajectory is licensed by this probe.
No model family, alternate strategy or number of new fits is selected here.
Any subsequent justified fresh fit still needs its own strategy/diff/feature
checks and unchanged official multi-seed evaluation gates.

## Validation, resources and artifacts

Before real inference, synthetic tests must distinguish aligned, conflicting,
zero-gain and opposite-sign outputs; verify no-split, L1/L2/min-child/clipping,
missing directions, candidate/row alignment, and nonnegative regret. These
scientific calculation tests are required; they do not establish future skill.
Read-only CPU1,CUDA_VISIBLE_DEVICES='',BLAS/OpenMP1. Admit only with measured
load+1<=24 and>=8GiB available memory; preserve other campaigns. Use an owned
controller tmux/log/exit and JSON context progress. Save protocol, actual
source/config/data/model identities, features, cut counts, complete gain/regret
arrays, output tables, exit and explicit limitations under output_sharing_probe/.
No official scorer call, fitted feature/parameter, submission or parent update.


Implementation precision clarification before gain inspection: pinned2.1.4
vector output weights are stored asfloat32 before Q_j is accumulated with the
output gradient statistic. Preserve that rounding step in the diagnostic;
do not silently replace the exact path with an idealfloat64 G²/(H+lambda)
shortcut. A fractional-weight synthetic regression must distinguish the paths.
This follows from source review and is not a changed empirical success rule.


Candidate representation clarification before data inspection: use float64
linear quantiles and ceil each retained q to the smallest float32 threshold
not below q. Since original X isfloat32, this preserves exactly X<float64q
membership while retaining representable thresholds. Require min<q<=max;
deduplicate by finite-left row count and missing direction within each feature.
There are no missing-only endpoint cuts. Candidate helpers require explicit
aligned unique origin IDs for X and derivative arrays; row permutation is
allowed only when both identities and all arrays move together.


Surrogate interpretation clarification before gain inspection: the retained
spectral objective supplies a diagonal surrogate Hessian. The computed library
split score is not an exact observed decrease of its coupled nonlinear loss,
nor a forecast-S improvement. This limitation applies even with exact2.1.4
arithmetic and positive available gain; report it with every outcome.


No-split interpretation clarification before real gains: shared no-split can
coexist with positive individual maxima and therefore informative conflict
(e.g.candidate contributions[[3,-4],[-4,3]] yield shared maximum0,individual
sum6,regret6). Classify informativeness from the individual-max sum, not the
shared selected candidate. The no-split score0 is relative to an unsplit
correction-root Q(parent); it does not mean no constant correction is possible.


Rare-feature support limitation before real gains: an unweighted31-quantile
grid can omit a feature whose variation is confined to a sufficiently rare
tail. Preserve per-feature finite/missing/cut counts and report zero-cut
features explicitly. Absence of conflict here cannot rule out rare-event
partition conflict or information effects; do not use this probe as a global
capacity or attainability bound. The candidate grid remains unchanged.
