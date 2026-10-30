# Point4 advancement v2 — fixed2026-10-29

Direct parent remains published pointweight4 (actual total3732.268939,
ordinary3365.965969, high6071.056247). Last uniform stable-retrain seasonal
total6150.587 versus parent6260.460 and low2805.019 versus3178.683 improved,
but high18222.423 versus17380.455 worsened4.8%; no actual transfer was established.
No original sources or parent weights/models are changed.

## Fixed single-model candidates

Let y be mature training truth, t the same-fold point4 teacher prediction, p the
student prediction. All new students retain the original400tree LightGBM settings.
The teacher is original point4; student objectives below are explicitly modified
unweighted supervised loss plus retention, not unchanged point4 sample weighting.

1. `high_anchor`: gamma=15 when training y>=550, else3. Minimize
   .5(p-y)^2+.5*gamma*(p-t)^2, exactly via target(y+gamma*t)/(1+gamma) and
   sample_weight=1+gamma. Do not normalize weights: retain objective curvature
   and fixed regularization scaling. Stronger high anchoring tests retaining the
   teacher's high-speed advantage while allowing ordinary error improvement.
2. `teacher_trust`: gamma=3+12*exp(-.5*((t-y)/50)^2), same exact target/weights.
   Anchor more where teacher training residual is small. Training-only teacher
   accuracy is not a guaranteed generalization-confidence estimate and is not
   out-of-fold. The50km/s scale and3/12 strengths are engineering choices.
3. `high_regret_guard`: e=p-y, et=t-y, d=e^2-et^2, pos=max(0,d), H=1[y>=550].
   Minimize .5e^2+1.5(p-t)^2+.5*H*pos^2/10000. Gradient
   e+3(p-t)+2H*e*pos/10000; Hessian4+2H*(pos+2e^2*1[d>0])/10000.
   Convex and C1, piecewise C2. At boundary use inactive-side Hessian4. A finite
   penalty does not guarantee zero regret; tree approximation can violate the
   empirical target. Use train-target-mean centering for custom objective
   initialization and persist/add that offset during prediction and restoration.

The conditional retention idea is inspired by FlipGuard2410.00508 (2024-10-01)
and Stable Update2402.13655 (2024-02-21); teacher trust is a regression adaptation
of conditional congruence, not the original LLM objective. Retention-Centric
Framework2410.03955 uses positive-part squared risk penalties; our separable
high-example squared-error regret differs from its task-level continual-learning
algorithm and inherits no guarantee. Repeated self-distillation2407.04600
(2024-07-05) provides regression context, not a prediction of tree gains.

## Data and sequential evidence

Reuse completed exactpoint4 controls in ../e2000fs_point4_advance/runs/full_20261029_v1/baseline.
Verify sources/data/model hashes, training-index identities, prediction origins,
and metrics. Development windows:2024-12-04..2025-02-22;
2025-03-04..2025-05-22;2025-09-04..2025-11-22. Reused confirmation:
2025-12-04..2026-02-22. Every training origin+72h strictly precedes heldout start.
No heldout label is used in gamma, pseudo targets, regret penalty or teacher fits.

**Amended 2026-10-29 (user decision): the internal gates below are removed.** The runner now fits only the
F1 and F2 stages for every lane (original stage-specific point4 teachers, inherited maturity cutoffs), scores
them on the official periods and publishes to the leaderboard. Outputs go to `runs/full_20261029_v2_f1f2/`;
the gated v1 runs in `runs/full_20261029_v1/` (all three `completed_rejected` at the development gate) are
kept unchanged. The original gate text is retained for the record:

Gate unchanged: all3fold total/low/high nonworse versus point4; pooled total
improves>=1%, ordinary>=5%, high nonworse. Confirmation same percentage targets.
Failures retain metrics and stop before actual F1/F2 arrays. Passing candidates
evaluate F1/F2 once with inherited maturity cutoffs and original stage-specific
point4 teacher models. Actual transfer requires internal gates AND combined
total1%/ordinary5% improvement and high no worse than point4. No official tuning.

All validation/evaluation periods and Top30 features were previously exposed.
No untouched holdout or future guarantee is claimed. Seasonal paired gates are
engineering acceptance conditions informed by temporal-assessment2402.08672,
not its adaptive tournament or a statistical noninferiority test.

## Tests and launch

Boundary/weights and exact quadratic loss identity, trust symmetry/bounds,
regret loss numerical gradient/Hessian away from knots plus boundary behavior,
teacher/parent immutability, strict72h maturity, gate failure stops actual stages,
custom mean-offset save/reload parity, source/checkpoint hashes. Actual reduced
data3tree smoke across all6stages; only smoke forces gate access. Then register
each experiment through lbstatus and start one service+tmux pane per direction.
