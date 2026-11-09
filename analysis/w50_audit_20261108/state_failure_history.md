# Bounded prior-failure audit: state, calibration and optimization

Read-only specialist lookup2026-11-08; leader preserved the report. This is an
as-of inventory of saved zero exits, completion receipts, code and local scores.
Missing/uncollected outcomes are not classified as hypothesis failures. No
training, model edits or evaluation-target fitting was performed.

## Main finding

The inspected failures test particular partitions, routing rules and one
partially completed residual architecture. They do not refute all causal
input-state amplitude/bias correction or more boosting rounds. None uses the
current frozen best single S97.49244208059969 as its fitted parent. Most retain
the older30-source-feature A+B1 baseline, whereas the current reference has
28features and no B1. Historical cross-parent score differences are not ablations.

## Completed evidence

|Arm|Completed seeds|Mean Total S|Interpretation|
|---|---:|---:|---|
|Cluster C3 causal hard route|3|99.9518|Worse than matching A+B1 at all seeds|
|Cluster F3 future-label training experts, equal inference average|3|109.4623|Partition/aggregation failure; not a global conditional offset|
|Cluster random partition R3|3|104.0703|Fragmentation control|
|Softgate temperature GMM|3|99.1028|Mixed matching-parent signs; adoption gate fails|
|Softgate fixed causal proportions|3|99.1774|Small difference from temperature gate|
|VD e1/e3/r1 GLOBAL fallback|3 each|99.3708|Refitted GLOBAL, not historical A+B1 reproduction|
|VD e2 recent-error routing|3|100.3203|Different, matured-deployment-error information policy|
|VD e2b own-expert/GLOBAL|3|99.2421|Mixed signs, below1% same-GLOBAL gain|
|Five-axis original A+B1|3|99.4262|Frozen older parent|
|Five-axis regime posterior feature union|3|99.2070|Mixed signs; mean gain~0.22%|
|Five-axis SIR feature union|3|99.0525|Mixed signs; mean gain~0.38%|
|Five-axis tail dilation|3|99.7043|No general improvement established|
|Five-axis residual MoE|2|103.1132|Worse partial results; three-seed/control gate incomplete|
|Five-axis residual uniform|1|103.8270|Incomplete matching control|

Scores are arithmetic means of available scalar Total S values, not scores of
an ensemble. Period averages and exact seed values are in each saved
runs/*/local_score.json. Cluster/softgate/VD have9/18/18 completed runs.
Five-axis has22locally scored completions out of39registrations at lookup;
controller still reports running with thread-budget restriction/SSH error.
Absent quantile_width outcomes and TiDE/PatchTST dependency/input-contract
blocks are not negative performance evidence. No final complete multiplicity/
bootstrap result was established for the unfinished campaign.

## What the interventions actually changed

- `research/offset_roll_cluster_experts/strategy.md` and experiment.py: retains
  30features,900rounds,three24h heads,tail4/spectral,offset andB1. It partitions
  training samples, fits separate prior/full experts and calibrations, then
  routes. C3 better than randomized partitions but worse than global supports
  useful partition information together with fragmentation cost. It does not
  establish that all state information is useless.
- `research/offset_roll_cluster_softgate/strategy.md`: reuses saved experts;
  no new boosting fits. Temperature uses training-only recent-three-month
  prior holdout; logistic gate learns training future-shape classes from causal
  inputs. Temperature seed totals99.01922/99.13008/99.15914 include one
  deterioration vs matching A+B1; mean advantage over fixed proportions is
  only0.0746S. Oracle future-label routing is explicitly an **invalid candidate**
  and must not be counted as deployable skill or SOTA.
- `research/offset_roll_vdcluster_20261107/report.md`, strategy.md and
  publish_audit.json: routing-only, no boosting retraining. Validation excludes
  final336offset rows. Refitted GLOBAL seed scores98.52128/99.64825/99.94289
  differ from historical A+B1 98.59147/99.68573/100.00147 because B1 weights were
  refitted. Comparing to the latter is reconstruction-confounded. e2/e2b use
  recently matured deployment errors; all18 are publish:not_required.
- `research/post_ablation_five_axes_20261108/baseline-lock.json` freezes
  original_A_plus_B1,exclude_features=[],alpha.25,horizon336offset,tail4,
  spectral,block24,B1leads6. common.py uses900fixed rounds without early stopping.
  State-feature arms explicitly preserve the baseline union. mixture_models.py
  uses four chronological training chunks,144h purge,OOF residual targets,
  residual-mean classes−50/+50,logistic gate and three soft residual experts.
  This is one input-state additive correction, not a general amplitude/bias
  family. Available MoE seeds42/43 worsen; uniform matching control only seed43
  is locally complete, so the mechanism failure is unresolved.
- `research/offset_roll_quantile_tail/strategy.md` omits spectral and offset
  when replacing mean regression with tail-weighted median. Restored tail
  weights recover much of plain pinball failure; the comparison does not
  isolate conditional correction with all parent components retained.
- `research/offset_roll_negative_combined/strategy.md` combines eight
  removals,alpha1,Hessian-only spectral and a three-seed mean. It cannot isolate
  one state/optimization mechanism or establish a best-single improvement.

## Decision-changing gaps

No inspected campaign isolates increased rounds or training-only early stopping
on the current parent. The current saved-tree training-loss decline is evidence
for learning, not an optimality certificate or proof that longer fitting helps.
Before revisiting residual correction, distinguish gating error, sample
fragmentation and OOF-to-full-fit residual transfer; finish inspecting the
existing MoE controls instead of duplicating their hypothesis. Current fresh
age-repair results remain pending; this audit neither adopts a component nor
preselects a model family, experiment quota or evaluation-policy change.
