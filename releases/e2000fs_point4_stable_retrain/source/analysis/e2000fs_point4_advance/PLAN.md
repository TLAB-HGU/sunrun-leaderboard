# Pointweight4 direct-parent advances —2026-10-29

Direct parent is the published e2000fs_feat30_pointweight4, not L2.
F1+F2 parent MSE3732.268939, ordinary3365.965969, high6071.056247.
All settings are frozen from the parent's LightGBM HPO snapshot, including
per-horizon weight4 for target>=550 and1 otherwise, mean-normalized.
Existing sources/models/results remain immutable. Reused official periods and
official-derived Top30 feature selection are disclosed; no untouched test claim.

## Literature and fixed lanes

1. `linear_leaf`: RaFFLE2502.10185v1 (2025-02-14) motivates locally linear
   tree models. Use native LightGBM4.6.0 linear_tree=True, tree_learner=serial,
   linear_lambda=10; retain parent400trees and point4 target weights. Per fold,
   rescale each input by training-only nanmean/nanstd, preserving NaNs. Constant
   and all-NaN features use safe0mean/1scale. This adds linear leaves and their
   required numerical preprocessing; it is not full RaFFLE reproduction.
2. `leaf_refit`: Stable Update of Regression Trees2402.13655 (2024-02-21),
   and official native refit documentation. Load exact point4 control trained on
   the same mature fold rows, keep splits, unweighted refit on those same rows
   with decay_rate=.9. Store a new Booster, never overwrite parent. The factor
   blends individual old/new leaf values; it does not guarantee90% retention of
   final predictions. Component adaptation, not the paper's full stable objective.
3. `stable_retrain`: from the same paper's stability-loss principle, fit a new
   direct model minimizing .5*(y-p)^2 +3*.5*(p-parent_train_prediction)^2.
   Equivalent pseudo target=.25*y+.75*parent_train_prediction and constant
   sample weight4. Preserve summed data-loss scaling and fixed regularization.
   Teacher is the same-fold point4 model, trained only on mature training rows.
   This is in-sample teacher regularization, not out-of-fold calibration. All raw
   features and400tree settings unchanged; new splits may be learned. The initial
   teacher uses point4 weights; student objective is explicitly unweighted MSE
   with stability anchor. No high-regime guarantee is assumed.

## Validation and actual evaluation

Train exact point4 controls for development windows inclusive UTC:
2024-12-04..2025-02-22;2025-03-04..2025-05-22;
2025-09-04..2025-11-22. Confirmation2025-12-04..2026-02-22 reuses the original
point4 internal predictions after exact origin/score/model provenance checks.
Every training origin+72h strictly precedes heldout start; no early stopping or
parameter tuning from validation. Fold controls/models may be shared read-only.

Development gate: each fold total,ordinary,high MSE<=its point4 parent; pooled
total improves>=1%,ordinary improves>=5%,high nonworse. Same aggregate criteria
on reused confirmation. Failed candidates retain results and are marked rejected.
Only passing candidates refit F1/F2 once with inherited maturity cutoffs and
original point4 parents. Actual transfer requires combined total improvement>=1%,
ordinary>=5%, high<=6071.056247, in addition to internal gates. No official tuning.
The goal is reducing ordinary error while retaining the parent's high-speed gain,
not merely beating the L2 model's much worse high-speed MSE.

Engineering acceptance criteria are not statistical noninferiority guarantees.
Paired seasonal validation is informed by Han et al.2402.08672 (2024-02-13), not
a reproduction of their adaptive tournament. All constants above are engineering
choices, not paper-recommended universal values.

## Execution

Shared Python environment, CPU2threads per lane. Tests: maturity and weight
boundaries, train-only scaling/future-input poisoning, stable pseudo-loss identity,
leaf split preservation and save/reload parity, fail-closed gates, source/model
hash tampering. Actual-data reduced-tree smoke before full runs. Each experiment
registered through lbstatus then launched in separate durable service+tmux pane.
Baseline is a fourth session. No original or unrelated live runner is edited.
