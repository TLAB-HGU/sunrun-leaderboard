# Research: XGBoost 2.1.4 output-sharing mechanism

## Request Type

Implementation reference lookup, with a training-only falsification protocol. Retrieved 2026-11-09. No fitting, treatment predictions/truth/scores, model selection, implementation, or policy change.

## Direct Answer

**Shared-split competition is possible within each 24-lead head, but long-lead domination is not established.** XGBoost 2.1.4 chooses one split by summing 24 output-specific gain contributions, then assigns 24 separate leaf weights. It does not sum signed gradients across outputs before squaring. Opposite gradient signs in two outputs therefore do not directly cancel. Signed gradients can cancel across training rows *within one output/node*. The three parent boosters are independent: worse bias in the 49–72h head cannot show that it stole split capacity from the 1–24h head. [weight_gain](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/param.h#L240-L321), [vector_split](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/hist/evaluate_splits.h#L502-L568)

Keeping one 24-output booster with the same 212 input columns and the same full-matrix custom objective, `one_output_per_tree` can preserve full-24-lead spectral coupling in the gradients while giving each output its own tree partition. The custom callback runs once on the full raw prediction matrix; only afterward does C++ slice its derivative columns to grow scalar trees. It is not necessary to create 24 independently fitted objectives or shorten the spectral window. [custom_callback](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/python-package/xgboost/core.py#L2099-L2108), [strategy_dispatch](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/gbm/gbtree.cc#L202-L269)

This is not a compute-matched or universally regularization-identical comparison. At the same 900 rounds and depth 3, scalar trees can create many more distinct partitions. Exact 2.1.4 vector/scalar gain and minimum-Hessian semantics also have differences described below.

The local history identifies no completed same-D sharing-only comparison. Prior block-size changes also changed feature width, spectral width, base score and booster count. The frozen training probe establishes in-sample high/low compression, but does not identify its cause. Sources: [structure history](output_structure_history.md), [training-tail note](training_tail_compression_note.md).

## Exact split and leaf calculations

For output j and training rows in node N, let G_j(N)=Σ_i g_ij and H_j(N)=Σ_i h_ij, where g/h are the **actual custom derivative arrays**, including whatever weighting and diagonal approximation the callback supplies. Define T_alpha(G)=sign(G)·max(|G|−alpha,0).

The unshrunk output weight is:

- u_j=0 if H_j < min_child_weight or H_j ≤ 0.
- Otherwise u_j=−T_alpha(G_j)/(H_j+lambda), clipped to ±max_delta_step when that parameter is positive.
- Stored child weights apply learning_rate to each u_j.

The vector-node score is exactly **Q(N)=Σ_j −u_j T_alpha(G_j)**. For eligible, unclipped components, this is Σ_j T_alpha(G_j)²/(H_j+lambda). Candidate split gain is **Δ(c)=Q(L_c)+Q(R_c)−Q(N)**. Candidate cuts accumulate separate G/H for every output; missing-value directions are considered. These are implementation gain units, without inserting a textbook 1/2 factor. [weight_gain](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/param.h#L240-L321), [vector_split](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/hist/evaluate_splits.h#L502-L568), [vector_apply](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/hist/evaluate_splits.h#L646-L694)

The shared split must pass the numerical gain threshold, total gain ≥ gamma, depth/leaf limits, and not have an all-zero Hessian vector on either side. Gamma is a threshold on the total vector gain, not a separate threshold charged 24 times. [vector_validity](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/hist/expand_entry.h#L198-L216)

**Version-specific caveats matter.**

- The scalar candidate gate requires *both* child Hessian sums to meet min_child_weight. The vector enumerator does not require this for every output: individual low-Hessian output weights become zero while a split can remain useful to other outputs. [scalar_child_gate](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/hist/evaluate_splits.h#L101-L104), [vector_split](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/hist/evaluate_splits.h#L502-L568), [weight_gain](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/param.h#L240-L321)
- Under max_delta_step clipping, the vector score −u·T(G) must not be replaced by the generic scalar quadratic gain routine. Their expressions differ in the pinned code. The unclipped simplification above is valid only when clipping is inactive. No unprovided parent alpha/lambda/max_delta_step values are assumed. [weight_gain](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/param.h#L240-L321)
- The Hessian supplied to the grower has one diagonal entry per row/output. Full-24 spectral loss can couple g_ij to other predictions, but neither strategy reconstructs omitted off-diagonal Hessian entries. This follows from the same full callback followed by per-output derivative storage/weight calculations. [custom_callback](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/python-package/xgboost/core.py#L2099-L2108), [weight_gain](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/param.h#L240-L321)

## Same-input scalar strategy: what changes and what can stay fixed

To interpret the semantics, retain all 212 columns, all 24 labels in each original head, the full24 spectral term, tail4 normalization, offset, original scalar base_score, training rows, cutoff, seed and derivative logic. A per-lead scalar tree may split on any of those same 212 columns; it is not restricted to that lead's eight varying inputs. Reducing inputs or the loss window would answer a different question.

Assuming num_parallel_tree=1 and fully grown depth-3 trees:

| Quantity per 24-output head, 900 rounds | Vector strategy | Scalar strategy |
|---|---:|---:|
| Trees | 900 | 21,600 |
| Maximum distinct internal split decisions | 6,300 | 151,200 |
| Maximum scalar leaf coefficients | 172,800 | 172,800 |
| Partition across outputs in a round | Shared | Independently chosen |

These are derived upper bounds, not measurements: one depth-3 vector tree has up to 8 leaves ×24 values, while 24 scalar trees each have up to 8 scalar leaves. Scalar topology has greater partition freedom despite the same fully grown coefficient count. Both perform derivative/histogram work over 24 outputs; runtime, memory and prediction speed are not guaranteed to scale by 24. Measure them if a future authorized comparison is made. [strategy_dispatch](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/gbm/gbtree.cc#L202-L269), [vector_split](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/hist/evaluate_splits.h#L502-L568)

Even identical sampling parameters/seed do not guarantee matched row or feature masks: the updater samples per tree, while the scalar strategy grows one tree group per output. Record actual masks or label this stochastic difference. L2/L1 leaf penalties apply to each output weight in both paths, but gamma, minimum-child feasibility and clipping semantics above require separate interpretation. [sampling_monotone](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/updater_quantile_hist.cc#L499-L545), [strategy_dispatch](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/gbm/gbtree.cc#L202-L269)

## Is long-lead domination fact or hypothesis?

**Hypothesis.** In the simplified unclipped formula, an output's influence depends on candidate-specific aggregated G, H, parent subtraction and regularization—not raw RMSE, gradient variance or horizon alone. A large-gradient output can favor exactly the same split as the others; that is gain concentration without partition conflict. Conversely, a smaller-scale output can lose a useful split because other outputs favor a different cut.

The existing tail-compression note compares three independent heads and reports high-tail fit improving from 300 to 900 rounds in all nine period/head combinations. This is consistent with useful remaining fit, but distinguishes neither shared partition limits from generic finite capacity nor missing information from objective/regularization effects. Pair counts repeat physical target hours; they cannot serve as independent-event uncertainty estimates. [Local probe interpretation](training_tail_compression_note.md)

## Decisive training-only diagnostic before another fit

**Specify, then evaluate candidate split compatibility on frozen parent checkpoints; do not fit a new booster.** This document specifies the diagnostic only; it does not claim execution or results.

1. Freeze the existing seed42 300/900 prefixes, all nine period/head combinations, node contexts and training row IDs before examining output-specific gains. Freeze the original 212-column schema, full24 objective and its original normalization, actual diagonal Hessians, tree hyperparameters, candidate histogram cuts, missing directions, feature/row eligibility and numerical tolerance. Use mature original training labels only.
2. Evaluate the callback on its original full training domain before selecting node rows; re-normalizing per-lead tail weights on a diagnostic subset would change the objective. Accumulate output-specific G/H for each frozen node and candidate.
3. Compute d_j(c)=Q_j(L_c)+Q_j(R_c)−Q_j(N) with the exact vector implementation. On a common feasible candidate set, including “no split” with zero contribution, compute the shared winner c*=argmax_c Σ_j d_j(c), per-output maxima m_j=max_c d_j(c), and regrets r_j=m_j−d_j(c*). The total one-step sharing regret is **R=Σ_j max_c d_j(c)−max_c Σ_j d_j(c) ≥ 0**.
4. Report signed contributions, regret by output, preferred-cut agreement, gain concentration and leave-one-output-out winner sensitivity separately. A separately labeled calculation can apply exact scalar candidate gates; do not conflate that with the common-set sharing regret.

**Refutation criterion:** if all outputs share a maximizing cut within numerical tolerance, R=0 and local one-step split conflict is absent. Consistently negligible normalized regret over the prespecified contexts weakens material conflict. Concentrated gain alone, with aligned preferred cuts, does not support the conflict hypothesis. A positive R with incompatible preferred cuts supports only a *local* partition tradeoff; it does not establish better generalization from separate trees.

**Replay limits:** a frozen-checkpoint split probe is a counterfactual correction diagnostic, not necessarily reconstruction of historical training decisions. Exact historical replay needs the original pre-tree gradients, row/feature sampling masks, histogram cuts, tie-breaking and node memberships. If these were not recorded/recoverable, disclose that gap. The complete original training domain is needed to preserve objective normalization; a smaller frozen probe can be descriptive but cannot claim exact historical gain. No finite sampled diagnostic globally rules out every later-node conflict.

An empirical gain threshold cannot be chosen after seeing results and then called preregistered. Gradient norm/correlation alone is insufficient. Tail strata can contribute G/H sufficient statistics, but separately computed stratum gains are nonlinear and **must not be summed as an exact attribution**. These diagnostics cannot justify retrospective loss/feature tuning on leaderboard labels.

## Official Docs Evidence

- [XGBoost release-2.1 multi-output tutorial](https://xgboost.readthedocs.io/en/release_2.1.0/tutorials/multioutput.html) — dense labels, experimental support, default per-target strategy and hist vector leaves. Exact v2.1.4 text is pinned in [multioutput_docs](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/doc/tutorials/multioutput.rst#L1-L71).
- Exact source rejects GPU vector growth and non-hist tree methods ([strategy_dispatch](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/gbm/gbtree.cc#L202-L269)), monotonic constraints ([sampling_monotone](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/updater_quantile_hist.cc#L499-L545)), SHAP contributions ([leaf_prediction](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/predictor/cpu_predictor.cc#L871-L927)) and SHAP interaction contributions ([interaction_shap](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/predictor/cpu_predictor.cc#L969-L974)).
- Ordinary local DMatrix `pred_leaf` **is implemented** for vector trees in 2.1.4 ([leaf_prediction](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/predictor/cpu_predictor.cc#L871-L927)). A 2023 prototype limitation is not valid evidence to deny this capability.
- Vector `get_dump` rejects formats other than dot ([dump](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/tree_model.cc#L769-L777)); model serialization has a separate vector path ([save](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/tree/tree_model.cc#L1175-L1184)). A dump failure does not establish inability to save a model. The capability list here is intentionally not exhaustive.

## Version Note

Exact target: **XGBoost 2.1.4**, official commit **62e7923619352c4079b24303b367134486b1c84f**, release commit dated 2025-02-06. Source retrieval: 2026-11-09. The 2.1 tutorial itself retains an “as of 2.0” warning; source at the exact release resolves operational details. The [2026 vector-leaf maintainer article](https://xgboost.ai/2026/08/25/introducing-the-xgboost-vector-leaf-model) explicitly concerns 3.4.1 and is not evidence of 2.1.4 feature support.

## Source-Reference Evidence

Pinned source is necessary because the tutorial does not document the gain formula, scalar/vector child eligibility, clipping differences, callback dispatch, or all version-specific feature checks. Each exact excerpt, line range and immutable URL is retained in [the companion JSON](output_sharing_mechanism_evidence.json).

- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:doc/tutorials/multioutput.rst:L1-L71` — Official v2.1.4 docs: experimental dense-label multi-output, per-target default and hist vector leaves.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:python-package/xgboost/core.py:L2099-L2108` — Full raw prediction matrix is passed once to custom objective before boost.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:src/gbm/gbtree.cc:L202-L269` — CPU hist vector path versus per-target gradient slicing and scalar tree growth.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:src/tree/param.h:L240-L321` — Exact scalar and vector leaf weights and gain routines.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:src/tree/hist/evaluate_splits.h:L502-L568` — Per-target histogram accumulation and summed child minus parent gain.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:src/tree/hist/evaluate_splits.h:L101-L104` — Scalar candidate requires both child Hessian sums at least min_child_weight.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:src/tree/hist/evaluate_splits.h:L646-L694` — Vector root and child weights; learning-rate application; unshrunk gain.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:src/tree/hist/expand_entry.h:L198-L216` — Summed split gain threshold, zero-child Hessians and depth/leaf constraints.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:src/tree/updater_quantile_hist.cc:L499-L545` — Monotonic constraints rejected for vector trees; sampling is per tree.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:src/predictor/cpu_predictor.cc:L871-L927` — Ordinary vector leaf-index prediction supported; SHAP contribution rejected.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:src/predictor/cpu_predictor.cc:L969-L974` — SHAP interaction contribution rejected.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:src/tree/tree_model.cc:L769-L777` — Vector get_dump supports dot, rejects other dump formats.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:src/tree/tree_model.cc:L1175-L1184` — Model serialization has vector-tree path, distinct from get_dump.

## Supporting Primary Literature

[Zhang and Jung, GBDT-MO: Gradient Boosted Decision Trees for Multiple Outputs, arXiv:1909.04373v2](https://arxiv.org/abs/1909.04373v2) describes summing output gains to choose shared partitions and extending histograms to multiple outputs. This is primary conceptual evidence that partition sharing can reduce redundant structure. It does not establish improved solar-wind forecasting, long-lead domination, or exact XGBoost behavior. Exa retrieval of both HTML and PDF returned the abstract only; no full-paper equation or dataset-specific effect is claimed.

## Caveats / Ambiguity Flags

No raw parent training derivatives, original masks/cuts, current treatment outputs, new fits, or diagnostic results were inspected. Actual parent regularization values beyond the supplied context were not independently established. Compatibility and complexity statements above are conditional where indicated. No maintained third-party implementation is needed because exact upstream source resolves the mechanism.

## Reusable Takeaway

Treat shared-split conflict as unproven. First quantify candidate-level gain compatibility on frozen training-only contexts with the exact 2.1.4 formula. The scalar strategy can retain the same full24 custom loss and 212 inputs, but equal rounds grant more independent partitions and have version-specific feasibility differences. This evidence authorizes no experiment selection or baseline change.

