# Transfer follow-up v2 — preregistered2026-10-28

Previous transfer experiments all completed_rejected. Seasonal pooled MSE:
baseline6495.240/low2529.718/high20804.058; center6347.743/2924.722/18699.049;
recency6566.891/2479.820/21314.295; recent3657764.270/2947.544/25144.494.
No candidate reached confirmation or actual F1/F2 evaluation. Centering improved
total and high errors but worsened low errors; this is not successful transfer.

## Fixed directions

### hyper_gain

Inspired by Hyper-Trees2405.07836v1, first2024-05-13: trees output target-model
parameters and optimize decoded prediction loss. Use per-horizon affine decoder
pred=a+b*g(X), a=current ACE speed, d=causal recurrence forecast-current speed,
b=sign(d)*clip(abs(d),50,300), sign(0)=+1. Missing recurrence uses b=100.
Train target(y-a)/b and sample weights b^2, making the unregularized squared
data loss exactly decoded physical squared error. Do not normalize these weights.
Raw237 inputs and fixed400-tree LightGBM parameters are unchanged. Positive
denominator floor and bounds are engineering choices; this is not their full AR
implementation. Restore a+b*g before clipping/scoring. No coefficient-range or
future performance guarantee is implied.

### gaussian

You et al.,2406.13871v1, first2024-06-19, Algorithm1: standardized running loss
with Gaussian weights. Train a100-tree pilot on mature training rows only,
compute each training row's squared residual separately per horizon, standardize
using those training losses, and compute exp(-z^2/2), floor0.1, mean-normalize.
Refit a new400-tree raw-target model using frozen weights. Save pilot/weight
receipts. This static two-stage reweighting and floor differ from the paper's
dynamic PyTorch resampling; weighting is not claimed to be out-of-fold. It may
downweight genuine high-speed cases, so high-region gates remain mandatory.

### mc_calibration

Inspired by regression multicalibration gradient boosting2602.06773v1 (2026),
whose regression oracle uses features and current prediction to learn residuals.
Reserve latest20% of mature training origins as calibration. Fit400-tree base on
earlier prefix whose72h targets precede calibration start. On calibration only,
fit a32-tree depth2/4-leaf residual head with minimum128 rows per leaf and .03
learning rate, inputs=[raw features,base prediction]. Final prediction=base+.25*head.
Deploy that same prefix base; no refit after head training. Save/reload both models
and compare base-only predictions as a diagnostic. Gate against the completed
full-history baseline; reduced base history is a documented method consequence.
This fixed single correction is a component adaptation, not the full convergence
algorithm. It does not guarantee group MSE or calibration under future shifts.

## Shared verification

Reuse the completed, frozen baseline from ../e2000fs_feat30_transfer, checking
completion, source/data identity, prediction origins and scores. Three development
windows:2024-12-04..2025-02-22;2025-03-04..2025-05-22;
2025-09-04..2025-11-22. Reused preofficial confirmation:
2025-12-04..2026-02-22. Every training origin+72h precedes validation.

All three development windows require total,low(<550),high(>=550) MSE nonworse
than paired baseline; pooled total improves>=1%,high>=5%,low nonworse.
Confirmation uses total1%,high5%,low nonworse. Failures terminate with retained
results and no actual F1/F2 prediction/training/scoring. Passing lanes evaluate
F1/F2 once under inherited strict cutoffs. `transferred` is true only if combined
actual total improves1%,high5%,low nonworse against same-HPO L2. No official
tuning, automatic upload, or selection from actual results.

All validation/evaluation periods and Top30 features have previous exposure.
No interval is labeled untouched. The shared gate adapts2402.08672's temporal
paired-risk rationale and is an engineering acceptance criterion, not a guarantee.

Before launch: numerical transformed-loss equivalence, causal pilot/decoder and
calibration split tests, parity/tamper and gate tests, reduced-tree actual-data
smoke for each lane. Launch one durable service and one tmux pane per direction.
