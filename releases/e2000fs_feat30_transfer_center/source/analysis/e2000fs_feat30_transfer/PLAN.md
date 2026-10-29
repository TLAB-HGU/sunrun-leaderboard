# Transfer-gated feat30 experiments — 2026-10-28

## Observed problem

The previous single internal validation period (2025-12-04 through 2026-02-22)
contains 35.14% targets >=550 km/s. The already-exposed F1/F2 evaluation contains
13.54%. Block extreme4 improved internal total MSE 1.12% but worsened F1/F2 total
MSE 13.11%. This is evidence of failed transfer, not proof that prevalence alone
causes the reversal. Conditional low/high errors and seasonal effects also matter.

## Literature and fixed experiments

1. `center`: SFNN, arXiv:2503.23621 (2025-03-30), input-window mean centering.
   Subtract the causal last24-hour ACE speed mean from target and speed-valued
   inputs, restore that same mean at prediction. Keep squared loss in km/s.
   This is a component adaptation to LightGBM, not the full neural architecture.
2. `recency`: RIDER, arXiv:2507.13287 (2025-07-17), weighted ERM under temporal
   shift. All mature history, weight 0.1+0.9*2^(-age_days/180), mean normalized.
   Age is relative to latest training origin, not query times or future labels.
3. `recent_window`: RIDER latest-only special case; use latest365 days of mature
   training origins. The window and decay constants are preregistered engineering
   choices, not RIDER's estimated optimal covariance weights.

Shared control: identical unweighted absolute-target LightGBM, same frozen
study1/trial32 parameters as previous experiments. No new HPO. Each lane has a
separate tmux monitor and durable service; the shared control is a fourth session.

## Sequential verification and stop rules

Three development validation windows, inclusive UTC calendar dates:
2024-12-04..2025-02-22; 2025-03-04..2025-05-22;
2025-09-04..2025-11-22. For every fold training origin+72h must precede its first
validation origin. Confirmation: 2025-12-04..2026-02-22, the reused historical
preofficial validation window. None of these is described as untouched: earlier
experiments and official-derived Top30 features already exposed these data.

There is one fixed arm per lane. Development passes only if total, low-target,
and high-target MSE are each no worse than the same-fold baseline in every fold,
and pooled total improves at least1%, pooled high improves at least5%, pooled low
does not worsen. A passing lane proceeds to confirmation and must again satisfy
total improvement1%, high improvement5%, low no worse. These are engineering
acceptance rules, not a theorem or statistical guarantee.

Failed lanes finish with `completed_rejected`, retained results, and no F1/F2
prediction, fitting, or scoring. Passing lanes refit F1 and F2 with the inherited
strict target cutoffs and score once against the frozen existing scorer. Record
`transferred` only when combined total improvement>=1%, high improvement>=5%,
and low no worse than the same-HPO official L2. Report failure explicitly and
perform no official-based tuning. Publication is outside this run's scope.

Han et al., arXiv:2402.08672 (2024-02-13), supports paired comparisons under
temporal shift. Our fixed multiwindow gate is a practical adaptation, not their
adaptive-window tournament implementation. Blocked uncertainty summaries can
describe evidence; point estimates and reused data cannot guarantee future gains.

## Validation before launch

Test causal speed centering/inversion, physical-feature preservation, chronological
weight/window construction, exact72h maturity boundary, fail-closed gate, immutable
checkpoints and saved-model prediction parity. Run actual-data reduced-tree smoke
through the shared baseline and every candidate. Then launch the full services
and verify a persisted training checkpoint and active tmux panes.
