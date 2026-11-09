# What a 50% official-S reduction requires

This is an algebraic interpretation of the existing official scorer, not a new
model, fitted coefficient, attainable lower bound, or experiment selection.
Source: `leaderboard/scorer/competition.py`, `prepare_context` and
`score_competition`; empirical-weight replay is in the existing
`research/post_ablation_five_axes_20261108/analysis_kernel.py`.

For a fixed lead and fixed observed evaluation population, write

\[
M=\frac1n\sum_i e_i^2,\qquad T=\sum_i a_i e_i^2,
\qquad S_h=(MT)^{1/4}.
\]

Here `a_i` is the nonnegative coefficient obtained from the scorer's empirical
quantile widths and nested upper-tail averages. It sums to one. It depends on
the observed population and its missingness, not on model predictions. Equal
observed values have equal tail inclusion. This representation reproduces the
official TIMSE; ordinary MSE and TIMSE allocate error energy differently.

**Scale implication.** If both M and T are multiplied by the same factor c,
S_h is multiplied by sqrt(c). Halving S_h therefore needs c=1/4, equivalently
halving both RMSE and TRMSE. If M stays fixed, T must fall to 1/16 to halve
S_h; the same holds with M and T exchanged. These are per-lead statements.
Official total S averages the 72 S_h values, so a total 50% reduction does not
require every lead to halve, and aggregate RMSE/TIMSE cannot be substituted
into this formula. A 50% reduction in squared-error energy is insufficient
by itself to establish a 50% reduction in official S.

**Exact first derivative.** Where M,T>0, for prediction p_i and e_i=p_i-y_i,

\[
\frac{\partial S_h}{\partial p_i}
=\frac{S_h}{2}e_i\left(\frac1{nM}+\frac{a_i}{T}\right).
\]

This derivative concerns valid positive scored predictions with fixed observed
targets. Raw outputs subjected to clipping or persistence replacement require
the associated chain rule and boundary treatment. Total-S derivatives carry
the additional 1/72 factor.

Thus ordinary unweighted RMSE and S have different local priorities. The
baseline's retained step weight (4 for y>=550, 1 otherwise, normalized per
lead) and spectral residual term are also not this exact derivative. This
establishes an objective mismatch, not evidence that replacing a successful
surrogate improves future S. The historical direct-S replacement tests on the
older parent failed their adoption gates; see `loss_failure_history.md`.
The current experiment retains the objective and changes only the pre-cutoff
tree-count rule. Its stopping RMSE can improve while S worsens.

**Information versus decision rule.** In a population version, let the tail
coefficient be w(Y), normalized so E[w(Y)]=1, and let X denote all actually
available inputs. Define M=E[(f(X)-Y)^2] and T=E[w(Y)(f(X)-Y)^2], with finite
positive moments and differentiation allowed under expectation. Empirically
w_i=n*a_i. With missing targets these expectations concern the observed
evaluation distribution; no missingness assumption here identifies a complete
physical population. At a differentiable stationary predictor, ignoring
function class restrictions, the conditional first-order condition is

\[
f(X)=\frac{E[Y\mid X]/M+E[w(Y)Y\mid X]/T}
                 {1/M+E[w(Y)\mid X]/T}.
\]

M and T themselves depend on f. This is a necessary stationary condition,
not a closed-form globally optimal estimator or an implemented fitting rule.
It illustrates why knowing only a conditional mean need not determine the
metric-optimal prediction: the conditional tail distribution matters too.
Missing eruption information, instrument errors, target imputation and temporal
distribution changes can all affect that conditional distribution. This
equation does not establish which cause dominates the actual residuals.

Evaluation truth must not be used to estimate w, M, T or these conditional
moments for model fitting. Any future justified fitting intervention would
need its own pre-cutoff, single-factor protocol and fresh training; no such
intervention is selected by this note. Empirical weights estimated on training
can differ from official evaluation weights under regime and missingness
changes. More metric-matched optimization cannot manufacture unavailable
information, while poor optimization alone cannot prove irreducible error.

Current evidence therefore cannot certify either 50% attainability or
impossibility. The existing restricted error-energy-removal sensitivities only
show that fixing isolated masks while holding all other errors fixed is
insufficient; they are not information-theoretic lower bounds. Genuine future
confirmation remains separate from conditional uncertainty on repeatedly
exposed F1/F2/F3 periods.
