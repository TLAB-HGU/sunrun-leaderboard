# Research: calendar-recency weighting without replacing the validated parent

## Request Type / Direct Answer

Bounded mechanism and version-pinned implementation research, 2026-11-09.
**Smooth recency weighting is a defensible conditional hypothesis, not an
established fix.** It could reduce the influence of stale regimes or stale
conditional mappings while retaining every original row and feature. Marginal
train-versus-recent distribution differences alone do not establish that the
mapping is stale. First inspect the leader's already-running fixed-prefix,
full-parent training diagnostic for **within-support recent residual drift** and
check the effective information remaining under the independently fixed legacy
weights. Do not duplicate that probe or introduce another model-selection fold.

No training data labels, scientific outcomes, model artifacts, or query truth
were opened in this research lane. No new model, half-life search, experiment
quota, coefficient fit, or performance gain is proposed as established.

## Original Papers: evidence and applicability

1. [Nakagawa, Nozawa & Shinbori (2019), *Relationship between the low-latitude
   coronal hole area, solar wind velocity, and geomagnetic activity during solar
   cycles 23 and 24*](https://link.springer.com/article/10.1186/s40623-019-1005-y).
   Original 1996–2016 observational analysis finds different CH-area/latitude and
   solar-wind-speed distributions across cycles, with a positive area–peak-speed
   association in both. This supports **changing physical regime mixtures**.
   It does not establish drift of `P(Y | all 28 current inputs)` in cycle 25,
   nor show that old observations should be discounted by a particular amount.
2. [Hofmeister et al. (2018), *Dependence of HSS peak velocity on source CH area
   and co-latitude*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5972456/),
   DOI [10.1002/2017JA024586](https://doi.org/10.1002/2017JA024586).
   Among 115 selected isolated CH/HSS associations in 2010–2017, peak speed
   depends on area **and** latitude relative to the observer. This supplies a
   concrete competing explanation: a change in the geometry mixture can change
   an apparent area–speed slope without a time-dependent physical law. Their
   isolated-event selection excludes ICME contamination; it is not validation
   of all-hour solar-wind forecasts or a remedy for missing CME observations.
3. [Zaidi et al. (2018), *On the Inter-relationships among Drift rate, Forgetting
   rate, Bias/variance profile and Error*, v2](https://arxiv.org/abs/1801.09354v2)
   ([full text](https://ar5iv.labs.arxiv.org/html/1801.09354)). Original experiments
   support hypotheses coupling drift rate, forgetting rate and learner variance:
   faster forgetting can reduce stale-data bias but leaves less information and
   can favor lower-variance learners. This is streaming/classification evidence,
   not a theorem that exponential recency improves this XGBoost regression task.
   It supplies neither a 180-day half-life nor an official-S improvement estimate.

These sources motivate a diagnostic, not an attribution of the observed shift.
Roughly 27-day recurrence and the roughly 11-year activity cycle describe
rotation and long-term solar activity respectively. Neither establishes a
180-day decay constant; dividing or combining them would not supply one.

## Distinguish the hypotheses

**Covariate shift** changes `P_t(X)` while retaining `P_t(Y|X)`. In the idealized
case, density-ratio weighting can transfer risk between overlapping covariate
supports. **Conditional shift** changes `P_t(Y|X)`; weights depending only on X
cannot generally recover a changed conditional distribution. Calendar-recency
weights are neither a measured density ratio nor proof of either kind of shift.
They assume recent labeled examples better represent the future joint process.

This assumption can fail under recurring regimes, abruptly changing conditions,
sensor/feature defects, or sparse recent tails. Reweighting existing rows cannot
create support for an unobserved eruption, its missing precursor, or a speed
regime absent from training. It can only alter how strongly observed examples
influence the fitted mapping. Those are mathematical scope limits, not empirical
claims of failure in the current campaign.

## Legacy candidate and comparability boundary

The requested history file is actually
`analysis/w50_audit_20261108/loss_failure_history.md`; it was read. The legacy
recipe was read without executing `research/rerun_df_ffill_20261103/advanced.py`:
its `RECIPES` assignment at line 27 contains
`e2000fs_feat30_transfer_recency`:

```
age_days = (max(fit_origins) - fit_origin) / one_day
raw_r = 0.1 + 0.9 * exp(-log(2) * age_days / 180)
r = raw_r / mean(raw_r)
```

This retains positive weights on all rows. The **excess above the floor** halves
in 180 days; total raw weight at age 180 days is 0.55, not 0.5. Mean normalization
rescales the floor as well. The origin reference is the maximum eligible fit
origin, not a query timestamp. Preserve the original maturity/cutoff rules.
The 180-day value is an independently fixed **legacy candidate**, not a physical
constant or a value to tune on F1/F2/F3.

That code uses LightGBM, 30 source features and ordinary L2 with sample weights;
it is not a clean addition to today's 28-feature XGBoost tail4+spectral parent.
The leader also reports that an earlier covariate-shift XGBoost arm kept point4
but removed spectral/current offset; that detail is supplied context, not
independently re-inspected here. The recorded direct-S replacements removed
point4 and spectral, and failed their specific comparisons. Those results are
negative evidence for those interventions, not a general proof against smooth
recency weighting of an otherwise unchanged parent. Likewise, a failed
round-count selector is not itself evidence of stale conditional mapping.

## Official Docs and Source-Reference Evidence: XGBoost 2.1.4

The official v2.1.4 tag resolves to commit
[`62e7923619352c4079b24303b367134486b1c84f`](https://github.com/dmlc/xgboost/commit/62e7923619352c4079b24303b367134486b1c84f),
released 2025-02-06. The readable
[2.1 branch tutorial](https://xgboost.readthedocs.io/en/release_2.1.0/tutorials/custom_metric_obj.html)
is grounded here by exact release source:

- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:doc/tutorials/custom_metric_obj.rst:L91-L101`
  ([link](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/doc/tutorials/custom_metric_obj.rst#L91-L101)):
  DMatrix supplies labels and weights to the callback; the tutorial's example
  explicitly does not use weights.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:python-package/xgboost/core.py:L2100-L2108`
  ([link](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/python-package/xgboost/core.py#L2100-L2108)):
  the native custom path calls `fobj(pred, dtrain)` and forwards returned
  gradients/Hessians to `boost`.
- `dmlc/xgboost@62e7923619352c4079b24303b367134486b1c84f:demo/guide-python/custom_softmax.py:L46-L80`
  ([link](https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/demo/guide-python/custom_softmax.py#L46-L80)):
  the official custom-objective example reads DMatrix weights and explicitly
  multiplies **both** gradient and Hessian by the row weight.

Thus merely setting DMatrix row weights is not an automatic native custom-loss
multiplier. For a parent objective additive over origins, with spectral coupling
within each origin's output block, a cumulative change is mathematically:

`L_B = sum_i r_i L_A,i`, hence `g_B,i = r_i g_A,i` and `H_B,i = r_i H_A,i`.

Use one calendar scalar across all outputs of that origin. Preserve the existing
tail4 weighting, spectral derivatives/Hessian approximation, offset, rows,
features, seeds and fitting configuration. Multiply the **whole original
per-origin objective** once; do not replace tail4 with recency, apply recency
only to the point term, or double-apply point4 through another weight path.
If a spectral component instead couples different rows, derive its weighted
loss explicitly before using a rowwise shortcut. This is an implementation
condition, not a claim about uninspected current code.

Before any future fit, verify all-ones recency reproduces parent derivatives;
check both arrays' row order, shapes, positivity/finiteness, and weight scale.
Check exact derivatives on small synthetic arrays, while identifying any
inherited approximate Hessian honestly. Mean-normalizing recency does not
necessarily preserve the combined tail-weighted Hessian mass; report it because
loss scaling interacts with fixed regularization. DMatrix weights may affect
other training/metric machinery, so preserve baseline handling rather than
claim they are globally irrelevant. These are native `xgb.train` findings;
do not silently substitute scikit-learn wrapper behavior.

## Decisive training-only diagnostic before an experiment

Use the already-running frozen full-parent prefix probe and its **existing,
predefined training diagnostic partition**. No new internal evaluation or
selection stage is introduced. Only labels mature before its training cutoff
may appear; F1/F2/F3 query targets remain excluded.

1. Separate changed input frequencies from changed residuals **within common
   support**. Use existing-feature strata with adequate counts—such as current
   speed, recurrence/CH geometry proxies and horizon—and compare signed residual
   direction and spread across the fixed older/recent training groups. A recent
   aggregate bias driven only by a different mix of regimes is not proof of
   conditional shift. A model residual trend can also reflect misspecification;
   it is suggestive rather than a test of the full Bayes conditional distribution.
2. Check whether source age, missingness or input-quality changes explain the
   trend before attributing it to solar physics. Use already available inputs;
   no new eruption catalog or outcome-linked annotations are needed.
3. Compute the fixed recipe's influence distribution and
   `ESS = (sum w)^2 / sum(w^2)` for recency alone and combined existing tail weights,
   globally and within supported training regimes. Report mass retained in old
   rare/high-speed examples and concentration in recent episodes. ESS measures
   weight concentration, not independent samples: neighboring origins and
   overlapping horizons are correlated. A positive floor preserves rows, not
   necessarily enough effective tail information.

This diagnostic does not estimate a benefit from recency weighting. It answers
whether the hypothesized stale-mapping problem and usable recent support are
present strongly enough to justify a separately preregistered cumulative test.
Do not use it to sweep half-lives, choose a new tree count, or tune F1/F2/F3 losses.

## Refutation Conditions / Reusable Takeaway

No persistent within-support recent residual drift, an explanation dominated by
input quality/composition, or inadequate recent effective support weakens the
recency mechanism. Strong concentration in a few correlated episodes raises a
variance risk despite keeping every row. If a correctly isolated cumulative
A+B later fails the existing matched-seed adoption criteria, reject this fixed
recipe; do not turn leaderboard residuals into a half-life search. Removing an
adopted component, changing maturity, or double-weighting invalidates isolation
rather than proving the mechanism false.

The handoff is a **training-only residual/support audit**, followed only if
warranted by a frozen, all-features/all-rows parent plus one time-weight change.
No particular model quota, gain, or 50% objective attainability follows from
these sources.
