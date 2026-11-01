Technical Specification for Competition Evaluation Metrics

1. Forecast product and evaluation structure

Each model is queried every 6 hours and must return an hourly 72-hour forecast of solar-wind bulk speed.

For a forecast issued at time $t_0$, denote the prediction at lead time $\tau$ by

$$ \hat{v}(t_0,\tau), \qquad \tau=1,\ldots,72\ \mathrm{h}. $$

The corresponding verifying observation is

$$ v(t_0+\tau). $$

The final competition metric is calculated by lead time first, rather than by pooling all forecast–observation pairs.

For each lead time

$$ \tau=1,\ldots,72, $$

all valid forecast–observation pairs at that lead time over the full competition period (alternatively, up to a given time if we want to provide live scores) are collected and used to calculate a lead-specific score $S_\tau$.

The final competition score is the arithmetic mean over the 72 lead times:

$$ \boxed{S_{\mathrm{final}}=\frac{1}{72}\sum_{\tau=1}^{72}S_\tau} $$

Lower scores are better.

2. Ground-truth solar-wind data

The official ground truth is the hourly ACE/SWEPAM data available from

https://sohoftp.nascom.nasa.gov/sdb/goes/ace/monthly/.

Specifically, the files YYYYMM_ace_swepam_1h.txt contain the field Bulk Speed, which is used as the target variable for evaluation.

The corresponding ACE spacecraft location is available from

https://izw1.caltech.edu/ACE/ASC/DATA/pos_att/ACE_GSE_position.txt.

This file contains the ACE spacecraft position in the GSE coordinate system.

These sources are the authoritative observation and spacecraft-location sources for competition evaluation.

3. Missing ground-truth observations

Missing or invalid ACE Solar Wind Bulk Speed observations are treated as NaN.

A forecast–observation pair for which the verifying ACE value is NaN is excluded from metric calculation.

There is no interpolation, forward filling, backward filling, or other reconstruction of missing ground truth for scoring purposes.

For lead time $\tau$, the evaluation set is therefore

$$ A_\tau=\left{(t_0,\tau):v(t_0+\tau)\ \text{is valid}\right}. $$

All competitors are evaluated using the same valid-observation mask.

4. Missing model output

Missing model predictions must not be discarded from scoring.

If a submitted model fails to provide a valid prediction for a required forecast hour, for whatever reason, that prediction is replaced by the persistence forecast.

This prevents a model from improving its score by selectively omitting difficult predictions.

For a forecast issued at $t_0$, persistence is defined as

$$ \hat{v}{\mathrm{pers}}(t_0,\tau)=v{\mathrm{latest}}(t_0), $$

where $v_{\mathrm{latest}}(t_0)$ is the most recent valid ACE Solar Wind Bulk Speed observation available at or before the forecast issue time.

The same persistence value is used for all affected lead times unless a valid submitted model prediction exists.

No observation after $t_0$ may be used to construct the persistence replacement.

If a valid ACE observation is always available exactly at $t_0$, this reduces to

$$ \hat{v}_{\mathrm{pers}}(t_0,\tau)=v(t_0). $$

5. Lead-time RMSE

For each lead time $\tau$, calculate

$$ \mathrm{RMSE}\tau=\sqrt{\frac{1}{N\tau}\sum_{i\in A_\tau}(v_i-\hat{v}_i)^2}, $$

where

$$ N_\tau=|A_\tau|. $$

RMSE is calculated independently for each of the 72 lead-time populations.

6. Percentile TIMSE

TIMSE is designed to give greater importance to forecast errors during high-speed solar-wind conditions.

It is calculated independently at each lead time.

6.1. Percentile thresholds

For each lead time $\tau$, use the valid observations belonging to $A_\tau$.

Construct 101 empirical quantiles:

$$ q_{\tau,k}=Q_\tau!\left(\frac{k}{100}\right),\qquad k=0,\ldots,100. $$

This defines 100 percentile intervals.

The percentile grid is recomputed separately for every lead time.

A single global percentile grid is not used.

The purpose of using lead-specific percentiles is to ensure that each percentile interval contains approximately the same fraction of the available verifying observations for that particular lead time. Thus, the tail-conditioning structure of TIMSE has the same probability-space interpretation at every forecast horizon.

The number of intervals,

$$ K=100, $$

is part of the definition of the metric and must not be treated as a numerical-resolution parameter.

6.2. Conditional tail MSE

For each lower percentile threshold $q_{\tau,k}$, define

$$ M_{\tau,k}=\frac{\displaystyle\sum_{i:v_i\ge q_{\tau,k}}(v_i-\hat{v}i)^2}{N{\tau,k}}, $$

where

$$ N_{\tau,k}=#\left{i:v_i\ge q_{\tau,k}\right}. $$

Thus, $M_{\tau,k}$ is the mean squared error restricted to observations exceeding the threshold $q_{\tau,k}$.

6.3. Integration in velocity space

TIMSE is integrated in velocity space, not percentile space:

$$ \boxed{\mathrm{TIMSE}\tau=\frac{1}{v{\max,\tau}-v_{\min,\tau}}\sum_{k=0}^{99}(q_{\tau,k+1}-q_{\tau,k}),M_{\tau,k}} $$

where

$$ v_{\min,\tau}=\min_{i\in A_\tau}v_i,\qquad v_{\max,\tau}=\max_{i\in A_\tau}v_i. $$

If two adjacent quantiles are equal,

$$ q_{\tau,k+1}=q_{\tau,k}, $$

that interval contributes zero.

The normalization by

$$ v_{\max,\tau}-v_{\min,\tau} $$

gives TIMSE the same physical units as MSE.

For reporting and combination with RMSE, define the tail-sensitive RMSE

$$ \mathrm{TRMSE}\tau=\sqrt{\mathrm{TIMSE}\tau}. $$

$\mathrm{TRMSE}_\tau$ has units of km/s and can therefore be compared directly with ordinary RMSE.

It is important not to replace the velocity-space integral with a simple arithmetic average of the conditional tail MSE values. The percentile grid determines the evaluation thresholds, but the integration weights remain

$$ q_{\tau,k+1}-q_{\tau,k}. $$

7. Combined lead-time score

The competition score at lead time $\tau$ is the geometric mean of ordinary RMSE and tail-sensitive RMSE:

$$ \boxed{S_\tau=\sqrt{\mathrm{RMSE}\tau,\mathrm{TRMSE}\tau}} $$

or, equivalently,

$$ S_\tau=\sqrt{\mathrm{RMSE}\tau\sqrt{\mathrm{TIMSE}\tau}}. $$

The score has units of km/s.

Lower values are better.

The geometric combination gives equal importance to relative changes in the two components. Approximately, a 1% deterioration in RMSE must be compensated by a 1% improvement in tail-sensitive RMSE to leave the combined score unchanged.

8. Final competition score

After calculating $S_\tau$ independently for all 72 forecast lead times, the official ranking metric is

$$ \boxed{S_{\mathrm{final}}=\frac{1}{72}\sum_{\tau=1}^{72}S_\tau} $$

The required order of operations is therefore:

Split forecast–observation pairs by lead time.
Remove pairs with missing ground truth.
Replace missing model predictions with persistence.
Calculate $\mathrm{RMSE}_\tau$.
Construct the 100-percentile observation grid for that lead time.
Calculate $\mathrm{TIMSE}_\tau$.
Calculate $\mathrm{TRMSE}\tau=\sqrt{\mathrm{TIMSE}\tau}$.
Calculate $S_\tau=\sqrt{\mathrm{RMSE}\tau,\mathrm{TRMSE}\tau}$.
Average the 72 values of $S_\tau$.
The implementation must not first pool all 72 lead times and calculate a single RMSE/TIMSE score.

The implementation must also not calculate the metric independently for each individual 72-hour forecast window and then average those forecast-window scores.

9. Diagnostic quantities

Although only $S_{\mathrm{final}}$ determines the primary competition ranking, the evaluation code should retain and report at least:

$S_{\mathrm{final}}$;
the mean RMSE over the 72 lead times;
the mean $\mathrm{TRMSE}=\sqrt{\mathrm{TIMSE}}$ over the 72 lead times;
mean forecast bias;
the complete $S_\tau$ curve for $\tau=1,\ldots,72$;
the complete $\mathrm{RMSE}_\tau$ curve;
the complete $\mathrm{TRMSE}_\tau$ curve.
It is also useful to report

$$ S_{49:72}=\frac{1}{24}\sum_{\tau=49}^{72}S_\tau $$

as a diagnostic of long-lead forecast performance.

This quantity is not part of the primary ranking metric. Its purpose is to identify models whose overall score is driven mainly by excellent short-lead or nowcasting performance while performance deteriorates strongly toward 72 hours.

Mean forecast bias should also be displayed alongside the primary score.

10. Official baseline models

Official scores will be provided for at least the following three baseline models.

10.1. Persistence

For a forecast issued at $t_0$,

$$ \hat{v}{\mathrm{pers}}(t_0,\tau)=v{\mathrm{latest}}(t_0) $$

for

$$ \tau=1,\ldots,72. $$

Here $v_{\mathrm{latest}}(t_0)$ is the most recent valid ACE Solar Wind Bulk Speed observation available at or before the forecast issue time.

Persistence must not use any observation occurring after the forecast issue time.

10.2. 27-day recurrence

For valid time

$$ t=t_0+\tau, $$

the 27-day recurrence forecast is

$$ \hat{v}_{27}(t_0,\tau)=v(t_0+\tau-27\ \mathrm{days}) $$

using the historical ACE Solar Wind Bulk Speed observation 27 days before the forecast valid time.

This baseline produces a lead-dependent 72-hour forecast by replaying the corresponding solar-wind sequence observed 27 days earlier.

If the required historical ACE observation is unavailable, that baseline prediction is replaced using the same persistence rule applied to missing participant predictions.

10.3. Climatology

The climatology baseline is a constant forecast:

$$ \hat{v}{\mathrm{clim}}(t_0,\tau)=v{\mathrm{clim}}. $$

The value $v_{\mathrm{clim}}$ must be fixed before evaluation of the competition period, using the designated historical or training dataset.

It must not be recomputed from the competition evaluation period.

The numerical value of $v_{\mathrm{clim}}$ and the historical interval used to determine it must be recorded in the released evaluation specification so that the baseline is exactly reproducible.

11. Implementation invariants

The reference implementation should include automated tests for the following conditions.

Missing ground truth

If $v_{\mathrm{obs}}$ is NaN, the forecast–observation pair contributes to no metric.
Participants are not penalized for an unavailable verifying observation.
Missing prediction

If $v_{\mathrm{pred}}$ is missing while $v_{\mathrm{obs}}$ is valid, the prediction is replaced with persistence.
The pair remains part of the evaluation.
No future information

Persistence and recurrence baselines may use only information that would have been available at forecast issuance.

TIMSE grid

Exactly 100 percentile intervals are used.
Quantiles are computed independently for every lead time.
Quantiles depend only on observations, never on model predictions.
Zero-width quantile intervals contribute zero.
Integration weights are velocity widths, not equal 1/100 weights.
Aggregation

Metrics are computed independently for all 72 lead times.
The 72 combined scores are averaged with equal weight.
Different numbers of valid observations at different lead times do not result in different final weights for those lead times.
Reproducibility

The implementation must explicitly fix:

the empirical quantile interpolation convention;
timestamp parsing;
time-zone convention;
handling of duplicate timestamps;
baseline constants;
preprocessing rules.
All of these should be version controlled in the official reference implementation.

12. Summary formula

For each lead time

$$ \tau=1,\ldots,72, $$

calculate

$$ \mathrm{RMSE}\tau=\sqrt{\frac{1}{N\tau}\sum_{i\in A_\tau}(v_i-\hat{v}_i)^2}, $$

$$ q_{\tau,k}=Q_\tau(k/100), $$

$$ \mathrm{TIMSE}\tau=\frac{1}{v{\max,\tau}-v_{\min,\tau}}\sum_{k=0}^{99}(q_{\tau,k+1}-q_{\tau,k})\left[\frac{1}{N_{\tau,k}}\sum_{i:v_i\ge q_{\tau,k}}(v_i-\hat{v}_i)^2\right]. $$

and

$$ S_\tau=\sqrt{\mathrm{RMSE}\tau\sqrt{\mathrm{TIMSE}\tau}}. $$

The final competition score is

$$ \boxed{S_{\mathrm{final}}=\frac{1}{72}\sum_{\tau=1}^{72}S_\tau} $$

with lower values indicating better performance.