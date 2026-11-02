# dirmo_24_feat30_metricweight (jsy301)

The team DirMO-24 feat30 recipe (seongeun `e2000fs_dirmo_24_feat30`) with one change: the training loss weight is
derived from the board score instead of the hand-set "4x above 550 km/s" of `dirmo_24_feat30_pointweight4`.
`experiment.py` is the full model (same code as the submitted replay scripts).

Two board entries:

| entry | loss weight |
|---|---|
| `dirmo_24_feat30_metricweight` | per row and lead: `w = a_l + (1 - a_l) n c_l(y)` |
| `dirmo_24_feat30_metricweight_lead` | the same, times a per-lead factor `S_l (1/MSE_l + 1/TIMSE_l)` (mean 1 per 24-lead block) |

## Where the weight comes from

The score of lead `l` is `S_l = sqrt(RMSE_l * sqrt(TIMSE_l))`. TIMSE decomposes exactly into per-row weights:
`TIMSE_l = sum_i e_il^2 c_l(y_il)`, where `c_l(y)` sums `dq_k / (N_k * range)` over the percentile thresholds
`q_k <= y`. So `d log S_l` is proportional to `sum_i e_il de_il [a_l + (1 - a_l) n c_l(y_il)]` with
`a_l = TIMSE_l / (MSE_l + TIMSE_l)`.

- `c_l` comes from the period's training targets.
- `MSE_l` and `TIMSE_l` come from 4-fold out-of-fold forecasts of the unweighted recipe on the same training rows
  (contiguous folds, 72 h purge).
- Nothing after the period's cutoff is used.
- Each period's model is the mean of seeds 42, 1 and 2.

What the weight looks like:

- Rows near the median get about 0.9.
- The top 1 % gets about 2–7, depending on the period.
- Rows with y ≥ 550 hold 16–27 % of the total weight, against 27–40 % under the 4x rule.
- The lead factor is about 2.3 at lead 1 and about 0.6 at lead 24. It is close to 1 for leads 25–72.

## Lab evidence before submission (pre-registered)

These are dirmo_lab round MW results, checked before any leaderboard number.

- **Windows:** 2023-07..12, 2024-01..06, 2024-07..12, 2025-01..06, 2025-07..12.
- **Training:** each window is trained only on earlier data.
- **Score:** the board metric on raw ACE truth, 00/06/12/18 UTC origins.
- **Seeds:** each arm is the mean of 3 seeds.

| arm | pooled S | vs 4x@550 [95 % block-bootstrap CI] | windows better than 4x |
|---|---|---|---|
| no weight | 118.37 | +0.16 % | 3/5 |
| 4x above 550 | 118.17 | — | — |
| metricweight | 116.79 | −1.17 % [−2.58, −0.15] | 5/5 |
| metricweight_lead | 116.66 | −1.28 % [−2.65, −0.18] | 5/5 |

How the gain splits:

- **Body error:** the metric weight keeps it at the unweighted level. RMSE is 73.0, against 72.7 unweighted and 74.4 for 4x.
- **Tail error:** it reaches the 4x level. TRMSE is 187.2, against 188.1 for 4x.

The pre-registered strict bar, "no window worse than the unweighted model by more than 1 %", was **not** met. In the
quiet 2023-H2 window the metric weight is +3.2 % worse than no weight; 4x is +4.3 % worse there. These entries are
submitted to see the leaderboard result. Neither entry was chosen by a leaderboard number.

## Differences from the lab runs

- **base_score:** the lab used a per-lead vector with xgboost 3.4.1. Here it is a scalar block mean, because
  xgboost 2.1.4 takes only a scalar.
- **Training data:** the training pool and targets are the team bundle (`train.npz` plus `official` rows every 3
  hours, with the filled `speed.parquet`). The input rows and the rebuilt rows are those of the jsy301
  `dirmo_24_feat30_delta12` entry.
