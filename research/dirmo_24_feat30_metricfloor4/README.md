# dirmo_24_feat30_metricfloor4 (jsy301)

This is the team DirMO-24 feat30 recipe (seongeun `e2000fs_dirmo_24_feat30`) with one change: the training loss weight. The weight is derived from the board score, and the "4x above 550 km/s" rule of `dirmo_24_feat30_pointweight4` sets its floor. `experiment.py` is the full model and has the same code as the submitted replay script.

## Weight

For each row i and lead l:

- **Score-derived weight:** `m_il = a_l + (1 - a_l) n c_l(y_il)` with `a_l = TIMSE_l / (MSE_l + TIMSE_l)`. This is the weight of the earlier entry `dirmo_24_feat30_metricweight`.
- **Weight used:** `w_il = max(m_il, 4 if y_il >= 550 else 1)`, normalised to mean 1 per lead.

As a result:

- **Below 550 km/s:** rows weigh about what they weigh under the 4x rule.
- **550–750 km/s:** rows get the 4x level.
- **Top 1 %:** rows keep the larger derived weight, about 5–7.

`c_l` comes from the period's training targets. `MSE_l` and `TIMSE_l` come from 4-fold out-of-fold forecasts of the unweighted recipe on the same training rows (contiguous folds, 72 h purge), so nothing after the period's cutoff is used. Each period's model is the mean of seeds 42, 1 and 2.

## Why the floor

On the board, `dirmo_24_feat30_metricweight` scored 103.75, against 100.48 for the same recipe with 4x weighting.

- **The whole loss is in TIMSE.** Its RMSE was lower.
- **It grows with lead:** +0.6 %, +2.1 % and +6.3 % for the 1–24 h, 25–48 h and 49–72 h blocks.
- **It comes with a lower forecast level.**

The derived weight gives 550–750 km/s only about 1.0–1.3, against about 2.8 under 4x. The floor gives the tail at least the 4x weight everywhere and keeps the derived weight's extra emphasis on the top 1 %.

## Lab evidence before submission (pre-registered)

These are dirmo_lab round HW results, checked before any board number for this entry.

- **Library:** xgboost 3.4.1.
- **Windows:** 2023-07..12, 2024-01..06, 2024-07..12, 2025-01..06, 2025-07..12. Each window is trained only on earlier data.
- **Score:** the board metric on raw ACE truth.
- **Seeds:** the mean of 3 seeds.

The gate compares each arm with 4x@550 separately in two groups of windows:

- **High-speed windows** (2025-H1 and 2025-H2, 22–28 % of hours at or above 550 km/s): all of the gate's checks are applied here (see below).
- **Quiet windows** (the other three): the arm must not lose more than 1 %.

The high-speed checks are:

- overall S;
- each window within +0.5 %;
- the 49–72 h block;
- TIMSE;
- a mean level no lower than 4x.

| arm | high-speed S vs 4x [95 % block-bootstrap CI] | quiet vs 4x [CI] | all 5 windows | gate |
|---|---|---|---|---|
| metricweight (earlier entry) | −0.40 % [−1.99, +1.03] | −1.44 % | −1.17 % | fails (49–72 h, level) |
| **metricfloor4 (this entry)** | **−0.71 % [−1.14, −0.34]** | −0.25 % [−0.67, +0.31] | −0.55 % | passes |
| 8x at 25–72 h (not submitted) | −1.03 % | +1.41 % | +0.07 % | fails (quiet) |

Caveats:

- **The gate is a hypothesis.** It was written after one board result (metricweight vs 4x), and this entry is its first real test.
- **One quiet window is worse:** 2023-H2 is +0.16 % (49–72 h: 117.05 vs 116.21).
- **The library differs:** the lab used xgboost 3.4.1 and this entry runs 2.1.4. In the lab, single windows move by about ±1 % between the two versions.
- **No board selection:** the entry was not chosen by any board number.

## Differences from the lab runs

- **base_score:** the lab used a per-lead vector. Here it is a scalar block mean, because xgboost 2.1.4 takes only a scalar.
- **Training pool and targets:** the team bundle.
- **Input rows:** those of the jsy301 `dirmo_24_feat30_delta12` entry.
