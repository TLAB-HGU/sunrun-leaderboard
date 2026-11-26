# current_offset_roll_cpc4_tail

## Strategy (registered before training)

Arm `CPC-MGDA4-TAIL`: PureDirectS900 initialization; mgda4_tail single-tree proposals, raw-risk gate, <=180 added trees. Fixed methodological parent `current_offset_roll_direct_s`. Best local recipe anchor is `e2000fs_calib_offset_roll_scap`; Point4 and pureDirectS comparisons are reported separately. This is an unverified transfer hypothesis; not an established performance gain.

Retain all original31source features in identical215expanded-column order,24lead blocks x3, seeds42/43/44, F1/F2/F3 cutoffs,144h maturity, causal feature and training-label fill, prior/full split, final336calibration origins x.25 frozen offsets, clipping[1,3000]. No current ongoing feature-selection/HPO component is silently adopted.

SEP replaces only A900's point4 term with a centered Gaussian-smoothed loss using train-only histogramgamma and targetrange, delta.01. lambda1 symmetric is the comparator for lambda73 asymmetry. Spectral.1 and S-cap continuation remain. This is a loss replacement ablation, not full EWT/attention-LSTM reproduction; a statistical hard threshold remains.

InHaR adaptation uses a fixed seed42 A900 teacher's three purged prequential chunks of the actual prior/full fit. A bounded1+rank(hardness) modifier multiplies original mean-one point4 weights; spectral/S-cap original risk population stays unchanged. The null arm permutes the same eligible seven-day hardness blocks before weighting. Warm-up and unsupported rows remain with neutral modifier. No six-model ensemble, shuffled OOF, Gaussian-noise augmentation, or exact RO-equivalence claim. The inherited550rule remains.

CPC reuses the matching pureDirectS900 checkpoint and adds <=180 single-tree proposals in one ordinary booster per24lead block. Feasibility budgets are its own initial raw training risks, not Point4's. Fixed initial gradient scales and common step scale, simplex min-norm direction, actual serialized-tree raw-risk acceptance and exact rollback are verified. C-S4 versus MGDA4 changes the proposal only; MGDA4 versus MGDA4-tail changes Tail gate only. No claim that raw-train feasibility guarantees clipped/offset or future evaluation improvement. For this row only the declared arm's architecture executes; other paragraphs document controlled siblings.

Primary official endpoint checks S/MSE/MAE/TIMSE/new continuousTailMAE_CPC together on fixed observation masks. Report ordinary/bulk/TIMSE-tail metrics and eval-onlytop1/5/10% errors, all periods/leads, seed mean/SD/worst and fixed mean3. Practical candidate gate separately requires >=1%S gain against bestlocalanchor, matched3seedsS improvement and no periodS/MAE/RMSE>1%worsening. Paired7day bootstrap2000/RNG20261113 with frozen multiplicity family. No official-label tuning, calibration, seed selection, automatic publication or baseline promotion.

Full immutable specification, retainedfeature list, exact baseline diff, versions/input/observation identities, teacher cache, gradient tests and final feedback are in `/home/t-lab01/sunrun-paper-followups/`. `--out` follows standard results/logs/weights contract. Official scoring for the campaign is deferred until the entire prediction cohort is frozen.

References: [SEP](https://ijcai.org/proceedings/2024/0568.pdf), [InHaRv2](https://arxiv.org/html/2607.20173v2), [MGDA](https://arxiv.org/html/1810.04650v2). Paper/source inconsistencies and explicit adaptations are in the campaign source notes. These sources do not prove improved solar-wind/XGBoost performance.

## Status

Prepared 2026-11-13T08:41:49.586660+00:00. Full experiment pending; completion requires exit0/allperiods/seeds/checkpoint replay/official evaluation. Launching is not completion.

## Completed controlled cohort

2026-11-13T21:34:37.556887+00:00; all seeds/periods exit0 and saved models+offset replay exactly. Official metrics, Pareto/practical/statistical decisions and failure classification: `research/result/current_offset_roll_cpc4_tail/logs/campaign-report.md`. No automatic publication or baseline promotion.

| Total mean3 | S | MSE | MAE | TIMSE | TailMAE_CPC |
|---|---:|---:|---:|---:|---:|
| Result | 97.530244 | 5204.284750 | 53.616192 | 19317.042549 | 71.992767 |

Endpoint classification: no_primary_metric_improvement; practical1% gate=False; adoption=not automatically promoted. Causal diagnosis requires the recorded fit/teacher/projection diagnostics; no parameter sweep is automatically launched.
