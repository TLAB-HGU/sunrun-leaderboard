# e2000fs_calib_offset_roll_relevance_fs

## Strategy (registered before training)

Arm E-FS: point4 + fixed selected additional information. Methodological baseline: `e2000fs_calib_offset_roll`. Best local recipe anchor: `e2000fs_calib_offset_roll_scap`; these comparisons are separate. This is an unverified hypothesis, not an established improvement.

All original31 features are retained in their original input order. Additional constant features: ['v_lag1', 'v_m6', 'v_lag3', 'v_lag6']. New total35source /219expanded per24lead block. Selection uses original-observed pre-F1-prior-calibration training data, lambda3/gamma1 continuous relevance and16-bin weighted histogram MI. No official F1/F2/F3 score is used in selection or HPO.

Seeds42/43/44, official F1/F2/F3 cutoffs,144h maturity, filled training labels,24lead x3 vector-leaf blocks,900tree budget, original train-only frozen offset and clip are retained. Parameters: `{"colsample_bytree": 0.7, "device": "cpu", "eval_metric": "rmse", "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 20, "multi_strategy": "multi_output_tree", "objective": "reg:squarederror", "reg_lambda": 5, "seed": 42, "subsample": 0.8, "tree_method": "hist", "verbosity": 0}`. Prior/full CDFs (CDF arms) fit original observations of their exact own fitted subset; no calibration labels fit the prior objective. Direct-S/MM arms replace point4 and have no fixed target-speed threshold or spectral term. Other arms keep their declared objective only.

FS adds only the four fixed features. HPO changes only training-only chosen parameters. CDF pooling and per-lead distributions are separate matched ablations. Direct-S/MM differ only coefficient refresh within matched30round stages, with common fixed initial block scale and same stage-start Q/S acceptance. Three24lead blocks remain the original architecture; no heterogeneous ensemble is added.

Criteria: frozen cohort only once officially scored; all MAE/MSE/TIMSE/bulkMAE/tailMAE/S improve against Point4 for Pareto support. Practical publication candidate separately requires >=1%S improvement against frozen best anchor, all matched seeds improve, no periodS/MAE/RMSE>1%worse. Report seed mean/SD/worst, fixed mean3, temporal-block uncertainty. No automatic publication or baseline promotion.

References: [XGBoost multioutput](https://xgboost.readthedocs.io/en/release_3.1.0/tutorials/multioutput.html), [MM](https://pmc.ncbi.nlm.nih.gov/articles/PMC10319036/), [MI API](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html). They support implementation/mathematical conditions, not measured solar-wind benefits.

Full frozen specification, feature list, provenance, baseline diff, input identity, versions, rolling boundaries and final failure analysis: `/home/t-lab01/sunrun-continuous-relevance/`. Campaign runs use `--out` and deferred cohort scoring. Raw retrospective source availability has the same unverified historical ingestion assumption as the baseline.

## Execution status

Prepared 2026-11-13T06:41:58.539495+00:00; full experiment results pending. Completion requires exit0, all periods/seeds, exact replay, predictions/metadata and official frozen-cohort evaluation. No launch is a completion claim.

## Completed controlled cohort

2026-11-13T19:29:31.126784+00:00; all seeds/periods exit0, replay validated. Results and failure analysis: `research/result/e2000fs_calib_offset_roll_relevance_fs/logs/campaign-report.md`. No automatic publication or baseline promotion.
