# vanila_mse — preregistered replacement ablation

2026-11-12. User explicitly requested fresh training and leaderboard publication as `vanila_mse`.

Hypothesis: removing both tail-dependent weights and the Fourier penalty makes squared error the sole fitting objective; this may lower bulk RMSE but can worsen tail-sensitive TIMSE/S. No improvement is assumed. This is a replacement ablation, not an additive cumulative improvement.

A: fixed previous point4+spectral A900 reference, 31 source / 215 expanded features; mean3 S 96.31074491162094. Secondary comparisons: current point4-only board row (S 97.10925993861315) and direct-S-only (S 97.53022862821581). These are distinct references, not interchangeable.
B: unweighted squared error only, gradient prediction-target and Hessian 1. No target threshold, spectral/Fourier term, direct-S weighting, continuation, new features, or new fitting data.

Preserve 31 source features / 215 block inputs, 3 x 24-lead vector-leaf XGBoost blocks, 900 rounds, depth3, learning_rate .03, subsample .8, colsample .7, min_child_weight20, lambda5; seeds42/43/44; causal feature and training-label fill; training cutoffs F1 2025-12-29, F2 2026-03-29, F3 2026-06-29; 144h maturity; prior/full split; frozen training-only 336-row per-lead offset x.25; clipping [1,3000]. All retained and removed features are enumerated in the runner and run metadata. No feature changes.

Evaluation: no internal 2025 folds. All three seeds and F1/F2/F3 completed before result interpretation. Official competition.py and frozen leaderboard observations only; official labels never used for model fitting or tuning. Fixed equal seed forecast mean, no fitted weights or seed selection. Report seed mean/SD/worst and mean3 separately, S/TIMSE/RMSE/MAE and periods. Improvement candidate only if matched-seed mean S falls >=1%, all seeds improve and auxiliary RMSE/MAE worsen <=1%; user explicitly requests publication regardless, so posting is not baseline promotion. No SOTA >=2% claim without external same-protocol comparator.

Execution: local CPU only, CUDA_VISIBLE_DEVICES=''; three independent tmux sessions, six threads per seed, at most eighteen training threads. Admission requires adequate actual free memory. GPU reserved for DL; golab2 unavailable (SSH No route to host), no duplicate remote assignment. Each run under runs/mse_s{seed}/ with launch.json, train.log JSON unit/units progress, F*_predictions.npz and metadata, train.exit and complete.json. Aggregate mean3 and submission after validated artifacts, replay, zero persistence replacements. Logs/errors preserved; finished tmux sessions cleaned by natural process exit. sunrun --once checked after launch and completion.

Mechanism/reference: XGBoost reg:squarederror is squared loss regression (https://xgboost.readthedocs.io/en/release_2.1.0/parameter.html). Its unweighted residual gradient/unit curvature is compared numerically with the custom callback before training. MSE's population minimizer is a conditional mean; this does not imply better rare-tail performance or official S. Official-reference audit will be appended before full training.

Stop: all seed exits0, model replay exact, fixed mean3 official scores and live leaderboard row + architecture link verified. Failures are diagnosed before any retry; no score-driven retuning or automatic baseline replacement.

## Official reference audit, completed before full training

- XGBoost 2.1.4 parameter documentation calls reg:squarederror squared loss regression: https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/doc/parameter.rst#L353-L355
- Exact implementation uses gradient=pred-label and Hessian=1, identity output transform: https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/objective/regression_loss.h#L18-L29 . This is the half-squared-error convention; doubling g/h could alter regularization's relative scale and is deliberately avoided.
- Conditional-mean target of squared error, not a future-S guarantee: https://github.com/scikit-learn/scikit-learn/blob/19ca9c861a7adb948a19484337159a748feede54/doc/modules/model_evaluation.rst#L25-L85
- Research specialist checked official pinned sources. Tavily authentication and ReadTheDocs HTTP403 prevented those retrieval paths; official GitHub source used as fallback.
