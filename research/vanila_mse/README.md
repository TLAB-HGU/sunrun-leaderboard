# vanila_mse

2026-11-12 사용자 요청에 따른 게시 완료. 31 source / 215 expanded 특징의 offset-roll 구조에서
손실만 순수 MSE(gradient=prediction-target, Hessian=1)로 바꿔 새로 학습했다.
Tail 가중치, spectral 항, direct-S 항은 없다. 각 블록 900 rounds, seed 42/43/44,
학습 cutoff·144h maturity·prior/full 분리·training-only frozen offset·clip은 유지했다.

세 seed 예측의 고정 등가중 평균이며 공식 평가 정답으로 계수나 가중치를 적합하지 않았다.

| 구간 | S | TIMSE | RMSE |
|---|---:|---:|---:|
| Total | 100.775874 | 24932.356726 | 65.881206 |
| F1 | 122.112963 | 35421.477619 | 82.263806 |
| F2 | 81.017198 | 12936.267801 | 60.053956 |
| F3 | 63.373792 | 6592.104452 | 52.324697 |

개별 seed 점수 평균 100.877281, 표본 SD 0.276684,
최악 101.061462. 위 표의 mean3 예측 점수와 구분한다.
사용자 요청 게시이며, 게시 자체는 성능 개선 판정이나 기준선 승격이 아니다.

모든 seed 학습 exit 0, 9개 seed×기간 저장 모델 예측 재생성 오차 0,
원 parquet와 비트 단위 동일, 공식 scorer 재채점 일치, 누락 대체 0을 확인했다.
상세 검증은 logs/verification.json, 실행 metadata와 학습 로그는 logs/에 있다.
weights는 로컬에 보관하고 Git에는 올리지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/vanila_mse/experiment.py --threads 6 --out /tmp/vanila-mse-retrain
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/vanila_mse/experiment.py --predict-only --weights research/vanila_mse/weights --out /tmp/vanila-mse-replay
```

## 전략 문서 (이전 strategy.md)

## vanila_mse — preregistered replacement ablation

2026-11-12. User explicitly requested fresh training and leaderboard publication as `vanila_mse`.

Hypothesis: removing both tail-dependent weights and the Fourier penalty makes squared error the sole fitting objective; this may lower bulk RMSE but can worsen tail-sensitive TIMSE/S. No improvement is assumed. This is a replacement ablation, not an additive cumulative improvement.

A: fixed previous point4+spectral A900 reference, 31 source / 215 expanded features; mean3 S 96.31074491162094. Secondary comparisons: current point4-only board row (S 97.10925993861315) and direct-S-only (S 97.53022862821581). These are distinct references, not interchangeable.
B: unweighted squared error only, gradient prediction-target and Hessian 1. No target threshold, spectral/Fourier term, direct-S weighting, continuation, new features, or new fitting data.

Preserve 31 source features / 215 block inputs, 3 x 24-lead vector-leaf XGBoost blocks, 900 rounds, depth3, learning_rate .03, subsample .8, colsample .7, min_child_weight20, lambda5; seeds42/43/44; causal feature and training-label fill; training cutoffs F1 2025-12-29, F2 2026-03-29, F3 2026-06-29; 144h maturity; prior/full split; frozen training-only 336-row per-lead offset x.25; clipping [1,3000]. All retained and removed features are enumerated in the runner and run metadata. No feature changes.

Evaluation: no internal 2025 folds. All three seeds and F1/F2/F3 completed before result interpretation. Official competition.py and frozen leaderboard observations only; official labels never used for model fitting or tuning. Fixed equal seed forecast mean, no fitted weights or seed selection. Report seed mean/SD/worst and mean3 separately, S/TIMSE/RMSE/MAE and periods. Improvement candidate only if matched-seed mean S falls >=1%, all seeds improve and auxiliary RMSE/MAE worsen <=1%; user explicitly requests publication regardless, so posting is not baseline promotion. No SOTA >=2% claim without external same-protocol comparator.

Execution: local CPU only, CUDA_VISIBLE_DEVICES=''; three independent tmux sessions, six threads per seed, at most eighteen training threads. Admission requires adequate actual free memory. GPU reserved for DL; golab2 unavailable (SSH No route to host), no duplicate remote assignment. Each run under runs/mse_s{seed}/ with launch.json, train.log JSON unit/units progress, F*_predictions.npz and metadata, train.exit and complete.json. Aggregate mean3 and submission after validated artifacts, replay, zero persistence replacements. Logs/errors preserved; finished tmux sessions cleaned by natural process exit. sunrun --once checked after launch and completion.

Mechanism/reference: XGBoost reg:squarederror is squared loss regression (https://xgboost.readthedocs.io/en/release_2.1.0/parameter.html). Its unweighted residual gradient/unit curvature is compared numerically with the custom callback before training. MSE's population minimizer is a conditional mean; this does not imply better rare-tail performance or official S. Official-reference audit will be appended before full training.

Stop: all seed exits0, model replay exact, fixed mean3 official scores and live leaderboard row + architecture link verified. Failures are diagnosed before any retry; no score-driven retuning or automatic baseline replacement.

### Official reference audit, completed before full training

- XGBoost 2.1.4 parameter documentation calls reg:squarederror squared loss regression: https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/doc/parameter.rst#L353-L355
- Exact implementation uses gradient=pred-label and Hessian=1, identity output transform: https://github.com/dmlc/xgboost/blob/62e7923619352c4079b24303b367134486b1c84f/src/objective/regression_loss.h#L18-L29 . This is the half-squared-error convention; doubling g/h could alter regularization's relative scale and is deliberately avoided.
- Conditional-mean target of squared error, not a future-S guarantee: https://github.com/scikit-learn/scikit-learn/blob/19ca9c861a7adb948a19484337159a748feede54/doc/modules/model_evaluation.rst#L25-L85
- Research specialist checked official pinned sources. Tavily authentication and ReadTheDocs HTTP403 prevented those retrieval paths; official GitHub source used as fallback.

## 2026-11-12 폴더 정리

사용자 지시로 repo를 보드 행 전용 레이아웃으로 정리했습니다. 이 폴더에는 `experiment.py`(유일한 스크립트), `README.md`, `logs/`, `results/`, `weights/`(로컬 전용)만 있고 하위 폴더는 없습니다. 결과는 `results/{tag}_predictions.parquet`·`results/{tag}_score.json`, 체크포인트는 `weights/F{n}_s{seed}.npz`입니다. 위 본문이 가리키는 캠페인 폴더(`research/*_2026…`, `analysis/…`, `runs/…`)는 삭제되어 이 폴더의 사본만 남아 있습니다.

## 2026-11-13 레이아웃 변경 (중복 제거)

- 공용 코드를 `research/_module/`로 옮겼습니다. 이 행이 쓰는 모듈: `gbm`. `experiment.py`에는 특징 목록, 목적함수, 보정 설정을 담은 `gbm.GBMSpec`만 남았습니다.
- `results/`와 `logs/`는 `research/result/vanila_mse/`로 옮겼습니다. 위 본문에 나오는 `results/…`, `logs/…` 경로는 이 위치 기준입니다. `weights/`와 `visualization/`은 이 폴더에 그대로 있습니다.
- `--out`을 주지 않으면 `research/result/vanila_mse/`에 씁니다. `--out DIR`을 주면 예전과 같이 `DIR/results`, `DIR/logs`, `DIR/weights`에 씁니다(재현 스크립트 호환).
- 검증: 새 코드의 `--predict-only`(weights/) 예측이 기존 `results/`의 s42·s43·s44·mean3와 완전히 같고 S도 같습니다. old/new `--smoke` 예측과 checkpoint 배열(부스터 바이트·calibration)도 모두 같습니다.
