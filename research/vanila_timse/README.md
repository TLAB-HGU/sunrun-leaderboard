# vanila_timse — pure TIMSE replacement ablation

2026-11-12 user-requested follow-up. State: implementation prepared; full training is dispatched by the external campaign controller.
Dependency: finish and validate all vanila_mse seeds before dispatch. The single-architecture experiment.py uses the same CLI and flat outputs as vanila_mse. Campaign execution lives in ~/sunrun-vanila-followups/.

Hypothesis: pure training-distribution TIMSE weighting changes tail-error behavior relative to pure MSE. No improvement is assumed. Baseline A is vanila_mse, B is pure TIMSE with the same architecture; this replaces the loss and is not cumulative baseline promotion.

Change only the fitting objective from unweighted MSE to TIMSE. Use the exact official 100-quantile TIMSE coefficients computed separately per lead using training labels only (leaderboard/scorer/competition.py; existing timse_coefficients in research/e2000fs_calib_offset_roll_scap/experiment.py is a reference to audit before reuse). Normalize coefficient scale explicitly and record its regularization implications before training. No MSE mixture, direct-S, spectral term, leaderboard-label fitting, or tuned weights.

Retain all 31 source / 215 expanded features, vector-leaf XGBoost 3x24-lead blocks, 900 rounds, all MSE hyperparameters, causal filling, prior/full split, frozen training-only offsets, clipping and official cutoffs. Seeds 42/43/44, fixed equal seed forecast mean. Record feature list, baseline diff, library versions, commit, splits, seeds, parameters and objective normalization before dispatch.

Evaluate F1/F2/F3 only with the same frozen observations and official scorer; no additional internal evaluation folds or score-driven tuning. Report each seed, mean/SD/worst, mean3, S/W/MAE/RMSE/TIMSE and period results. Diagnose failures before any new hypothesis. Preserve the verified baseline regardless of outcome. Candidate gate: W at least 1% below the matched best single-model baseline, no serious auxiliary regression; SOTA 2% target is a separate claim. Queue registration does not authorize unconditional leaderboard publication.

Execution preflight: check actual local/golab2 CPU, memory and jobs; no duplicate assignment. CPU GBM only, CUDA_VISIBLE_DEVICES='', independent tmux sessions and logs, explicit --experiment/--out/--device cpu/--threads. Use runs/{name}, launch.json, JSON unit/units progress, F*_predictions.npz and metadata, train.exit, complete.json. Check sunrun after launch and completion. No placeholder resource processes.

Completion: exit0 for all seeds, complete artifacts, objective numerical verification, exact replay, validated official scores; record failure cause/adoption decision. Before execution, audit referenced scorer implementation and official XGBoost objective documentation, and freeze the precise loss scaling. References establish mechanisms, not a measured improvement.

## Objective frozen before execution

For each lead and each fit (prior/full independently), compute the official linear 0..100 percentile thresholds from that fit's training labels. Coefficient c_i is the sum of each qualifying tail interval width divided by target range and tail count. Thus sum(c)=1 and TIMSE=sum(c_i*error_i²). Use w_i=N*c_i, gradient=w_i*(prediction_i-target_i), Hessian=w_i. This is half TIMSE scaled by N, preserving mean curvature one like the MSE objective while allowing target-dependent local curvature. Lambda5 and min_child_weight20 retain their numeric values; their local effective strength changes as an intended consequence of weighting. Quantiles/coefficients use float64 original training labels; callback derivatives return float32 for XGBoost. Degenerate constant targets fail closed. The baseline unweighted mean base_score is retained.

All original feature generation, 900 rounds, architecture, split, offsets, clipping and seeds are retained. This isolates the fitting-loss replacement. Training-only frozen offset remains the baseline MSE-derived offset procedure; pure TIMSE refers to the booster training objective, not removal of that retained baseline component.

References: leaderboard/scorer/competition.py prepare_context/score_competition provides the exact metric; research/e2000fs_calib_offset_roll_scap/experiment.py timse_coefficients provides the equivalent coefficient construction. XGBoost custom objective API: https://xgboost.readthedocs.io/en/release_2.1.0/tutorials/custom_metric_obj.html . Mean-one positive Hessian coefficients satisfy the diagonal custom objective interface. These references support implementation, not a performance claim.

Validation: external campaign test_timse.py checks direct percentile-tail integration, finite-difference derivatives, tied targets, invalid/constant targets, retained settings, and a tiny 24-output CPU XGBoost fit with exact checkpoint replay. Baseline diff is ~/sunrun-vanila-followups/baseline-mse-vs-timse.diff.

Example: CUDA_VISIBLE_DEVICES='' .venv/bin/python research/vanila_timse/experiment.py --experiment vanila_timse --device cpu --threads 6 --seeds 42 --out /home/t-lab01/sunrun-vanila-followups/runs/timse_s42

## Dispatch 2026-11-12

External campaign: `/home/t-lab01/sunrun-vanila-followups/`. Controller: `followup_scheduler.py`; assignment and scheduler-state record lifecycle. TIMSE seeds 42/43/44 use independent CPU tmux sessions (6 threads each). Fixed 50:50 ensemble automatically runs only after every TIMSE seed exits zero and passes artifact/replay checks. No leaderboard publication is scheduled. Results are archived here when complete. Local resource admission requires >9 GiB available memory per launch, max three jobs; golab2 unreachable, Colab has no active sessions. Official-objective reference audit and 14 passing synthetic tests are recorded in the external campaign.

## Execution result

Completed UTC: 2026-11-12T13:49:32.859149+00:00

| Scope | S | TIMSE | RMSE |
|---|---:|---:|---:|
| total | 102.060637 | 16638.113926 | 83.009978 |
| f1 | 122.999333 | 28134.638792 | 93.472353 |
| f2 | 78.839928 | 5995.519599 | 82.636027 |
| f3 | 69.620732 | 4937.396738 | 71.748669 |

Seed S mean=102.301545, SD=0.140947, worst=102.384454. Fixed mean3 is scored separately above.
Execution requested; publication later done on user request (see below), no baseline promotion. Failure/adoption diagnosis remains a separate review after all fixed experiments finish. Campaign: /home/t-lab01/sunrun-vanila-followups/.

## 리더보드 게시 (2026-11-13, 사용자 요청)

사용자 요청으로 `vanila_mse`·`vanila_timse`·`vanila_emsemble` 세 행을 보드에 함께 올렸다. 게시 자체는
성능 개선 판정이나 기준선 승격이 아니다(제출 게이트·SOTA 판정과 분리).

| 실험 (mean3) | S | TIMSE | RMSE | Bias | F1 S | F2 S | F3 S |
|---|---:|---:|---:|---:|---:|---:|---:|
| vanila_mse | 100.776 | 24932 | 65.88 | +4.55 | 122.11 | 81.02 | 63.37 |
| vanila_timse | 102.061 | 16638 | 83.01 | +42.53 | 123.00 | 78.84 | 69.62 |
| vanila_emsemble | 97.602 | 19651 | 69.71 | +23.54 | 119.24 | 75.31 | 61.89 |

Seed S (42/43/44): MSE 101.061/100.559/101.011, TIMSE 102.139/102.381/102.384,
emsemble 97.756/97.563/97.774. 앙상블은 세 seed와 세 기간 모두에서 두 단일 모델보다 낮다.

해석: 순수 TIMSE는 tail 오차를 줄이지만(TIMSE −33 %) 전체 예측을 위로 끌어올려(bias +42.5 km/s)
RMSE가 +26 % 나빠졌다. 고정 50:50 평균은 bias를 +23.5로 절반만 남기고 TIMSE 개선(−21 %)을 대부분 유지해
S = (MSE·TIMSE)^¼ 맞교환에서 이득(−3.15 % vs MSE)을 얻는다. 남은 +23.5 km/s bias가 다음 병목이다.
현재 보드 1위(S-cap 95.498) 대비 +2.2 %, `e2000fs_calib_offset_roll`(97.109) 대비 +0.5 %로 SOTA는 아니다.

dry-run 공식 scorer 점수는 저장된 mean3 점수와 1e-10 이내로 일치했다. 제출 meta·기록·라이브 행 검증은
캠페인 폴더 `~/sunrun-vanila-followups/{이름}-submission*.json`, `*-publication-verification.json`에 있다.

## 2026-11-13 레이아웃 변경 (중복 제거)

- 공용 코드를 `research/_module/`로 옮겼습니다. 이 행이 쓰는 모듈: `gbm`, `objectives`. `experiment.py`에는 특징 목록, 목적함수, 보정 설정을 담은 `gbm.GBMSpec`만 남았습니다.
- `results/`와 `logs/`는 `research/result/vanila_timse/`로 옮겼습니다. 위 본문에 나오는 `results/…`, `logs/…` 경로는 이 위치 기준입니다. `weights/`와 `visualization/`은 이 폴더에 그대로 있습니다.
- `--out`을 주지 않으면 `research/result/vanila_timse/`에 씁니다. `--out DIR`을 주면 예전과 같이 `DIR/results`, `DIR/logs`, `DIR/weights`에 씁니다(재현 스크립트 호환).
- 검증: 새 코드의 `--predict-only`(weights/) 예측이 기존 `results/`의 s42·s43·s44·mean3와 완전히 같고 S도 같습니다. old/new `--smoke` 예측과 checkpoint 배열(부스터 바이트·calibration)도 모두 같습니다.
