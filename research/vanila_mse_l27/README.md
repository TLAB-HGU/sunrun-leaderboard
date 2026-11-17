# vanila_mse_l27

2026-11-17 사용자 요청: `vanila_mse`와 같은 학습 구조에서 입력을 **현재 타겟 값(`v_lag0`)**과
**27일 전 값(`rec27`)** 두 개로만 줄여 학습한다.

## 전략 (실행 전 작성)

- 분류: replacement ablation(피처 제거). 누적 개선 실험이 아니며 기준선을 승격하지 않는다.
- 가설: 순수 MSE XGBoost의 성능 중 지속성(현재 속도)과 27일 재귀(태양 자전 주기 재현성) 신호만으로 설명되는
  몫을 측정한다. 사전 예상은 A보다 나쁨(S 증가)이며, 개선 효과는 가정하지 않는다.
- A(비교 기준): `vanila_mse` mean3 S 100.775874 (F1 122.112963 / F2 81.017198 / F3 63.373792),
  seed 평균 100.877281, SD 0.276684, 최악 101.061462. 31 source / 215 expanded 특징.
- B(이 실험): 입력만 `constant=['v_lag0']`, `varying=['rec27']`로 교체. 블록당 입력 25열
  (`v_lag0`, `rec27_h{start+1..start+24}`), parent_constant 없음.
  - `v_lag0`: origin 칸 O의 SWEPAM 속도(origin마다 상수).
  - `rec27`: target_time − 27 d의 속도(lead마다 다름, h ≤ 72에서 항상 origin 이전이라 인과적).
- 제거 특징: `ace_speed_kms_last`, `ace_density_cm3_last`, `ace_density_cm3_m6`, `ace_temperature_k_last`,
  `ace_temperature_k_m24`, `ace_bt_nt_last`, `ace_bt_nt_m6`, `ace_by_gsm_nt_m24`, `ace_epam_p_channel3_last`,
  `ace_sis_gt10_flux_last`, `suvi_t035_top2_area`, `suvi_t035_meridian15_equatorial_area`,
  `suvi_t035_meridian30_equatorial_area`, `suvi_t055_meridian15_area`, `suvi_t055_meridian30_equatorial_area`,
  `suvi_t055_equatorial_area`, `img_age_h`, `enlil_n_pending`, `enlil_kp`, `v_lag24`, `v_lag48`, `v_lag72`,
  `rec26`, `rec28`, `rec_mean`, `strip_v350_logmed`, `strip_v450_logmed`, `strip_v550_dark`, `strip_v750_logmed`.
- 유지(변경 없음): 목적함수 gradient=prediction−target, Hessian=1; 3×24-lead multi_output_tree 블록;
  900 rounds, depth 3, lr .03, subsample .8, colsample .7, min_child_weight 20, lambda 5; seed 42/43/44;
  cutoff F1 2025-12-29 / F2 2026-03-29 / F3 2026-06-29; 144h maturity; prior/full 분리;
  training-only frozen 336-row per-lead offset ×0.25; clip [1, 3000]; 특징·학습 라벨 인과 ffill.
- 평가: 내부 검증 없음. F1/F2/F3 리더보드 기간만 공식 `competition.py`로 채점. 정답은 적합에 쓰지 않는다.
  seed 3개 고정 등가중 mean3, seed별 S 평균/SD/최악과 기간별 S/TIMSE/RMSE/MAE를 A와 함께 보고한다.
- 완료 조건: 3 seed exit 0, F1/F2/F3 metadata, booster replay 오차 0, `--predict-only` 재생 예측이
  seed run 예측과 완전 일치, `n_replaced` 0. 게시는 사용자 결정 시에만.
- 실행: 로컬 CPU(`CUDA_VISIBLE_DEVICES=''`), seed별 tmux 세션 `vml27-s{seed}` 3 threads 병렬,
  캠페인 `~/sunrun-vanila-mse-l27/`.
- smoke(2 rounds) 확인: exit 0, `expanded_feature_count` 25, 블록 0 특징 `['v_lag0', 'rec27_h1', …, 'rec27_h24']`.
  `v_lag0` 결측 906840 → ffill 후 72, `rec27` 908340 → 88 (선행 NaN 유지, vanila_mse와 같은 정책).

## diff (vanila_mse/experiment.py 대비)

- `EXPERIMENT = 'vanila_mse_l27'`
- `PARENT_CONSTANT`/`ADDED_CONSTANT`/`VARYING`(8개) → `CONSTANT = ['v_lag0']`, `VARYING = ['rec27']`
- `GBMSpec(constant=CONSTANT, varying=VARYING)`, `parent_constant` 미지정. 목적함수·나머지 설정 동일.

## 결과 (2026-11-17, 실행 후)

실행 상태: 3 seed 모두 exit 0, 기간별 metadata 6개 블록 × 3기간 정상, booster replay 오차 0.
병합한 weights로 `--predict-only`를 다시 돌린 예측이 seed별 run 예측과 완전히 일치했고, `n_replaced`는 0이다.
git 210883179, xgboost 2.1.4. 산출물은 `~/sunrun-vanila-mse-l27/`에 있고 리더보드에는 게시하지 않았다.

| 구간 | B S (l27) | A S (vanila_mse) | ΔS | B TIMSE | A TIMSE | B RMSE | A RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Total | 124.561 | 100.776 | +23.6 % | 41278 | 24932 | 79.124 | 65.881 |
| F1 | 146.183 | 122.113 | +19.7 % | 56520 | 35421 | 96.141 | 82.264 |
| F2 | 100.651 | 81.017 | +24.2 % | 21400 | 12936 | 72.554 | 60.054 |
| F3 | 78.714 | 63.374 | +24.2 % | 9716 | 6592 | 65.852 | 52.325 |

- seed S: s42 124.700 / s43 124.400 / s44 124.678. 평균 124.593, SD 0.167, 최악 124.700
  (A: 100.877 / 0.277 / 101.061). 같은 seed끼리 비교해도 세 seed 모두 A보다 나쁘다.
- 참고 기준선 Total S: naive 149.444, seasonal_naive_648 142.480, mean_reversion 134.733.
  두 입력만 쓴 GBM은 이 세 기준선보다 낫지만 vanila_mse와의 격차가 크다.
- lead별 S (B / A): h1 32.8 / 40.3, h6 89.9 / 77.1, h12 118.3 / 91.5, h24 142.5 / 109.7,
  h48 141.9 / 120.6, h72 126.8 / 113.6. h1은 B가 더 좋다. 짧은 lead에서는 v_lag0 지속성만으로 충분하고,
  나머지 특징은 h1에서 오히려 분산을 늘리는 것으로 보인다(가설이며 따로 검증하지 않았다).
  h ≥ 6에서는 제거한 ACE·SUVI·ENLIL·추가 lag 특징이 S를 크게 낮춘다.
- 판정: 사전 예상대로 B가 A보다 나쁘다(replacement ablation, 기준선 승격 없음). 지속성과 27일 재귀만으로는
  vanila_mse S의 약 81 %(100.8/124.6) 수준이며, 남은 차이는 제거된 특징의 기여다.

## 게시 (2026-11-17, 사용자 요청)

사용자 요청으로 리더보드에 게시했다. 게이트(기존 최고 대비 개선)와는 무관한 replacement ablation이며 기준선을 승격하지 않는다.

- 제출 `20261117T121636072Z`, 공개 보드 22위 / 29 (Space `71177e17cc`), Total S 124.561.
- 아키텍처 링크: GitHub `bf53a953b` (`research/vanila_mse_l27/experiment.py`), replay 래퍼 `scripts/seongeun/vanila_mse_l27/`.
- 산출물: weights는 `research/vanila_mse_l27/weights/F{n}/s{seed}.npz`(로컬 전용), results·logs는 `research/result/vanila_mse_l27/`.
  보드 모드 `--predict-only` 예측이 학습 run 예측과 s42/s43/s44/mean3 모두 완전히 일치했다.
  submit dry-run의 S와 실제 제출 S가 같고, `visualization/`은 저장된 제출을 다시 채점해 1e-9 이내로 일치할 때만 그렸다.
