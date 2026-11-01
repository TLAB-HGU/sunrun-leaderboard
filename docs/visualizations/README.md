# 리더보드 예측 GIF

제출 예측을 월별 GIF로 확인할 수 있습니다. 2026-11-01 이후 폴더는 현재 공식 기간인 F1–F3(2026년 1~9월)을, 그 이전 폴더는 이전 F1+F2 분할(2026년 3~9월)을 담습니다. 최근 168시간 관측, 이후 72시간 실제값과 예측을 동일한 200~850 km/s Y축으로 비교하며, 시각화는 저장된 제출 예측을 읽기만 하고 학습이나 예측 변경을 수행하지 않습니다. 기존 입력 계약과 현재 연구 기준의 차이는 각 폴더의 설명에 기재했습니다.

게시 후 2주가 지난 폴더는 정리하여 제거합니다. 2026-10-31에 E1604·HSSPEAK72·E0472 폴더를 2주 경과 기준으로 제거했습니다.

## 2026-11-01 추가 — F1/F2/F3 재적합 lane 2편

2026-10-31 재분할(F1 = 2026-01~03, F2 = 04~06, F3 = 07~09, 6,315 origins, 454,680 pairs) 기준으로 제출된 두 실험을
추가했습니다. 이 두 폴더의 GIF는 2026년 1~9월 9개월이고, 아래 표의 기존 폴더는 재분할 전 F1+F2 창(4,970 origins)의
기록이므로 점수를 직접 비교하지 마십시오.

- [e2000fs_calib5_detpool_all](e2000fs_calib5_detpool_all/) — 게시 기준 리더보드 **1위**, MSE **4154.152633**
  (F1 6007.06 / F2 3600.17 / F3 2876.28, ≥550 **9641.59**). `e2000fs_calib4_detpool_gpd` 위에 아카이브된 refit arm
  16개를 얹은 선형 스택이며, arm별 평균 β(gpd 0.193 · detroute 0.148 · lgbm_recent 0.114 …), 168-origin 주기
  재적합 37창 중 **15창이 β = 0**(base 단독)으로 끝난 궤적, 원래 81-arm lane의 가중치 분포, 9개월 예측 GIF를 담았습니다.
- [e2000fs_calib_point4_144h](e2000fs_calib_point4_144h/) — 게시 기준 10위, MSE **4452.152081**(F1 6426.86 /
  F2 3896.40 / F3 3055.62, ≥550 **8597.82**). tweedie15 lane의 paired control arm으로, 900 round × 9 booster의
  학습·held-out loss(재학습이 archive 예측을 **비트 단위로 재현**), 목표 속도 밴드별 MSE(tweedie15·detpool_gpd와 같은
  folds에서 비교), 9개월 예측 GIF를 담았습니다.

## 2026-10-31 추가

[E2000FS Calib2 DetRoute의 detector 학습 loss, 라우팅 진단, permutation 중요도, F1+F2 월별 예측 GIF](e2000fs_calib2_detroute/)를 추가했습니다. 게시 기준 리더보드 **1위**, MSE **3213.191868**(F1 3551.74 / F2 2957.82, ≥550 7343.43)입니다. 두 archive arm(tweedie15 몸통, mixture2comp 꼬리) 사이를 `binary:logistic` detector가 (origin, horizon)마다 확률로 라우팅한 제출이며, τ=0.234는 채점 전에 비용비 1233/(1233+4045)로 고정했습니다. 몸통 arm 대비 전체 −14.40 / ≥550 −2082.75의 Pareto 개선이지만 F1 전체 MSE는 +39.03 나빠졌고, 사전 목표(전체 <3200, 모든 ≥550 <6000)는 달성하지 못했습니다. loss는 저장 booster를 비트 단위로 재현한 재학습으로 기록했고, 중요도는 두 arm 예측을 고정한 채 라우팅만 다시 실행해 계산했습니다.

## 2026-10-28 추가

[E2000FS Feat30 Extreme의 train·validation loss, feature importance, F1+F2 월별 예측 GIF](e2000fs_feat30_extreme/)를 추가했습니다. 게시 기준 리더보드 14위, MSE **3585.635750**(F1 4025.01 / F2 3254.20)이며 실제 제출 예측과 저장 booster를 검증해 사용했습니다. 30개 특징은 공식 F1/F2 permutation importance를 본 뒤 사후 선택한 집합이므로 fresh holdout 결과가 아니라는 점을 명시했습니다.

## 2026-10-27 추가

[E2000FS DirMO-24의 train·validation loss와 F1+F2 월별 예측 GIF](e2000fs_dirmo_24/)를 추가했습니다. 게시 기준 리더보드 1위, MSE **3480.270151**(F1 3878.88 / F2 3179.59)이며 실제 제출한 단일 seed 42 예측을 사용합니다. GIF는 2026-03~09 7개월, 4,970개 시작점 전체입니다. loss는 제출 F1·F2 모델의 학습 중 기록이며, validation 구간이 학습 구간 안에 있으므로 진단용입니다.

같은 모델의 [feature importance](e2000fs_dirmo_24/#feature-importance)(permutation + split 횟수)도 추가했습니다. 제출 모델을 다시 학습해 제출 예측과 완전히 같은 것을 확인한 뒤 계산했습니다.

### e2000fs 전체 시각화

E2000FS 전략 비교(XGBoost)와 Kumo 비교의 나머지 리더보드 기록도 같은 형식으로 추가했습니다. XGBoost 기록은 train·validation loss, feature importance, F1+F2 월별 GIF를, Kumo 기록은 학습 round와 tree가 없어 GIF만 담았습니다. 순위는 각 폴더를 게시한 시점 기준입니다.

| 게시 시점 순위 | 실험 | 공식 MSE (F1 + F2) | 폴더 |
| ---: | --- | ---: | --- |
| 1 | E2000FS Calib2 DetRoute | 3213.191868 | [e2000fs_calib2_detroute](e2000fs_calib2_detroute/) |
| 3 | E2000FS DirRecMO-12 | 3534.223345 | [e2000fs_dirrecmo_12](e2000fs_dirrecmo_12/) |
| 4 | E2000FS DirMO-12 | 3553.341592 | [e2000fs_dirmo_12](e2000fs_dirmo_12/) |
| 5 | E2000FS MIMO | 3553.553891 | [e2000fs_mimo](e2000fs_mimo/) |
| 6 | E2000FS DirMO-6 | 3568.334464 | [e2000fs_dirmo_6](e2000fs_dirmo_6/) |
| 7 | E2000FS DirRecMO-24 | 3580.432193 | [e2000fs_dirrecmo_24](e2000fs_dirrecmo_24/) |
| 8 | E2000FS RecMO-12 | 3581.893558 | [e2000fs_recmo_12](e2000fs_recmo_12/) |
| 9 | E2000FS RectifyMO-24 | 3596.842904 | [e2000fs_rectifymo_24](e2000fs_rectifymo_24/) |
| 10 | E2000FS DirRecMO-6 | 3606.352259 | [e2000fs_dirrecmo_6](e2000fs_dirrecmo_6/) |
| 11 | E2000FS RecMO-24 | 3608.390614 | [e2000fs_recmo_24](e2000fs_recmo_24/) |
| 12 | E2000FS Rectify | 3627.786587 | [e2000fs_rectify](e2000fs_rectify/) |
| 13 | E2000FS RecMO-6 | 3652.003202 | [e2000fs_recmo_6](e2000fs_recmo_6/) |
| 15 | E2000FS Stratify-DirRecMO-DirRecMO-24 | 3672.664438 | [e2000fs_stratify_dirrecmo_dirrecmo_24](e2000fs_stratify_dirrecmo_dirrecmo_24/) |
| 18 | E2000 Direct-multioutput | 3818.316284 | [e2000_direct_s42](e2000_direct_s42/) |
| 18 | E2000FS Kumo DirRec | 4259.032058 | [e2000fs_kumo_dirrec](e2000fs_kumo_dirrec/) |
| 19 | E2000 Direct-pooled | 4266.967935 | [e2000_horizon_conditioned_s42](e2000_horizon_conditioned_s42/) |
| 21 | E2000FS DirRec | 4347.016897 | [e2000fs_dirrec](e2000fs_dirrec/) |
| 22 | E2000FS Kumo Direct | 4391.086418 | [e2000fs_kumo_direct](e2000fs_kumo_direct/) |
| 22 | E2000FS Recursive | 6043.244806 | [e2000fs_recursive](e2000fs_recursive/) |
| 31 | E2000FS Kumo pooled | 13844.759176 | [e2000fs_kumo_pooled](e2000fs_kumo_pooled/) |
| 27 | E2000FS Kumo Recursive | 18306.617946 | [e2000fs_kumo_recursive](e2000fs_kumo_recursive/) |

[e2000fs_feat30_pointweight4](e2000fs_feat30_pointweight4/): MSE3732.269, >=550 MSE6071.056; pointwise weighting follow-up, loss/importance and daily-sampled monthly GIFs. High-speed improvement accompanied by ordinary-region degradation.
