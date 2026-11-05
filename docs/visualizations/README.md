# 리더보드 예측 GIF

제출 예측을 월별 GIF로 확인할 수 있습니다. 2026-11-01 이후 폴더는 현재 공식 기간인 F1–F3(2026년 1~9월)을, 그 이전 폴더는 이전 F1+F2 분할(2026년 3~9월)을 담습니다. 최근 168시간 관측, 이후 72시간 실제값과 예측을 동일한 200~850 km/s Y축으로 비교하며, 시각화는 저장된 제출 예측을 읽기만 하고 학습이나 예측 변경을 수행하지 않습니다. 기존 입력 계약과 현재 연구 기준의 차이는 각 폴더의 설명에 기재했습니다.

게시 후 2주가 지난 폴더는 정리하여 제거합니다. 2026-10-31에 E1604·HSSPEAK72·E0472 폴더를 2주 경과 기준으로 제거했습니다.

## 2026-11-04 추가 — 리더보드 상위 10개 예측 시각화 (리더보드 visualization 열)

게시 시점(Space `68490c61e0`, 순위 행 41개)의 리더보드 상위 10개 제출입니다. 작업 중 새로 상위 10에 들어온 2개(Space `fc4f3c796e` 기준 순위, `*` 표시)도 추가했습니다. 순위는 각 시점 기준이며, 제출이 계속 들어오면서 바뀝니다. 각 폴더는 리더보드 dataset에 **저장된 제출 예측 자체**를 현재 채점기(`lead-timse-6h-v1`)로 다시 채점해, 기록과 1e-9 이내로 같을 때만 그렸습니다. 폴더마다 lead 곡선, 채점 origin 1,052개 전체의 9개월 예측 GIF, 점수 검증 manifest가 들어 있습니다. 리더보드의 **visualization** 열은 해당 제출(제출 시각 기준)의 폴더로 연결됩니다. 같은 이름으로 다시 제출하면 새 제출에는 링크가 붙지 않습니다.

- 생성 코드: [code/submission_viz.py](code/submission_viz.py). 모든 멤버의 제출을 같은 방식으로 다룹니다.
- 1위 폴더에는 TCN 학습 loss가 추가로 있습니다(아래 섹션).
- `e2000fs_calib_point4_144h`는 이름이 같은 2026-11-01 옛 df 폴더가 이미 있어서, 덮어쓰지 않고 하위 폴더 `fresh-df-20261103/`에 넣었습니다.
- 10개 모두 F3 TIMSE에 같은 6시간 톱니가 있습니다. 이 톱니는 00/06/12/18 UTC에 놓인 F3 최고속 관측 두 개 때문이며, 각 폴더 README에 진단값이 있습니다.

| 순위 | 실험 | 멤버 | Total S | F1 / F2 / F3 S | Total TIMSE | Total RMSE | 폴더 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | e2000fs_feat30_pointweight4_tcn_lb120 | seongeun | 100.00 | 107.32 / 96.61 / 67.63 | 16,565.4 | 78.57 | [e2000fs_feat30_pointweight4_tcn_lb120](e2000fs_feat30_pointweight4_tcn_lb120/) |
| 2 | e2000fs_calib11g_serarounds2 | seongeun | 100.23 | 122.25 / 76.37 / 61.12 | 22,768.7 | 68.19 | [e2000fs_calib11g_serarounds2](e2000fs_calib11g_serarounds2/) |
| 3 | e2000fs_calib3_era | seongeun | 100.43 | 124.13 / 74.66 / 59.80 | 23,836.6 | 67.17 | [e2000fs_calib3_era](e2000fs_calib3_era/) |
| 4 | dirmo_24_feat30_pw4_d12_rawdf | jsy301 | 100.53 | 125.70 / 73.45 / 60.02 | 24,252.8 | 66.68 | [dirmo_24_feat30_pw4_d12_rawdf](dirmo_24_feat30_pw4_d12_rawdf/) |
| 5 | e2000fs_feat30_pointweight4_tcn_lb120_regskip | seongeun | 100.56 | 112.58 / 92.11 / 67.88 | 19,415.6 | 73.77 | [e2000fs_feat30_pointweight4_tcn_lb120_regskip](e2000fs_feat30_pointweight4_tcn_lb120_regskip/) |
| 6 | e2000fs_calib_point4_144h | seongeun | 100.62 | 125.43 / 73.75 / 59.73 | 23,906.9 | 67.22 | [e2000fs_calib_point4_144h/fresh-df-20261103](e2000fs_calib_point4_144h/fresh-df-20261103/) |
| 7 | dirmo_24_feat30_metricfloor4 | jsy301 | 100.80 | 125.51 / 72.80 / 60.55 | 24,059.5 | 67.22 | [dirmo_24_feat30_metricfloor4](dirmo_24_feat30_metricfloor4/) |
| 8 | e2000fs_calib11d_sera | seongeun | 100.87 | 121.38 / 75.82 / 64.61 | 22,279.8 | 69.83 | [e2000fs_calib11d_sera](e2000fs_calib11d_sera/) |
| 9 | e2000fs_calib11h_noenlil | seongeun | 100.97 | 121.83 / 77.10 / 64.61 | 22,670.9 | 69.62 | [e2000fs_calib11h_noenlil](e2000fs_calib11h_noenlil/) |
| 10 | e2000fs_calib11f_serarounds | seongeun | 101.23 | 122.25 / 76.37 / 63.65 | 22,746.9 | 69.60 | [e2000fs_calib11f_serarounds](e2000fs_calib11f_serarounds/) |
| 2* | e2000fs_calib2_mixture2comp | seongeun | 100.18 | 123.61 / 73.79 / 60.66 | 22,620.5 | 68.51 | [e2000fs_calib2_mixture2comp](e2000fs_calib2_mixture2comp/) |
| 8* | e2000fs_budget_complexity | seongeun | 100.78 | 125.12 / 74.74 / 59.96 | 23,919.4 | 67.52 | [e2000fs_budget_complexity](e2000fs_budget_complexity/) |

## 2026-11-04 추가 — 새 df 재실행 TCN (현재 채점 지표)

- [e2000fs_feat30_pointweight4_tcn_lb120](e2000fs_feat30_pointweight4_tcn_lb120/) — 게시 기준 리더보드 **1위 / 41**,
  Total S **100.00**(F1 107.32 / F2 96.61 / F3 67.63), TIMSE 16,565.4, RMSE 78.57. 새 raw-only df + causal ffill 재실행으로
  제출된 120시간 입력 TCN입니다.
  - 담은 것: 세 기간의 선택 학습·재학습 loss, lead 곡선, 채점 origin 1,052개 전체의 9개월 예측 GIF.
  - 세 기간 모두 validation loss가 1 epoch에서 가장 낮아, 제출 모델은 1 epoch 학습입니다.
  - F3 TIMSE의 6시간 톱니는 00/06/12/18 UTC에 놓인 F3 최고속 관측 두 개(743.4 · 690.0 km/s) 때문임을 확인했습니다.
- 이 폴더는 **현재 lead-first 지표(`lead-timse-6h-v1`, S/TIMSE/RMSE)와 ACE status 0 무보정 정답**으로 다시 채점·검증했습니다.
  아래 이전 폴더의 MSE와는 직접 비교하지 마십시오.

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
  folds에서 비교), 잔차 분포(전체·본체·꼬리 3패널 막대그래프), 9개월 예측 GIF를 담았습니다.

## 2026-10-31 추가 (B) — 추론 손풀이 앱

제출 모델 두 개의 추론을 **원자료부터 채점까지** 한 단계씩 따라가는 앱 형태(bbycroft.net/llm 과 같은
레이아웃: 좌 sidebar 3 : 우 캔버스 5, `#f3f3f3`, 단계 칩 `#8587f7`) 자기완결 HTML 2편을
[`docs/explainers/`](../explainers/)에 추가했습니다.

- [e2000fs_calib4_detpool_gpd](../explainers/e2000fs_calib4_detpool_gpd.html) — 실제 SUVI 이미지 1장 →
  패치·코로나홀 13개 → 30개 피처 → 입력 행 237열 → detroute·gpd 트리 모델 → `(1−w)·detroute + w·gpd` →
  72시간 예측. 한 시점의 실제 계산값을 스크롤형으로 보여 줍니다.
- [e2000fs_calib2_mixture2comp](../explainers/e2000fs_calib2_mixture2comp.html) — 원자료 → 정답 72h →
  입력 237열 → level → 확률 → gap → 합성 → 72개 출력 → 채점 (9단계). 칸을 클릭하면
  `level + p·(gap−level)` 계산이 뜹니다.

## 2026-10-31 추가

[E2000FS Calib2 DetRoute의 detector 학습 loss, 라우팅 진단, permutation 중요도, F1+F2 월별 예측 GIF](e2000fs_calib2_detroute/)를 추가했습니다. 게시 기준 리더보드 **1위**, MSE **3213.191868**(F1 3551.74 / F2 2957.82, ≥550 7343.43)입니다. 두 archive arm(tweedie15 몸통, mixture2comp 꼬리) 사이를 `binary:logistic` detector가 (origin, horizon)마다 확률로 라우팅한 제출이며, τ=0.234는 채점 전에 비용비 1233/(1233+4045)로 고정했습니다. 몸통 arm 대비 전체 −14.40 / ≥550 −2082.75의 Pareto 개선이지만 F1 전체 MSE는 +39.03 나빠졌고, 사전 목표(전체 <3200, 모든 ≥550 <6000)는 달성하지 못했습니다. loss는 저장 booster를 비트 단위로 재현한 재학습으로 기록했고, 중요도는 두 arm 예측을 고정한 채 라우팅만 다시 실행해 계산했습니다.

## 2026-10-30 추가

[E2000FS Budget-Spectral의 학습 loss, permutation importance, F1+F2 월별 예측 GIF](e2000fs_budget_spectral/)를 추가했습니다. point4 weighted L2에 24시간 잔차 FFT의 radial pseudo-Huber 항(λ 0.10, scale 50 km/s)을 더한 spectral loss를 24-output vector tree 3개로 학습한 제출 모델이며, 공식 MSE **3495.593477**(F1 4028.97 / F2 3093.25, ≥550 5727.17)입니다. 같은 행·같은 cutoff로 학습한 paired control(3494.210802)과의 loss 비교, 개발 창(D1/D2/D3/C) 곡선, 제출 예측 4,970개 시작점 전체의 월별 GIF를 담았습니다. XGBoost 2.1.4가 multi-target 트리를 덤프하지 못해 특징 중요도는 permutation(ΔMSE)으로 계산했고, 저장 booster가 제출 예측을 비트 단위로 재현하는지 먼저 확인했습니다.

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


## 2026-11-04 추가 — Offset Roll · Budget Spectral · DetRoute 최신 제출

세 실험의 현재 제출 예측을 공식 채점기(`lead-timse-6h-v1`)로 재계산해 Total/F1/F2/F3의 S·TIMSE·RMSE·tRMSE·bias가 기록과 1e-9 이내로 같은지 확인한 뒤 시각화했습니다. 각 폴더는 lead 곡선, 2026년 1~9월 모든 채점 origin의 예측 GIF 9개, 재현·검증 manifest를 담습니다. Spectral과 DetRoute의 과거 시각화는 그대로 보존했습니다.

| 실험 | 제출 시각 UTC | Total S | 시각화 |
| --- | --- | ---: | --- |
| e2000fs_calib_offset_roll | 20261104T034835263Z | 98.76 | [시각화](e2000fs_calib_offset_roll/) |
| e2000fs_budget_spectral | 20261104T035518124Z | 99.75 | [시각화](e2000fs_budget_spectral/fresh-df-20261103/) |
| e2000fs_calib2_detroute | 20261104T040143309Z | 100.26 | [시각화](e2000fs_calib2_detroute/fresh-df-20261103/) |

## 2026-11-05 추가 — Offset Roll q10/q50/q90 분위수 (미제출 ablation)

`e2000fs_calib_offset_roll`에서 calibration을 빼고, smoothed pinball로 q10/q50/q90을 함께 학습해 q50을 점예측으로 쓴 replacement ablation입니다(seed 42/43/44). **리더보드에 제출하지 않았으므로** board의 visualization 열에는 연결되지 않습니다. 로컬 run의 q50 예측을 공식 채점기로 다시 채점해 저장 기록과 1e-9 이내로 같은 것을 확인한 뒤 그렸습니다. GIF는 seed 42이고, 각 프레임에 q10–q90 구간과 대조군 점예측을 함께 그렸습니다.

| 실험 | Total S (s42 / s43 / s44) | 대조군 A − offset | ΔS 3-seed 평균 (95 % CI) | [q10, q90] coverage | 시각화 |
| --- | ---: | ---: | ---: | ---: | --- |
| q50_pinball | 105.58 / 105.44 / 105.10 | 99.75 / 100.91 / 101.31 | +4.72 (+0.97, +8.55) | 0.876 | [시각화](offset_roll_quantile_q50_pinball/) |

RMSE와 MAE는 좋아졌지만, median readout과 꼬리 가중 제거 때문에 고속풍 꼬리 오차(TRMSE +16 ~ +21)가 커져 S가 나빠졌습니다. 사전 기준으로 "점 모형이 낫다"입니다.
