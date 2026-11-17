# vanila_tsmixer

5일(120h) 시간별 시퀀스와 XGBoost 요약 특징을 함께 받는 TSMixer, 비가중 MSE, seed 42/43/44 고정 등가중 평균. PyTorch 보조 모델 실험이며 공용 러너는 `research/_module/seq.py`, 입력은 `data-v2/input/df_ts.parquet`와 `research/_module/summary.py`다.

2026-11-17 사용자 요청으로 보드 행을 이 버전으로 교체했다. 이전 버전(시퀀스 입력만, mean3 S 119.031)의 기록은 리더보드 저장소에서 지웠고(데이터셋 git 이력에는 남음), 코드·weights·결과·시각화는 `~/sunrun-vanila-dl-board/replaced_board_row/vanila_tsmixer/`에 보관했다. 이전 버전의 전략·진단은 아래 "이전 버전" 절에 그대로 둔다.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/vanila_tsmixer/experiment.py --device cuda --threads 2
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/vanila_tsmixer/experiment.py --predict-only
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/vanila_tsmixer/experiment.py --smoke --out /tmp/vanila_tsmixer-smoke
```

## 구성

- 입력 1: origin마다 120h × 38채널 시퀀스 — ACE 21개 변수와 결측 마스크 4개, 27일 재귀 3채널(72h 앞당김), ENLIL 5개 + 마스크, SUVI 프레임 한 장의 경도 스윕 3통계 + 프레임 마스크.
- 입력 2: `vanila_mse`의 31개 요약 특징 — origin별 상수 23개(`v_lag0/24/48/72`, ACE `_last`/`_m6`/`_m24` 11개, SUVI 어두운 영역 면적 6개, `img_age_h`, `enlil_n_pending`, `enlil_kp`), 리드별 8개 × 72리드(`rec26/27/28`, `rec_mean`, `strip_v350/450/750_logmed`, `strip_v550_dark`). `df_ts.parquet`에서 다시 계산하며 전 학습·query origin에서 XGBoost 입력과 같다(읽지 못한 프레임 4건의 `img_age_h`만 다름).
- 모델: 보조 정보를 받는 확장 TSMixer(Chen et al. 2023, arXiv 2303.06053; `torchtsmixer.TSMixerExt`, pytorch-tsmixer 0.2.0). 리드별 요약 특징 8개를 미래에 알려진 특징으로, 상수 23개를 정적 특징으로 넣는다. mixer block 2개, ff_dim 64, hidden 64, dropout 0.7, batch norm.
- 출력: 창의 마지막 관측 속도 + 망 출력(persistence 잔차).
- 학습: 비가중 MSE, AdamW lr 1e-4·weight decay 1e-2, batch 256, 최대 60 epoch, patience 8, gradient clip 1.0. 기간별 학습 cutoff(F1 2025-12-29, F2 2026-03-29, F3 2026-06-29), 144h maturity, prior/full 분리(prior가 학습 구간 마지막 3개월에서 조기 종료한 epoch 수로 full 학습), training-only frozen per-lead offset(×0.25, 336행), clip [1, 3000]. 표준화 통계는 기간별 학습 구간에서만 계산한다.
- 구성의 근거와 단계별 통제 실험: 잔차 출력과 정규화는 `vanila_tsmixer_res_reg` 행 README, 요약 특징 추가는 `~/sunrun-vanila-dl-summary/README.md`. 이 행은 그 캠페인의 `vanila_tsmixer_res_reg_sum` arm과 같은 구성이며, 같은 seed로 다시 학습한 예측·weights가 비트 단위로 같았다.

## 결과 (2026-11-17)

mean3, 공식 scorer `lead-timse-6h-v1`. 9개 seed×기간 학습 exit 0, 저장 weights로 predict-only 재생성 일치, scorer 대체 0건.

| 구간 | S | TIMSE | RMSE | bias | 이전 버전 S | vanila_mse S |
|---|---:|---:|---:|---:|---:|---:|
| Total | 106.496144 | 27979.00 | 69.370 | +6.35 | 119.031 | 100.776 |
| F1 | 126.316384 | 38590.38 | 84.127 | +2.35 | 127.789 | 122.113 |
| F2 | 87.968134 | 14949.84 | 65.891 | +6.82 | 115.011 | 81.017 |
| F3 | 67.268224 | 7317.05 | 55.689 | +9.92 | 77.682 | 63.374 |

seed별 Total S: 42 105.969244, 43 106.995564, 44 108.439667 (평균 107.135, 표본 SD 1.241, 최악 108.440).

리드 구간별 S (Total):

| | h1–6 | h7–24 | h25–48 | h49–72 |
|---|---:|---:|---:|---:|
| 이 행 | 60.6 | 96.8 | 112.8 | 118.9 |
| 이전 버전 | 101.2 | 112.3 | 121.0 | 126.6 |
| `vanila_mse` | 59.0 | 85.5 | 112.3 | 111.1 |

- 이전 버전 대비 -10.5%, `vanila_mse` 대비 +5.7%, 최고 단일 모델(93.864) 대비 +13.5%. 게이트 실패, 사용자 요청 게시이며 기준선 승격이 아니다.
- 요약 특징이 XGBoost와의 격차를 줄였지만 넘지는 못했다. 학습 데이터만 쓴 진단에서 데이터 크기·과대/과소적합·epoch 선택은 주원인이 아니었다(`~/sunrun-vanila-dl-summary/README.md`의 추가 진단 1·2, `~/sunrun-vanila-dl-epoch/README.md`). SWA와 리드 구간별 checkpoint는 ±1% 안으로 효과가 없어 넣지 않았다.

## 이전 버전 (2026-11-17 교체 전, 시퀀스 입력만)

아래는 교체 전 README 그대로다. 안의 경로 `research/result/vanila_tsmixer/…`는 이제 `~/sunrun-vanila-dl-board/replaced_board_row/vanila_tsmixer/result/…`에 있다.

TSMixer — 5일(120h) 시간별 시퀀스 입력, 비가중 MSE, seed 42/43/44 고정 등가중 평균.
PyTorch 보조 모델 실험이며 GBM 누적 기준선과 구분한다. 공용 러너는 `research/_module/seq.py`, 입력은 `data-v2/input/df_ts.parquet`다.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/vanila_tsmixer/experiment.py --device cuda --threads 2
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/vanila_tsmixer/experiment.py --predict-only
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/vanila_tsmixer/experiment.py --smoke --out /tmp/vanila_tsmixer-smoke
```

### 전략 (2026-11-17, 실행 전 작성)

사용자 요청: 시계열로 쓸 수 있는 데이터는 5일 lookback 시계열로, 이미지는 한 장에서 경도를 stripe해 가며 뽑은 특징으로 넣고 손실은 MSE를 쓴다. 입력은 `df.parquet`가 아니라 새로 만든 `df_ts.parquet`다. 같은 입력으로 GRU·iTransformer·TSMixer 세 행을 만든다.

#### 분류와 기준

- 분류: **replacement ablation (아키텍처 + 입력 표현)**. 누적 개선 실험이 아니다.
- 기준선 A: `vanila_mse` (mean3 S 100.775874). 같은 손실(비가중 MSE), 같은 prior/full 분리·frozen offset·clip, 같은 학습 cutoff·144h maturity·seed. 다른 점은 모델(XGBoost 24리드 블록 3개 → 신경망)과 입력 표현(요약 특징 → 시퀀스)이다.
- 별도 비교: 현재 최고 단일 모델 `e2000fs_calib_offset_roll_scap_arrival_cme` (mean3 S 93.864). SOTA 대비 2% 개선 여부는 따로 판정한다.

#### 모델

mixer block 2개(시간 축 MLP와 특징 축 MLP를 번갈아 적용, ff_dim 64, dropout 0.3, batch norm, 출력 1채널) 뒤에 시간 투영 `Linear(120, 72)`가 72리드 속도를 낸다. `pytorch-tsmixer` 0.2.0의 `torchtsmixer.TSMixer`를 그대로 쓴다.

`pytorch-tsmixer==0.2.0`(MIT, torch만 요구)을 `.venv`에 추가했다.

#### 가설과 예상 메커니즘

시간 축 선형 사상과 특징 축 MLP만으로도 persistence·추세·재귀를 표현할 수 있고, 파라미터가 가장 적어(약 4.4만) 3만 개 남짓한 자기상관 높은 학습 origin에서 과적합이 덜할 수 있다.

개선은 가정하지 않는다. `vanila_mse`가 보인 것처럼 순수 MSE는 꼬리 가중이 없어 TIMSE에서 불리할 수 있고, 학습 origin이 약 3만 개(1h 간격, 강한 자기상관)라 신경망이 GBM보다 나쁠 가능성도 크다.

#### 입력 (origin O마다 120스텝 × 38채널, 스텝 j = 칸 O−119+j)

| 묶음 | 채널 | 내용 |
|---|---:|---|
| ACE | 21 | 속도, 밀도, 온도, Bx/By/Bz/Bt, MAG lat/lon, EPAM e1–e2·p1–p5, SIS >10/>30 MeV, 위치 X/Y/Z. 밀도·온도·입자 flux는 log10. 인과적 앞값 채움(최대 119h) |
| 마스크 | 4 | SWEPAM/MAG/EPAM/SIS 관측 여부 |
| 재귀 | 3 | `speed(t + 72h − d일)`, d = 26/27/28. 창의 마지막 72스텝이 리드 1…72의 지난 자전 속도(항상 t보다 과거) |
| ENLIL | 6 | `next_arr_h`, `since_arr_h`, `n_pending`, `kp`, `glancing` + 지난 도착 마스크 |
| 이미지 | 4 | 칸 O 프레임 **한 장**의 경도 스윕: 스텝 j ↔ stripe k = 119−j(O보다 k시간 전에 중앙 자오선을 지난 경도, 폭 10°, 위도 ±30°)의 `logmed`·`dark`·`cover` + 프레임 마스크. 칸 O에 프레임이 없으면 24h 안의 최근 프레임(m시간 전)의 stripe k−m |

표준화 통계는 기간별 학습 구간(첫 lookback 스텝 ~ 마지막 학습 origin)에서만 계산해 checkpoint에 저장한다. 남은 NaN은 표준화 후 0이다.

#### `vanila_mse` 대비 특징 diff

- **유지(시퀀스에 포함)**: 속도(`v_lag0/24/48/72`, `ace_speed_kms_last`), 밀도·온도·Bt·By·EPAM p3·SIS >10 MeV의 `_last`/`_m6`/`_m24`가 요약하던 원 시계열 120h, `rec26/27/28`, `enlil_n_pending`, `enlil_kp`.
- **대체**: SUVI 어두운 영역 면적 6개(`suvi_t035_*`, `suvi_t055_*`)와 속도별 strip 4개(`strip_v350/450/750_logmed`, `strip_v550_dark`) → 프레임 한 장의 경도 스윕 3통계. `rec_mean` → 재귀 3채널. `img_age_h` → stripe 중심 경도 보정과 프레임 마스크.
- **추가**: ACE 나머지 14개 변수, ENLIL `next_arr_h`/`since_arr_h`/`glancing`, 결측 마스크 6개.
- **학습 설정**: XGBoost 900 rounds → AdamW(lr 1e-4, weight decay 1e-2), batch 256, 최대 60 epoch, patience 8, gradient clip 1.0. 학습 라벨의 target time 앞값 채움, offset(최근 336행 평균 잔차 × 0.25), clip [1, 3000]은 동일.

#### 프로토콜

- 내부 검증 없음. 기간별 `fit_F*` origin(origin + 144h ≤ cutoff: F1 2025-12-29, F2 2026-03-29, F3 2026-06-29)만 학습한다.
- prior 모델: 기준 구간(마지막 3개월 − 144h 이전)으로 학습하고 최근 3개월 보정 구간의 MSE로 epoch를 조기 종료해 E*를 정한다. 같은 구간의 prior 예측으로 리드별 frozen offset을 정한다. 모두 학습 데이터 안의 holdout이다.
- full 모델: 전체 학습 origin을 E* epoch 학습 → query 예측 → clip → offset → clip.
- 리더보드 기간 정답은 채점에만 쓴다. 하이퍼파라미터는 실행 전에 고정했고 리더보드 점수로 조정하지 않는다.
- 학습률 1e-4 (2026-11-17 사용자 지시로 고정, 본 실행 전). 근거는 학습 데이터 안의 prior holdout(F1 기준 구간 → 2025-09-23~12-23 보정 구간, seed 42, 리더보드 정답 미사용) 진단이다: 1e-3에서는 GRU의 holdout MSE가 1 epoch에서 최저(0.973)가 된 뒤 계속 올라 epoch 선택이 거칠었고, 1e-4에서는 6 epoch에서 최저(0.944)였다. 진단은 stripe 채널을 잡음으로 채운 임시 표에서 했으므로 수치는 참고용이다.
- seed 42/43/44를 모두 돌려 평균·표본 SD·최악을 보고하고, 보드 값은 세 seed 예측의 고정 등가중 평균이다.

#### 검증 기준

- 후보 조건: 최고 단일 모델 대비 mean3 S 1% 이상 감소, 세 seed 모두 개선, 기간별 S·RMSE 악화 1% 이내.
- A(`vanila_mse`) 대비 S/TIMSE/RMSE/bias 차이를 Total·F1·F2·F3로 보고한다.
- 게시는 사용자 요청(2026-11-17)으로 게이트와 무관하게 진행하며 기준선 승격이 아니다.
- 완료 조건: 9개 seed×기간 학습 exit 0, 기간별 query origin 2089/2113/2113, scorer 대체 0건, 저장 weights 재생성 예측 일치.

#### 참고 문헌

- Chen et al. 2023, *TSMixer: An All-MLP Architecture for Time Series Forecasting* (TMLR, arXiv 2303.06053): time-mixing과 feature-mixing MLP의 반복. 문제 조건은 장기 예측 벤치마크와 M5이며 태양풍 데이터에서의 효과는 문헌에서 확인되지 않았다.
- Dhuri et al. 2025, *A Multimodal Encoder–Decoder Neural Network for Forecasting Solar Wind Speed at L1* (arXiv 2507.17298): 지난 자전의 태양풍 이력 인코더와 EUV 영상 인코더를 결합해 일평균 속도를 1–4일 예측(상관 0.78–0.63). 일평균·SDO 영상 조건이라 본 시간별 72리드·SUVI stripe 설정에 그대로 적용되지는 않는다.
- Upendran et al. 2020 (arXiv 2006.05825), Brown et al. 2022 (Space Weather): EUV 영상의 코로나홀 영역이 고속풍 예측에 가장 유효한 입력이라는 결과. 영상 전체가 아니라 경도 stripe 통계를 쓰는 본 실험의 효과를 보장하지 않는다.

### 결과 (2026-11-17)

실행: 캠페인 `~/sunrun-vanila-dl/`, seed별 독립 tmux 세션, 로컬 RTX A5000 2장(GPU 0·1), seed당 약 4분. 9개 seed×기간 학습 모두 exit 0, 파라미터 62,314개.

mean3(세 seed 예측의 고정 등가중 평균), 공식 scorer `lead-timse-6h-v1`:

| 구간 | S | TIMSE | RMSE | bias | A S | S 변화 | RMSE 변화 | TIMSE 변화 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Total | 119.031326 | 31912.56 | 79.785 | +0.19 | 100.776 | +18.1% | +21.1% | +28.0% |
| F1 | 127.789186 | 33458.62 | 91.357 | -2.73 | 122.113 | +4.6% | +11.1% | -5.5% |
| F2 | 115.011384 | 26356.22 | 82.015 | -4.12 | 81.017 | +42.0% | +36.6% | +103.7% |
| F3 | 77.682328 | 9470.79 | 63.815 | +7.44 | 63.374 | +22.6% | +22.0% | +43.7% |

A = `vanila_mse`. 개별 seed 점수 평균 119.981284, 표본 SD 0.305917, 최악 120.334151.

| seed | F1 E* (holdout MSE) | F2 E* (holdout MSE) | F3 E* (holdout MSE) | Total S |
|---|---|---|---|---:|
| 42 | 16 (1.123) | 5 (0.799) | 15 (0.562) | 120.334151 |
| 43 | 23 (1.176) | 8 (0.824) | 11 (0.564) | 119.790754 |
| 44 | 15 (1.155) | 4 (0.753) | 13 (0.526) | 119.818946 |

E*는 prior 단계에서 조기 종료로 정한 epoch 수, holdout MSE는 표준화 단위다.

| 리드 (h) | RMSE | A RMSE | naive RMSE | S | A S |
|---:|---:|---:|---:|---:|---:|
| 1 | 68.5 | 24.3 | 15.8 | 93.2 | 40.3 |
| 3 | 72.8 | 30.8 | 31.9 | 103.7 | 44.5 |
| 6 | 72.3 | 41.4 | 43.9 | 105.3 | 77.1 |
| 12 | 73.7 | 51.9 | 63.6 | 111.5 | 91.5 |
| 24 | 78.1 | 66.9 | 89.5 | 123.0 | 109.7 |
| 48 | 84.4 | 77.8 | 121.0 | 122.8 | 120.6 |
| 72 | 91.2 | 78.1 | 135.7 | 134.6 | 113.6 |

#### 판정

- 게이트 실패. 최고 단일 모델 `e2000fs_calib_offset_roll_scap_arrival_cme`(93.864) 대비 S +26.8%, A 대비 +18.1%이고 세 seed 모두 A보다 나쁘다. 기준선에 반영하지 않는다.
- 같은 캠페인의 다른 두 행: `vanila_gru` 117.064, `vanila_itransformer` 125.197.
- 게시는 사용자 요청(2026-11-17)에 따른 것이며 성능 개선 판정이나 기준선 승격이 아니다.

#### 원인 분석

- 구현 오류는 아니다. 입력 텐서 검증(채널 원천 시각 ≤ origin, 재귀 채널 = `df.parquet`의 `rec26/27/28`, 이미지 스윕 인덱싱)과 저장 weights 재생성(9개 모두 최대 절대 오차 0), scorer 대체 0건을 확인했고 bias도 작다.
- 검증 조건도 A와 같다(같은 cutoff·maturity·query 격자·scorer).
- 가설/최적화 결함이다. **짧은 리드에서 마지막 관측 속도를 그대로 잇지 못한다.** h=1 RMSE가 68.5로 A(24.3)와 naive(15.8)보다 크고, 이 차이가 리드 전체에 남는다.
- 같은 약점이 리더보드 정답 없이도 보인다. F1 학습 구간 안의 prior holdout(2025-09-23~12-23, seed 42)에서 h1–6 표준화 MSE는 persistence 0.190인데 이 모델은 0.828이다.
- 메커니즘: 72리드를 한 번에 내는 MSE에서 조기 종료가 전체 holdout MSE(긴 리드 오차가 지배)로 정해져 E*가 2~23 epoch로 작다. 긴 리드가 과적합되기 시작하는 시점에 짧은 리드의 "입력 속도 복사"는 아직 수렴하지 않았다. 출력이 표준화된 절대 속도라서 persistence를 망이 직접 학습해야 하는 구조도 원인이다.
- 긴 리드에서도 A보다 RMSE가 크다(h=24 +11.2, h=48 +6.6, h=72 +13.1 km/s). 학습 origin 약 3만 개(1h 간격, 강한 자기상관)에서 시퀀스 모델이 요약 특징 GBM보다 표본 효율이 낮다.

#### 다음 가설 (실행하지 않음, 제안)

출력에 persistence 잔차 연결을 추가한다(예측 = 마지막 관측 속도 + 망 출력, NLinear식 last-value 정규화). 근거는 위 학습 내부 holdout 진단이며, 구성 요소 하나만 바꾸는 통제 실험으로 이 행을 기준선으로 비교한다.

### 과적합·과소적합 재확인 (2026-11-17, 사용자 요청)

학습 데이터 안의 prior 분할만 썼다(리더보드 정답 미사용). 본 실행과 같은 학습 루프를 조기 종료 없이 40 epoch 돌리며, epoch마다 eval 모드로 학습 표본(기준 구간에서 균등 추출 4,000 origin)과 holdout의 리드 구간별 MSE를 기록했다. 전체 곡선은 `research/result/vanila_tsmixer/logs/fit-diagnostic_F1_s42.json`, `…_F3_s42.json`에 있다.

**F1** (기준 구간 25,969 origin 학습, holdout 2025-09-23 ~ 2025-12-23, seed 42, 표준화 MSE)

| | h1–6 | h7–24 | h25–48 | h49–72 | 전체 |
|---|---:|---:|---:|---:|---:|
| 학습 표본: persistence | 0.099 | 0.398 | 0.872 | 1.261 | 0.819 |
| 학습 표본: 모델 @ E*=16 | 0.308 | 0.324 | 0.401 | 0.507 | 0.409 |
| 학습 표본: 모델 @ 40 epoch | 0.186 | 0.235 | 0.292 | 0.351 | 0.288 |
| holdout: persistence | 0.190 | 0.788 | 2.002 | 3.049 | 1.896 |
| holdout: 학습 평균 예측 | 2.106 | 2.138 | 2.191 | 2.231 | 2.184 |
| holdout: 모델 @ E*=16 | 0.828 | 0.832 | 1.087 | 1.451 | 1.123 |
| holdout: 모델 @ 40 epoch | 0.748 | 0.910 | 1.205 | 1.484 | 1.186 |
| holdout: 잔차 연결 probe @ 18 epoch | 0.171 | 0.554 | 1.027 | 1.298 | 0.928 |

**F3** (기준 구간 30,313 origin 학습, holdout 2026-03-23 ~ 2026-06-23, seed 42, 표준화 MSE)

| | h1–6 | h7–24 | h25–48 | h49–72 | 전체 |
|---|---:|---:|---:|---:|---:|
| 학습 표본: persistence | 0.112 | 0.448 | 0.999 | 1.434 | 0.932 |
| 학습 표본: 모델 @ E*=15 | 0.347 | 0.369 | 0.457 | 0.569 | 0.463 |
| 학습 표본: 모델 @ 40 epoch | 0.177 | 0.255 | 0.332 | 0.400 | 0.323 |
| holdout: persistence | 0.095 | 0.375 | 0.904 | 1.445 | 0.885 |
| holdout: 학습 평균 예측 | 0.770 | 0.748 | 0.723 | 0.744 | 0.740 |
| holdout: 모델 @ E*=15 | 0.440 | 0.487 | 0.563 | 0.647 | 0.562 |
| holdout: 모델 @ 40 epoch | 0.248 | 0.418 | 0.656 | 0.763 | 0.598 |
| holdout: 잔차 연결 probe @ 16 epoch | 0.089 | 0.296 | 0.557 | 0.681 | 0.494 |

결론: **두 가지가 리드 구간에 따라 동시에 있다.**

- **짧은 리드는 과소적합.** 조기 종료 시점(E*)의 h1–6 오차가 holdout뿐 아니라 **학습 표본에서도** persistence보다 크다. 학습을 계속하면 줄어들지만 E*에서는 수렴 전이다.
- **긴 리드는 과적합.** E* 이후 학습 표본의 h25–72 오차는 계속 내려가는데 holdout은 올라간다(40 epoch 행).
- 조기 종료 기준이 72리드 전체 MSE라서 E*는 긴 리드의 과적합이 시작되는 시점에 잡히고, 그때 짧은 리드는 아직 과소적합이다. epoch 수 하나로 두 구간을 함께 맞출 수 없다.
- 잔차 연결 probe(예측 = 마지막 관측 속도 + 망 출력, 나머지 동일)는 h1–6을 persistence 수준 이하로 내리고 긴 리드는 거의 바꾸지 않는다. 즉 짧은 리드 과소적합은 출력 구조로 해결되고, 긴 리드의 일반화 격차는 별도 문제로 남는다. probe 곡선은 같은 폴더의 `fit-diagnostic-residual-probe_F1_s42.json`, `…_F3_s42.json`에 있다. 이 probe는 진단이며 보드 행에는 반영하지 않았다.
