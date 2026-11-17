# vanila_gru

plain GRU — 5일(120h) 시간별 시퀀스 입력, 비가중 MSE, seed 42/43/44 고정 등가중 평균.
PyTorch 보조 모델 실험이며 GBM 누적 기준선과 구분한다. 공용 러너는 `research/_module/seq.py`, 입력은 `data-v2/input/df_ts.parquet`다.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/vanila_gru/experiment.py --device cuda --threads 2
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/vanila_gru/experiment.py --predict-only
CUDA_VISIBLE_DEVICES=0 .venv/bin/python research/vanila_gru/experiment.py --smoke --out /tmp/vanila_gru-smoke
```

## 전략 (2026-11-17, 실행 전 작성)

사용자 요청: 시계열로 쓸 수 있는 데이터는 5일 lookback 시계열로, 이미지는 한 장에서 경도를 stripe해 가며 뽑은 특징으로 넣고 손실은 MSE를 쓴다. 입력은 `df.parquet`가 아니라 새로 만든 `df_ts.parquet`다. 같은 입력으로 GRU·iTransformer·TSMixer 세 행을 만든다.

### 분류와 기준

- 분류: **replacement ablation (아키텍처 + 입력 표현)**. 누적 개선 실험이 아니다.
- 기준선 A: `vanila_mse` (mean3 S 100.775874). 같은 손실(비가중 MSE), 같은 prior/full 분리·frozen offset·clip, 같은 학습 cutoff·144h maturity·seed. 다른 점은 모델(XGBoost 24리드 블록 3개 → 신경망)과 입력 표현(요약 특징 → 시퀀스)이다.
- 별도 비교: 현재 최고 단일 모델 `e2000fs_calib_offset_roll_scap_arrival_cme` (mean3 S 93.864). SOTA 대비 2% 개선 여부는 따로 판정한다.

### 모델

2층 GRU(hidden 128, dropout 0.1)가 120스텝을 읽고, 마지막 hidden state를 `Linear(128, 72)`로 72리드 속도에 사상한다. `torch.nn.GRU`를 그대로 쓴다.

추가 의존성 없음(torch 2.7.1).

### 가설과 예상 메커니즘

순환 구조는 최근 5일의 속도·밀도·자기장 변화(상승/하강 국면)를 요약 통계(`_m6`, `_m24`, lag) 없이 직접 읽을 수 있다. 짧은 리드에서는 persistence 주변의 변화율 정보가, 긴 리드에서는 경도 스윕과 27일 재귀가 쓰일 것으로 기대한다.

개선은 가정하지 않는다. `vanila_mse`가 보인 것처럼 순수 MSE는 꼬리 가중이 없어 TIMSE에서 불리할 수 있고, 학습 origin이 약 3만 개(1h 간격, 강한 자기상관)라 신경망이 GBM보다 나쁠 가능성도 크다.

### 입력 (origin O마다 120스텝 × 38채널, 스텝 j = 칸 O−119+j)

| 묶음 | 채널 | 내용 |
|---|---:|---|
| ACE | 21 | 속도, 밀도, 온도, Bx/By/Bz/Bt, MAG lat/lon, EPAM e1–e2·p1–p5, SIS >10/>30 MeV, 위치 X/Y/Z. 밀도·온도·입자 flux는 log10. 인과적 앞값 채움(최대 119h) |
| 마스크 | 4 | SWEPAM/MAG/EPAM/SIS 관측 여부 |
| 재귀 | 3 | `speed(t + 72h − d일)`, d = 26/27/28. 창의 마지막 72스텝이 리드 1…72의 지난 자전 속도(항상 t보다 과거) |
| ENLIL | 6 | `next_arr_h`, `since_arr_h`, `n_pending`, `kp`, `glancing` + 지난 도착 마스크 |
| 이미지 | 4 | 칸 O 프레임 **한 장**의 경도 스윕: 스텝 j ↔ stripe k = 119−j(O보다 k시간 전에 중앙 자오선을 지난 경도, 폭 10°, 위도 ±30°)의 `logmed`·`dark`·`cover` + 프레임 마스크. 칸 O에 프레임이 없으면 24h 안의 최근 프레임(m시간 전)의 stripe k−m |

표준화 통계는 기간별 학습 구간(첫 lookback 스텝 ~ 마지막 학습 origin)에서만 계산해 checkpoint에 저장한다. 남은 NaN은 표준화 후 0이다.

### `vanila_mse` 대비 특징 diff

- **유지(시퀀스에 포함)**: 속도(`v_lag0/24/48/72`, `ace_speed_kms_last`), 밀도·온도·Bt·By·EPAM p3·SIS >10 MeV의 `_last`/`_m6`/`_m24`가 요약하던 원 시계열 120h, `rec26/27/28`, `enlil_n_pending`, `enlil_kp`.
- **대체**: SUVI 어두운 영역 면적 6개(`suvi_t035_*`, `suvi_t055_*`)와 속도별 strip 4개(`strip_v350/450/750_logmed`, `strip_v550_dark`) → 프레임 한 장의 경도 스윕 3통계. `rec_mean` → 재귀 3채널. `img_age_h` → stripe 중심 경도 보정과 프레임 마스크.
- **추가**: ACE 나머지 14개 변수, ENLIL `next_arr_h`/`since_arr_h`/`glancing`, 결측 마스크 6개.
- **학습 설정**: XGBoost 900 rounds → AdamW(lr 1e-4, weight decay 1e-2), batch 256, 최대 60 epoch, patience 8, gradient clip 1.0. 학습 라벨의 target time 앞값 채움, offset(최근 336행 평균 잔차 × 0.25), clip [1, 3000]은 동일.

### 프로토콜

- 내부 검증 없음. 기간별 `fit_F*` origin(origin + 144h ≤ cutoff: F1 2025-12-29, F2 2026-03-29, F3 2026-06-29)만 학습한다.
- prior 모델: 기준 구간(마지막 3개월 − 144h 이전)으로 학습하고 최근 3개월 보정 구간의 MSE로 epoch를 조기 종료해 E*를 정한다. 같은 구간의 prior 예측으로 리드별 frozen offset을 정한다. 모두 학습 데이터 안의 holdout이다.
- full 모델: 전체 학습 origin을 E* epoch 학습 → query 예측 → clip → offset → clip.
- 리더보드 기간 정답은 채점에만 쓴다. 하이퍼파라미터는 실행 전에 고정했고 리더보드 점수로 조정하지 않는다.
- 학습률 1e-4 (2026-11-17 사용자 지시로 고정, 본 실행 전). 근거는 학습 데이터 안의 prior holdout(F1 기준 구간 → 2025-09-23~12-23 보정 구간, seed 42, 리더보드 정답 미사용) 진단이다: 1e-3에서는 GRU의 holdout MSE가 1 epoch에서 최저(0.973)가 된 뒤 계속 올라 epoch 선택이 거칠었고, 1e-4에서는 6 epoch에서 최저(0.944)였다. 진단은 stripe 채널을 잡음으로 채운 임시 표에서 했으므로 수치는 참고용이다.
- seed 42/43/44를 모두 돌려 평균·표본 SD·최악을 보고하고, 보드 값은 세 seed 예측의 고정 등가중 평균이다.

### 검증 기준

- 후보 조건: 최고 단일 모델 대비 mean3 S 1% 이상 감소, 세 seed 모두 개선, 기간별 S·RMSE 악화 1% 이내.
- A(`vanila_mse`) 대비 S/TIMSE/RMSE/bias 차이를 Total·F1·F2·F3로 보고한다.
- 게시는 사용자 요청(2026-11-17)으로 게이트와 무관하게 진행하며 기준선 승격이 아니다.
- 완료 조건: 9개 seed×기간 학습 exit 0, 기간별 query origin 2089/2113/2113, scorer 대체 0건, 저장 weights 재생성 예측 일치.

### 참고 문헌

- Cho et al. 2014, *Learning Phrase Representations using RNN Encoder–Decoder* (arXiv 1406.1078): GRU 셀의 정의. 태양풍 문제에 대한 근거는 아님.
- Dhuri et al. 2025, *A Multimodal Encoder–Decoder Neural Network for Forecasting Solar Wind Speed at L1* (arXiv 2507.17298): 지난 자전의 태양풍 이력 인코더와 EUV 영상 인코더를 결합해 일평균 속도를 1–4일 예측(상관 0.78–0.63). 일평균·SDO 영상 조건이라 본 시간별 72리드·SUVI stripe 설정에 그대로 적용되지는 않는다.
- Upendran et al. 2020 (arXiv 2006.05825), Brown et al. 2022 (Space Weather): EUV 영상의 코로나홀 영역이 고속풍 예측에 가장 유효한 입력이라는 결과. 영상 전체가 아니라 경도 stripe 통계를 쓰는 본 실험의 효과를 보장하지 않는다.

## 결과 (2026-11-17)

실행: 캠페인 `~/sunrun-vanila-dl/`, seed별 독립 tmux 세션, 로컬 RTX A5000 2장(GPU 0·1), seed당 약 4분. 9개 seed×기간 학습 모두 exit 0, 파라미터 172,872개.

mean3(세 seed 예측의 고정 등가중 평균), 공식 scorer `lead-timse-6h-v1`:

| 구간 | S | TIMSE | RMSE | bias | A S | S 변화 | RMSE 변화 | TIMSE 변화 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Total | 117.064107 | 36131.20 | 73.230 | +1.72 | 100.776 | +16.2% | +11.2% | +44.9% |
| F1 | 140.620125 | 51769.74 | 88.683 | -8.58 | 122.113 | +15.2% | +7.8% | +46.2% |
| F2 | 96.830086 | 18641.21 | 71.188 | +3.36 | 81.017 | +19.5% | +18.5% | +44.1% |
| F3 | 69.096023 | 7899.64 | 56.989 | +10.50 | 63.374 | +9.0% | +8.9% | +19.8% |

A = `vanila_mse`. 개별 seed 점수 평균 118.159389, 표본 SD 0.734976, 최악 118.877986.

| seed | F1 E* (holdout MSE) | F2 E* (holdout MSE) | F3 E* (holdout MSE) | Total S |
|---|---|---|---|---:|
| 42 | 5 (0.929) | 3 (0.721) | 5 (0.433) | 118.877986 |
| 43 | 13 (0.822) | 3 (0.741) | 4 (0.441) | 117.409061 |
| 44 | 8 (0.851) | 3 (0.729) | 4 (0.432) | 118.191119 |

E*는 prior 단계에서 조기 종료로 정한 epoch 수, holdout MSE는 표준화 단위다.

| 리드 (h) | RMSE | A RMSE | naive RMSE | S | A S |
|---:|---:|---:|---:|---:|---:|
| 1 | 36.2 | 24.3 | 15.8 | 63.6 | 40.3 |
| 3 | 42.3 | 30.8 | 31.9 | 68.2 | 44.5 |
| 6 | 48.5 | 41.4 | 43.9 | 91.7 | 77.1 |
| 12 | 57.6 | 51.9 | 63.6 | 104.7 | 91.5 |
| 24 | 72.2 | 66.9 | 89.5 | 121.5 | 109.7 |
| 48 | 83.9 | 77.8 | 121.0 | 131.2 | 120.6 |
| 72 | 89.3 | 78.1 | 135.7 | 144.9 | 113.6 |

### 판정

- 게이트 실패. 최고 단일 모델 `e2000fs_calib_offset_roll_scap_arrival_cme`(93.864) 대비 S +24.7%, A 대비 +16.2%이고 세 seed 모두 A보다 나쁘다. 기준선에 반영하지 않는다.
- 같은 캠페인의 다른 두 행: `vanila_itransformer` 125.197, `vanila_tsmixer` 119.031.
- 게시는 사용자 요청(2026-11-17)에 따른 것이며 성능 개선 판정이나 기준선 승격이 아니다.

### 원인 분석

- 구현 오류는 아니다. 입력 텐서 검증(채널 원천 시각 ≤ origin, 재귀 채널 = `df.parquet`의 `rec26/27/28`, 이미지 스윕 인덱싱)과 저장 weights 재생성(9개 모두 최대 절대 오차 0), scorer 대체 0건을 확인했고 bias도 작다.
- 검증 조건도 A와 같다(같은 cutoff·maturity·query 격자·scorer).
- 가설/최적화 결함이다. **짧은 리드에서 마지막 관측 속도를 그대로 잇지 못한다.** h=1 RMSE가 36.2로 A(24.3)와 naive(15.8)보다 크고, 이 차이가 리드 전체에 남는다.
- 같은 약점이 리더보드 정답 없이도 보인다. F1 학습 구간 안의 prior holdout(2025-09-23~12-23, seed 42)에서 h1–6 표준화 MSE는 persistence 0.190인데 이 모델은 0.291이다.
- 메커니즘: 72리드를 한 번에 내는 MSE에서 조기 종료가 전체 holdout MSE(긴 리드 오차가 지배)로 정해져 E*가 2~23 epoch로 작다. 긴 리드가 과적합되기 시작하는 시점에 짧은 리드의 "입력 속도 복사"는 아직 수렴하지 않았다. 출력이 표준화된 절대 속도라서 persistence를 망이 직접 학습해야 하는 구조도 원인이다.
- 긴 리드에서도 A보다 RMSE가 크다(h=24 +5.3, h=48 +6.1, h=72 +11.1 km/s). 학습 origin 약 3만 개(1h 간격, 강한 자기상관)에서 시퀀스 모델이 요약 특징 GBM보다 표본 효율이 낮다.

### 다음 가설 (실행하지 않음, 제안)

출력에 persistence 잔차 연결을 추가한다(예측 = 마지막 관측 속도 + 망 출력, NLinear식 last-value 정규화). 근거는 위 학습 내부 holdout 진단이며, 구성 요소 하나만 바꾸는 통제 실험으로 이 행을 기준선으로 비교한다.
