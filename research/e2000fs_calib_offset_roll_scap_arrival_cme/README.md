# e2000fs_calib_offset_roll_scap_arrival_cme

- 작성자: `seongeun`
- 리더보드: 2026-11-15 사용자 요청으로 게시 (3 seed 등가중 평균, mean3)
- 기준선: `e2000fs_calib_offset_roll_scap` (S-cap, mean3 S 95.498, 설계 열 215개)
- 출처: 사건 전조 특징 ablation의 N2 구성 `precursor_tk` (`~/sunrun-event-precursor-ablation/`, 사전 계획은 `.omx/plans/event-precursor-feature-ablation-20261114.md`)

S-cap 레시피(A900 point4 + spectral, 이어서 학습 MSE 1.01배 제약 아래 direct-S continuation 180 tree)는 **바꾸지 않았습니다.** 215개 설계 열 뒤에 origin 상수 8개만 붙였습니다. 앞 215개 열은 이름·순서·값이 기준선과 같습니다.

| 그룹 | 특징 | 뜻 |
|---|---|---|
| T (도착 시각) | `pub_enlil_next_arr_h` | 아직 유효한 다음 지구 도달 예보까지 남은 시간 (h) |
| | `pub_enlil_since_arr_h` | 6–96 h 전에 지난 가장 최근 도달 예보 이후 경과 시간 (h) |
| | `pub_enlil_glancing` | 다음 예보가 glancing blow인지 (1 / 0 / −1) |
| K (CME 운동학) | `pub_cme_next_speed_kms`, `pub_cme_next_half_angle_deg`, `pub_cme_next_lon_deg`, `pub_cme_next_lat_deg` | 다음 예보의 지구 대상 CME 중 가장 빠른 CME의 속도·반각·방향 |
| | `pub_cme_last_speed_kms` | 지난 예보 CME의 속도 |

- 원자료: CCMC DONKI CME notifications (`https://ccmc.gsfc.nasa.gov/DONKI-API/get/notifications?type=CME`), 2022-01-01부터.
- 각 origin에서는 본문 발행 시각이 origin 이전인 메시지만 씁니다. 같은 CME의 갱신·철회 메시지가 나오면 예전 예보를 지웁니다. 7일이 지난 예보는 버립니다.
- 파싱 규칙은 2025-11-01 이전 메시지만 보고 정한 뒤 고정했습니다. 정답 값은 어디에도 쓰지 않습니다.
- −999는 "값 없음"입니다. 모든 origin에 값을 쓰기 때문에 로더의 forward fill이 예전 값을 끌고 오지 않습니다.
- 파싱 코드는 `research/_module/public_events.py`에 있습니다.

## 가설과 근거

- 기준선의 가장 큰 오차는 CME가 만든 빠른 태양풍 구간(≥ 700 km/s)을 크게 과소예측하는 것입니다(RMSE 182.8, 편향 −142.5).
- ACE는 L1에서 약 1시간 전에야 CME를 봅니다. 반면 DONKI는 1–4일 전에 공개 WSA-ENLIL 실행으로 지구 도달 시각과 CME 속도·반각·방향을 발표합니다.
- 기대한 메커니즘: 모델이 "언제" 오는지(T)와 "얼마나 강한지"(K)를 미리 알면 해당 구간의 lead에서 예측을 올릴 수 있습니다.
- 이 개선 크기를 보장하는 문헌은 없습니다. 효과는 아래 공식 채점으로만 판단합니다.

## 점수 (공식 scorer `lead-timse-6h-v1`, 낮을수록 좋음)

| 구간 | S | S-cap S | TIMSE | RMSE | 편향 |
|---|---:|---:|---:|---:|---:|
| Total | **93.864** | 95.498 | 17,920.4 | 67.48 | +16.9 |
| F1 | 114.306 | 116.424 | 27,061.7 | 82.51 | +20.2 |
| F2 | 73.939 | 74.673 | 8,087.2 | 62.78 | +15.6 |
| F3 | 59.865 | 60.731 | 4,778.2 | 54.75 | +14.9 |

seed별 Total S (S-cap 대비): 42 = 93.872 (−1.685), 43 = 94.121 (−1.867), 44 = 93.994 (−1.383). seed 평균 93.996, 표준편차 0.124.

- mean3 S는 1.71 % 낮아졌습니다. 세 seed 모두 좋아졌고 RMSE와 TIMSE도 둘 다 줄었습니다.
- F2 RMSE는 61.82에서 62.78로 1.5 % 올랐습니다. F2 S는 줄었습니다.
- 빠른 태양풍 구간(≥ 700 km/s, 634건): RMSE 182.8 → 176.2. 편향 −142.5 → −140.1로 과소예측은 거의 그대로입니다.
- 같은 캠페인의 T만 붙인 기준(A+T, 95.018)보다 0.90 낮지만, 그 차이의 95 % 구간(−1.88 ~ +0.20)은 0을 포함합니다.

## 사전 게이트

| 기준 | 결과 |
|---|---|
| mean3 S 1 % 이상 감소 | 통과 (−1.71 %) |
| 세 seed 모두 기준선보다 좋음 | 통과 |
| RMSE와 TIMSE 둘 다 감소 | 통과 |
| 분기별 1 % 이상 악화 없음 | 통과 |
| MAE 1 % 이상 악화 없음 | 통과 |
| 7일 이동 블록 bootstrap(5000회), Bonferroni 6 arm 보정 후 상한 < 0 | **미통과** (상한 +0.064, 보정 없는 95 % 구간 −2.39 ~ −0.23) |

신뢰구간 기준 하나를 근소하게 넘지 못해 **게이트 미통과**입니다. 누적 기준선으로 채택하지 않았습니다. 이 행은 사용자 요청으로 게시했습니다.

같은 캠페인의 다른 구성(모두 미통과): CME만 94.206 (−1.35 %), 양성자 96.317, 전부 합침 96.463, 플레어 96.760, 플레어 + 양성자 97.269. WSA-ENLIL 지구 위치 속도는 원자료 범위 기준(80 %)을 채우지 못해 학습하지 않았습니다.

## 폴더 구성

- `experiment.py` — 유일한 스크립트입니다. S-cap의 `SPEC`을 가져와 상수 8개를 붙입니다. 실행하면 DONKI 메시지로 `df.parquet` 임시 사본에 T·K 열을 붙이고 공용 러너 `research/_module/gbm.py`를 실행합니다. 임시 사본은 실행이 끝나면 지웁니다.
- `weights/F{1,2,3}/s{42,43,44}.npz` — 체크포인트입니다. 로컬에만 두고 git에는 올리지 않습니다.
- `research/result/e2000fs_calib_offset_roll_scap_arrival_cme/results/` — `{s42,s43,s44,mean3}_predictions.parquet`, `_score.json`.
- `research/result/e2000fs_calib_offset_roll_scap_arrival_cme/logs/` — `F{n}_s{seed}_metadata.json`과 `train_s{seed}.jsonl`(학습 run 기록), `{run,data,s*,mean3}.jsonl`(weights 추론 기록), `donki_notifications.json`(2026-11-13에 받은 메시지 1634개).

## 재현

```bash
CUDA_VISIBLE_DEVICES= python research/e2000fs_calib_offset_roll_scap_arrival_cme/experiment.py --threads 6        # DONKI를 받아 전체 학습
CUDA_VISIBLE_DEVICES= python research/e2000fs_calib_offset_roll_scap_arrival_cme/experiment.py --predict-only \
  --donki research/result/e2000fs_calib_offset_roll_scap_arrival_cme/logs/donki_notifications.json           # weights로 추론만
```

## 검증 기록 (2026-11-15)

- 학습은 캠페인 `runs/precursor_tk_s{42,43,44}`에서 했습니다. 3회 모두 exit 0이고 각 18단위를 마쳤습니다. 학습 commit은 79066584이고, xgboost 2.1.4, CPU 6 thread로 돌렸습니다.
- 학습에 쓴 모듈 사본과 저장소 commit 79066584의 `_module`은 `continuation.py` 한 곳만 다릅니다. 그 차이는 proposal 깊이를 인자로 뺀 것(기본 3 = PARAMS의 `max_depth`)과 진단용 callback뿐이어서 계산은 같습니다.
- `public_events.py`로 다시 만든 8개 특징은 캠페인 학습표와 34,968개 origin 전부에서 같습니다.
- 2026-11-15에 DONKI에서 새로 받은 메시지(1636개)로 만들어도 8개 특징이 모든 origin에서 같습니다.
- `--predict-only` 결과: seed 예측 3개가 학습 run 예측과 비트 단위로 같고, mean3 S는 93.864입니다.
- `--smoke`(2 rounds) exit 0.
