# e2000fs_calib_offset_roll_arrival_cme_direct_s

- 분류: **replacement ablation** (목적함수 교체). 누적 기준선 후보가 아닙니다.
- 요청: 2026-11-15 사용자 요청 — `e2000fs_calib_offset_roll_scap_arrival_cme`의 목적함수를 `e2000fs_calib_offset_roll_relevance_direct_s`의 것으로 바꿔 학습.
- 캠페인 실행 폴더: `~/sunrun-arrival-cme-direct-s/runs/arrival_cme_direct_s_s{42,43,44}` (로컬, CPU 6 thread × 3)

## 전략 (실행 전 작성)

### 비교 기준

| 기호 | 행 | 특징 | 목적함수 | mean3 S |
|---|---|---|---|---:|
| A | `e2000fs_calib_offset_roll_scap_arrival_cme` | 223열 (S-cap 215 + DONKI T/K 8) | A900 point4 + 0.1 spectral → S-cap continuation 180 tree | 93.864 |
| B | `e2000fs_calib_offset_roll_relevance_direct_s` | 35 원특징 (v_lag1/v_m6/v_lag3/v_lag6 포함, DONKI 없음) | stagewise dynamic Direct-S 900 tree | 97.128 |
| A′ | **이 행** | A와 동일 223열 | B와 동일 목적함수 | — |

- A → A′: 목적함수 한 가지만 바뀝니다. 특징 이름·순서(223 확장열), XGBoost 파라미터, 900 round, prior/full split, 336 origin × α 0.25 frozen offset, clip [1, 3000], seed 42/43/44, DONKI 말뭉치(arrival_cme `logs/donki_notifications.json`, 1634개)는 같습니다. 실행 전 `feature_names()`로 세 블록 모두 A와 일치하는 것을 확인했습니다.
- 제거된 구성: point4 tail 가중(550 km/s, ×4), 0.1 × FFT pseudo-Huber spectral 항, S-cap continuation(학습 MSE ≤ 1.01 × A900 제약, 180 tree). 총 tree 수는 A의 1080에서 900으로 줄어듭니다.
- B와는 특징 집합이 달라서 A′ − B는 목적함수 효과가 아니라 특징 효과(DONKI T/K 추가, v_lag1/v_m6/v_lag3/v_lag6 제거)입니다. 참고로만 기록합니다.

### 가설

- DONKI T/K 특징이 CME 구간의 TIMSE를 줄였다면(A: TIMSE 17,920), 처음부터 학습 S를 직접 최소화하는 목적함수가 그 정보를 point4 + spectral보다 더 직접 활용할 수 있다.
- 반대 근거: 같은 특징끼리 비교한 기존 기록에서 Direct-S는 point4보다 나빴습니다(`current_offset_roll_direct_s` 97.530 vs `e2000fs_calib_offset_roll` 97.109). RMSE와 편향(+19)이 커지는 경향이 있습니다. 그래서 개선은 보장되지 않으며, 효과는 공식 채점으로만 판단합니다.

### 판정 기준 (사전 고정)

- 공식 scorer(`leaderboard/scorer/competition.py`, 리더보드 관측 파일)로 F1/F2/F3와 Total의 S, TIMSE, RMSE, MAE, 편향을 seed별과 mean3로 보고합니다.
- 제출 후보 조건: A 대비 mean3 S 1 % 이상 감소(≤ 92.925), 세 seed 모두 A보다 좋음, 분기별 S·MAE·RMSE 1 % 이상 악화 없음.
- 위 조건을 통과하지 못하면 `complete.json`에 `publish: not_required`로 남기고 게시하지 않습니다(사용자가 따로 요청하면 게시).
- 리더보드 정답은 채점에만 씁니다. 보정·튜닝·seed 선택에 쓰지 않습니다.

## 상태

- 2026-11-15: smoke(2 rounds, F1, seed 42) exit 0, 확장열 223, StagewiseS 적용 확인. 전체 학습 시작.

## 결과 (2026-11-15, 공식 scorer, 낮을수록 좋음)

- 학습: 로컬 CPU 6 thread × 3 seed, 2026-11-15 03:18–04:30 UTC. 세 seed 모두 exit 0, 기간별 metadata·예측·score 기록. 18개 block 모두 900 round 전부 채택(`proposal_budget_exhausted`, 중간 거부 없음).
- mean3 평가: `~/sunrun-arrival-cme-direct-s/evaluation/{mean3_predictions.parquet, mean3_score.json, report.json}`.

| 구간 | S (A′) | S (A) | TIMSE (A′ / A) | RMSE (A′ / A) | MAE (A′ / A) | 편향 (A′ / A) |
|---|---:|---:|---:|---:|---:|---:|
| Total | **95.572** | 93.864 | 18,236 / 17,920 | 69.43 / 67.48 | 53.38 / 51.85 | +19.8 / +16.9 |
| F1 | 117.113 | 114.306 | 28,517 / 27,062 | 84.72 / 82.51 | 64.70 / 62.77 | +22.9 / +20.2 |
| F2 | 74.869 | 73.939 | 7,844 / 8,087 | 65.25 / 62.78 | 51.65 / 49.68 | +18.9 / +15.6 |
| F3 | 60.778 | 59.865 | 4,881 / 4,778 | 55.85 / 54.75 | 43.67 / 42.97 | +17.6 / +14.9 |

seed별 Total S (A 대비): 42 = 95.265 (+1.393), 43 = 96.300 (+2.179), 44 = 95.671 (+1.677).

### 판정

- mean3 S +1.82 % (93.864 → 95.572). 세 seed 모두 A보다 나쁘고, 모든 기간에서 S·RMSE·MAE가 1 % 넘게 나빠졌습니다. **게이트 미통과, 채택하지 않음, 게시하지 않음.**
- MAE는 scorer에 없어 같은 관측 파일로 별도 계산했습니다(관측 315,839쌍).

### 원인 분석

- 구현 결함은 아닙니다: 특징 223열은 A와 일치하고, fitter는 `relevance_direct_s`와 같은 `StagewiseS(dynamic, 30)`이며 모든 stage가 채택되었습니다.
- 가설 결함입니다. 목적함수만 바꾼 효과(A′ − A)는 +1.71 S입니다. 같은 특징에서 point4보다 Direct-S가 나빴던 기존 기록(97.530 vs 97.109)과 방향이 같고, 크기는 S-cap까지 포함한 A가 기준이라 더 큽니다.
- 기전: Direct-S는 F2에서 TIMSE를 3 % 줄였지만 편향이 모든 기간에서 +2.7~3.3 km/s 커지고 RMSE가 1.8~3.9 % 늘었습니다. 기존 Direct-S 계열(편향 +19, RMSE ≈ 70)과 같은 패턴입니다. point4 + spectral로 먼저 맞춘 뒤 MSE 제약 아래에서만 S를 미는 S-cap 구조가 이 편향·RMSE 손실을 막고 있었다고 해석합니다.
- 참고(특징 효과, 목적함수 같음): A′ 95.572 vs `relevance_direct_s` 97.128 → −1.56. S-cap에서의 DONKI T/K 효과(−1.63)와 비슷해, DONKI 정보의 효과는 목적함수와 거의 독립입니다. 특징 집합이 v_lag1/v_m6/v_lag3/v_lag6도 다르므로 엄밀한 단일 요인 비교는 아닙니다.
- 다음: 목적함수 축에서 S-cap을 Direct-S 단독으로 바꾸는 방향은 기각합니다. 같은 seed run을 다시 돌리지 않습니다.
