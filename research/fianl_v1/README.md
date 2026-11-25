# fianl_v1

`e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel_plus7`의 세 가지 제거를 결합한 탐색 실험이다. 비교 대상은 원본 plus7 Mean3의 Total S 90.17361022099254다.

- 원본 F1–F3 및 seed 42–44의 최종 GBM 분기에서 쓰이지 않은 확장 입력 96열을 고정해 제거한다. 0이 아닌 continuation 가중치를 가진 트리까지 검사했다.
- `enlil_earth_speed`는 1–24h, `rec54_consensus`는 49–72h에서만 24열씩 추가 제거한다. 최종 입력 폭은 305 / 308 / 283열이다(NCEI 판). df.parquet 판은 `onset700_hazard`를 모든 리드에서 더 빼 284 / 304 / 283열이다.
- GBM은 블록마다 재학습한다. 600라운드, S-cap, prior/full, offset, seam xf8, 위상 전진 3h와 동일 가중 Mean3를 사용한다(seed 합의 uplift는 NCEI 판에만 있었고 df.parquet 판에서 뺐다, 아래 참고). 별도 ridge 적합이나 결합은 없다.
- 학습 자료는 `data-v2/input/df.parquet`다(2026-11-25 재학습부터, 아래 참고). F1–F3 cutoff 이전 자료로 모델과 offset을 적합하고, 공식 채점기로 seed 및 Mean3의 S/RMSE/TIMSE, 기간별 결과를 기록한다. 블록별 열 제거가 원본의 일부 중복 정보를 정리해 손실을 작게 유지하는지가 가설이다.
- 부모 `lbsel`의 구성 선정과 이번 제거 후보는 이전 F1–F3 결과에 의존한다. 이 결과는 탐색적이며 동일 평가 구간에서 독립적인 일반화를 주장할 수 없다.

재현에는 `data-v2/input/df.parquet`(457열, `enlil_earth_speed`는 NOAA open-data S3 실행만)를 쓴다.

`experiment.py`는 한 파일로 끝까지 실행된다(2026-11-25, 사용자 요청): df.parquet 읽기, 인과 앞값 채움, (origin × 72 lead) 격자, 블록 설계(284/304/283열), smooth-tail A 단계 600트리 + S-cap continuation 6 × 30트리, prior/full과 동결 offset, checkpoint, seam xf8 → 위상 전진, 세 seed 평균, 공식 채점기까지 들어 있고 부모 실험이나 `research/_module`을 import하지 않는다. uplift를 빼기 전의 한 파일 판(커밋 `85553acc3`)은 이전의 부모 체인 import 버전과 smoke 예측·checkpoint 배열이 bit 단위로 같았고, NCEI 판 checkpoint를 `--predict-only`로 돌리면 게시했던 Total S 89.34587910011088과 예측이 그대로 재현됐다.

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/fianl_v1/experiment.py --threads 6
```

## df.parquet 재학습 (2026-11-25, 사용자 요청)

- 이유: 첫 게시판(아래 "NCEI 판 결과")은 NCEI WSA-Enlil 아카이브가 들어간 고정 표로 학습했다. NCEI는 실행 후 약 25 h 뒤에 공개되어 실시간 예보에서 같은 값을 얻을 수 없다(h25–72 `enlil_earth_speed` 칸의 약 10 %가 S3판과 다름).
- 바꾼 것은 입력 표 하나다: 고정 표 → `data-v2/input/df.parquet`. 제거 열 목록, 블록 폭 305/308/283, 600라운드, 목적함수, 후처리, seed 42/43/44는 그대로다.
- `df.parquet`의 나머지 열은 고정 표와 같다. 차이는 `enlil_earth_speed`(S3 실행만)와, 이 실험이 쓰지 않는 ACE 위치 9열이 없는 것뿐이다.
- 기준선: NCEI 판 fianl_v1 Mean3 Total S 89.34587910011088. 같은 데이터 변경으로 plus7은 90.174 → 91.563(+1.39)이 되었으므로, 비슷한 악화를 예상한다. 개선 가설이 아니라 실시간 재현이 가능한 판으로 바꾸는 것이다.
- seed 합의 uplift 40 제거(2026-11-25, 사용자 요청): overnight loop round 15(2026-11-21, seam v1 기준)에서 mean3 −0.067(−0.07 %), CI [−0.21, +0.32]로 효과 없음 판정이었는데 union9의 "2 seed 이상 개선" 규칙으로 들어와 있었다. 후처리는 seam xf8 → 위상 전진 3 h만 남는다. 학습·checkpoint는 영향이 없다. 같은 NCEI 판 checkpoint에서 uplift만 빼면 Mean3 Total S 89.3459 → 89.3734(+0.0275, F1 −0.19, F2 +0.08, F3 −0.02; seed별 +0.022/+0.034/+0.028)로, 영향은 0.03 % 수준이다.
- `onset700_hazard` 제거(2026-11-25, 사용자 요청): `~/sunrun-discrimination10` 단독 평가(2026-11-24, seam v1 기준)에서 mean3 −0.299(−0.32 %)였지만 7일 블록 bootstrap CI [−0.54, +0.06]이 0을 포함했고, 1월 사건을 빼면 −0.12로 줄어 연구 게이트를 통과하지 못했다. plus7에서 검증 없이 추가된 특징이다. 블록 폭은 284 / 304 / 283열이 되고(블록별 21 / 4 / 0열 제거), 나머지 열의 값은 그대로다. 학습을 다시 한다.
- 대조: 같은 df.parquet에서 `onset700_hazard`를 두고 uplift만 뺀 판(`~/sunrun-fianl-v1-df`, 게시 안 함)은 Mean3 Total S 91.5251(F1 109.8571, F2 73.6445, F3 60.3082)이다.
- 따라서 NCEI 판과의 차이는 입력 표, uplift 제거, `onset700_hazard` 제거 세 가지다.
- 보드 행 `fianl_v1`은 이 결과로 교체한다. NCEI 판 기록은 캠페인 폴더 `~/sunrun-fianl-v1-df/old-ncei-lb-record/`에 백업한다.

## NCEI 판 결과 (2026-11-25 첫 게시, 교체됨)

| 기간 | 원본 S | fianl_v1 S | RMSE | TIMSE |
|---|---:|---:|---:|---:|
| TOTAL | 90.1736 | 89.3459 | 72.0907 | 13131.8078 |
| F1 | 107.3455 | 106.1042 | 85.8107 | 19449.8815 |
| F2 | 73.8227 | 73.1532 | 68.5847 | 6511.9933 |
| F3 | 59.8433 | 60.4245 | 60.0005 | 4142.5370 |

세 시드의 재학습 예측과 체크포인트 재실행 예측은 최종 후처리 후 정확히 일치했다.
