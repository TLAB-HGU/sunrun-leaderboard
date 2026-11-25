# fianl_v1

`e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel_plus7`의 세 가지 제거를 결합한 탐색 실험이다. 비교 대상은 원본 plus7 Mean3의 Total S 90.17361022099254다.

- 원본 F1–F3 및 seed 42–44의 최종 GBM 분기에서 쓰이지 않은 확장 입력 96열을 고정해 제거한다. 0이 아닌 continuation 가중치를 가진 트리까지 검사했다.
- `enlil_earth_speed`는 1–24h, `rec54_consensus`는 49–72h에서만 24열씩 추가 제거한다. 최종 입력 폭은 305 / 308 / 283열이다.
- GBM은 블록마다 재학습한다. 600라운드, S-cap, prior/full, offset, seam xf8, 위상 전진 3h, seed 합의 uplift와 동일 가중 Mean3를 사용한다. 별도 ridge 적합이나 결합은 없다.
- 학습 자료는 `data-v2/input/df.parquet`다(2026-11-25 재학습부터, 아래 참고). F1–F3 cutoff 이전 자료로 모델과 offset을 적합하고, 공식 채점기로 seed 및 Mean3의 S/RMSE/TIMSE, 기간별 결과를 기록한다. 블록별 열 제거가 원본의 일부 중복 정보를 정리해 손실을 작게 유지하는지가 가설이다.
- 부모 `lbsel`의 구성 선정과 이번 제거 후보는 이전 F1–F3 결과에 의존한다. 이 결과는 탐색적이며 동일 평가 구간에서 독립적인 일반화를 주장할 수 없다.

재현에는 `data-v2/input/df.parquet`(457열, `enlil_earth_speed`는 NOAA open-data S3 실행만)를 쓴다.

```sh
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/fianl_v1/experiment.py --threads 6
```

## df.parquet 재학습 (2026-11-25, 사용자 요청)

- 이유: 첫 게시판(아래 "NCEI 판 결과")은 NCEI WSA-Enlil 아카이브가 들어간 고정 표로 학습했다. NCEI는 실행 후 약 25 h 뒤에 공개되어 실시간 예보에서 같은 값을 얻을 수 없다(h25–72 `enlil_earth_speed` 칸의 약 10 %가 S3판과 다름).
- 바꾼 것은 입력 표 하나다: 고정 표 → `data-v2/input/df.parquet`. 제거 열 목록, 블록 폭 305/308/283, 600라운드, 목적함수, 후처리, seed 42/43/44는 그대로다.
- `df.parquet`의 나머지 열은 고정 표와 같다. 차이는 `enlil_earth_speed`(S3 실행만)와, 이 실험이 쓰지 않는 ACE 위치 9열이 없는 것뿐이다.
- 기준선: NCEI 판 fianl_v1 Mean3 Total S 89.34587910011088. 같은 데이터 변경으로 plus7은 90.174 → 91.563(+1.39)이 되었으므로, 비슷한 악화를 예상한다. 개선 가설이 아니라 실시간 재현이 가능한 판으로 바꾸는 것이다.
- 보드 행 `fianl_v1`은 이 결과로 교체한다. NCEI 판 기록은 캠페인 폴더 `~/sunrun-fianl-v1-df/old-ncei-lb-record/`에 백업한다.

## NCEI 판 결과 (2026-11-25 첫 게시, 교체됨)

| 기간 | 원본 S | fianl_v1 S | RMSE | TIMSE |
|---|---:|---:|---:|---:|
| TOTAL | 90.1736 | 89.3459 | 72.0907 | 13131.8078 |
| F1 | 107.3455 | 106.1042 | 85.8107 | 19449.8815 |
| F2 | 73.8227 | 73.1532 | 68.5847 | 6511.9933 |
| F3 | 59.8433 | 60.4245 | 60.0005 | 4142.5370 |

세 시드의 재학습 예측과 체크포인트 재실행 예측은 최종 후처리 후 정확히 일치했다.
