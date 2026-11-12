# e2000fs_calib_offset_roll_negative_combined_mean3 — lead 곡선 · F1–F3 월별 예측 GIF

작성자 **seongeun**의 `e2000fs_calib_offset_roll_negative_combined_mean3`입니다(제출 `20261107T103041788Z`). Total S **98.08**입니다.

이 페이지는 리더보드 dataset에 저장된 **제출 예측 자체**를 리더보드 채점기(`leaderboard/scorer/competition.py`, `lead-timse-6h-v1`)로 다시 채점해 그렸습니다. 관측은 기록이 채점된 dataset revision `ec692d9c5d`의 `competition-observations.parquet`입니다. Total/F1/F2/F3의 S·TIMSE·RMSE·tRMSE·bias가 저장 기록과 **1e-9 이내로 같을 때만** 그림을 그립니다([manifest.json](manifest.json)). 재학습이나 후처리는 하지 않았습니다.

| 구간 | S | TIMSE | RMSE | tRMSE | bias | 채점 origin | 유효 pair | persistence 대체 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Total** | 98.08 | 18,061.5 | 73.46 | 131.86 | +24.2 | 1,052 | 52,619 | 0.00% |
| F1 (2026-01..03) | 122.53 | 25,802.7 | 97.75 | 156.27 | +54.5 | 348 | 17,674 | 0.00% |
| F2 (2026-04..06) | 75.15 | 9,282.1 | 60.62 | 93.35 | +4.6 | 352 | 17,473 | 0.00% |
| F3 (2026-07..09) | 61.02 | 5,061.7 | 55.43 | 68.09 | +13.2 | 352 | 17,472 | 0.00% |

단위는 km/s(TIMSE는 (km/s)²)입니다. 채점 origin은 00/06/12/18 UTC이고, 정답은 ACE status 0 관측을 채우지 않고 씁니다. 그래서 1,052 × 72 = 75,744 pair 중 69.5%만 유효합니다.

## 제출 설명 (제출자 작성, 원문)

> Exploratory user-selected combination, not a proven cumulative baseline: remove eight features (ch_ch_t035_equatorial_area, strip_v450_logmed, rec27, strip_v650_dark, ch_ch_t055_meridian15_area, ace_density_cm3_last, strip_v750_logmed, ace_density_cm3_m6), set offset alpha=1 and spectral Hessian-only. All other original-A settings retained, no B1 blend. Fixed arithmetic mean of fresh seeds 42/43/44; equal 1/3 weights, no fitting to leaderboard labels. Official F1/F2/F3 only; training-only frozen calibration and 144h maturity/purge. Combination selected by user from prior diagnostic scores; no independent holdout claim. Inference time sums measured constituent times plus measured mean computation.

- 모델 정의 코드: [architecture](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/7801b0b13d90f9585e1fc72e6d5ed786638e3768/research/offset_roll_negative_combined_20261107/published/ensemble.py)
- 학습 데이터 종료: F1 2025-12-29T00:00:00Z, F2 2026-03-29T00:00:00Z, F3 2026-06-29T00:00:00Z

## Lead 곡선

![lead curves](lead_curves.png)

리더보드 lead curves 열과 같은 값을 이 페이지에서 직접 다시 채점해 그렸습니다([lead_metrics.csv](lead_metrics.csv), lead를 6으로 나눈 나머지별 평균은 [lead_residue.csv](lead_residue.csv)).

F3 TIMSE는 lead 6의 배수(6, 12, …, 72)에서 평균 **11,058.2**이고, 그 밖의 lead에서는 3,343.8–4,706.6입니다. 채점 origin이 00/06/12/18 UTC라서, lead 6·k의 목표 시각도 00/06/12/18 UTC에만 놓입니다. F3에서 가장 빠른 관측 두 개(743.4 km/s, 2026-09-15 18:00 UTC · 690.0 km/s, 2026-07-23 18:00 UTC)도 이 시각에 있어서, lead 6·k의 정답 모집단에만 들어갑니다. 두 관측을 정답에서 빼고 다시 계산하면(점수가 아닌 진단값) F3 lead 6·k TIMSE는 **5,228.4**입니다.

## 월별 예측 GIF

GIF 한 장의 프레임은 채점 origin 하나입니다. 각 프레임은 origin 직전 168시간의 관측(파랑), 이후 72시간의 정답(검정), 제출 예측(주황)을 Y축 200–850 km/s로 고정해 보여 줍니다. 관측과 정답은 ACE status 0 값만 쓰고, 결측은 **채우지 않고 비워 둡니다**. 제출 예측이 없거나 유효하지 않은 lead는 채점기와 같은 persistence 값으로 그리고, 제목에 그 개수를 표시합니다. 프레임 간격은 250 ms이며, 9개월 합계 1,052 프레임으로 채점 origin 전체입니다.

| 월 | 기간 | 프레임 | GIF |
| --- | --- | ---: | --- |
| 2026-01 | F1 | 124 | ![2026-01](forecast_2026-01.gif) |
| 2026-02 | F1 | 112 | ![2026-02](forecast_2026-02.gif) |
| 2026-03 | F1 | 112 | ![2026-03](forecast_2026-03.gif) |
| 2026-04 | F2 | 120 | ![2026-04](forecast_2026-04.gif) |
| 2026-05 | F2 | 124 | ![2026-05](forecast_2026-05.gif) |
| 2026-06 | F2 | 108 | ![2026-06](forecast_2026-06.gif) |
| 2026-07 | F3 | 124 | ![2026-07](forecast_2026-07.gif) |
| 2026-08 | F3 | 124 | ![2026-08](forecast_2026-08.gif) |
| 2026-09 | F3 | 104 | ![2026-09](forecast_2026-09.gif) |

## 재현

```bash
/home/t-lab01/sunrun/.venv/bin/python tools/submission_viz.py --member seongeun \
    --experiment e2000fs_calib_offset_roll_negative_combined_mean3 --submitted-utc 20261107T103041788Z --out /home/t-lab01/sunrun/research/e2000fs_calib_offset_roll_negative_combined/visualization
```

CPU만 씁니다. 입력은 리더보드 dataset `tlabtlab/sunrun-lb-store`에 저장된 이 제출의 결과 기록·예측 parquet과 `competition-observations.parquet`뿐입니다. [leaderboard-result.json](leaderboard-result.json)은 저장 기록에서 제출 키와 그 키를 담은 URL을 뺀 사본입니다.
