# e2000fs_calib_offset_roll

- 작성자: `seongeun`
- 리더보드 상태: 보드 표시

## 설명 (최신 제출)

Fresh raw-only df.parquet + causal feature ffill rerun of e2000fs_calib_offset_roll. Input features and TRAINING y causally forward filled; official evaluation truth unchanged. Model architecture and seed preserved; current F1/F2/F3 144-hour maturity masks. Full training completed; no evaluation-target calibration or selection. Baseline submission: 20261031T162554000Z. Existing results retained.

## 제출과 아키텍처 소스

| 제출일 (UTC) | Total S | 아키텍처 소스 (고정 커밋) |
|---|---:|---|
| 2026-11-04 | 98.763 | [`research/rerun_df_ffill_20261103/e2000fs_calib_offset_roll/experiment.py`](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/38fc345ca8f7a31a8215b8942ce6e596019af500/research/rerun_df_ffill_20261103/e2000fs_calib_offset_roll/experiment.py) @ `38fc345ca` |
| 2026-10-31 | 101.542 | [`research/e2000fs_calib-20261030/lanes.py#L101-L192`](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/b930cc556b626a633c58d8fd74680cbf80f27008/research/e2000fs_calib-20261030/lanes.py#L101-L192) @ `b930cc556` |

## 이 폴더의 파일

- `experiment.py` — 현재 보드 행의 아키텍처. main `c538d5b97`의 단일 아키텍처 정리본(정리 전 `38fc345ca`와 F1 smoke 예측·`frozen_calibration.json` 동일).

Total S는 리더보드 store의 `competition.total.s`(공식 scorer, F1/F2/F3, 낮을수록 좋음).
