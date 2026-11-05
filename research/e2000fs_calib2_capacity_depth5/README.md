# e2000fs_calib2_capacity_depth5

- 작성자: `seongeun`
- 리더보드 상태: 숨김(visibility.json)

## 설명 (최신 제출)

Fresh raw-only df.parquet + causal feature ffill rerun of e2000fs_calib2_capacity_depth5. Input features and TRAINING y causally forward filled; official evaluation truth unchanged. Model architecture and seed preserved; current F1/F2/F3 144-hour maturity masks. Full training completed; no evaluation-target calibration or selection. Baseline submission: 20261030T235559445Z. Existing results retained.

## 제출과 아키텍처 소스

| 제출일 (UTC) | Total S | 아키텍처 소스 (고정 커밋) |
|---|---:|---|
| 2026-11-04 | 103.144 | [`research/rerun_df_ffill_20261103/e2000fs_calib2_capacity_depth5/experiment.py`](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/e3c3f87306cbf520ff3b2653d389e889526b5e63/research/rerun_df_ffill_20261103/e2000fs_calib2_capacity_depth5/experiment.py) @ `e3c3f8730` |
| 2026-10-30 | 123.747 | — |

## 참고

- 코드는 제출 행마다 위 표의 고정 커밋 소스에 있다. 최신 제출은 `research/rerun_df_ffill_20261103/` 공용 재학습 러너(여러 recipe를 `--experiment`로 고름)이고, 이전 제출은 2026-10-31 레인 스크립트의 해당 줄이다. 여러 실험이 공유하는 러너라 여기에 복제하지 않는다.
- 재실행(최신): `python research/rerun_df_ffill_20261103/e2000fs_calib2_capacity_depth5/experiment.py --experiment e2000fs_calib2_capacity_depth5 --data <df_ffill.parquet> --out runs/e2000fs_calib2_capacity_depth5 --threads N` (`CUDA_VISIBLE_DEVICES=''`).

Total S는 리더보드 store의 `competition.total.s`(공식 scorer, F1/F2/F3, 낮을수록 좋음).
