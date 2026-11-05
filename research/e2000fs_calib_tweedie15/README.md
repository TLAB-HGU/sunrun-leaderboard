# e2000fs_calib_tweedie15

- 작성자: `seongeun`
- 리더보드 상태: 숨김(visibility.json)

## 설명 (최신 제출)

Fresh raw-only df.parquet + causal feature ffill rerun of e2000fs_calib_tweedie15. Input features and TRAINING y causally forward filled; official evaluation truth unchanged. Model architecture and seed preserved; current F1/F2/F3 144-hour maturity masks. Full training completed; no evaluation-target calibration or selection. Baseline submission: 20261031T172050000Z. Existing results retained.

## 제출과 아키텍처 소스

| 제출일 (UTC) | Total S | 아키텍처 소스 (고정 커밋) |
|---|---:|---|
| 2026-11-04 | 106.465 | [`research/rerun_df_ffill_20261103/e2000fs_calib_tweedie15/experiment.py`](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/d0e151c25d156532122c5646c7159e5283fd94c8/research/rerun_df_ffill_20261103/e2000fs_calib_tweedie15/experiment.py) @ `d0e151c25` |
| 2026-10-31 | 107.079 | [`research/e2000fs_calib-20261030/tweedie_lane.py#L91-L128`](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/b930cc556b626a633c58d8fd74680cbf80f27008/research/e2000fs_calib-20261030/tweedie_lane.py#L91-L128) @ `b930cc556` |

## 참고

- 코드는 제출 행마다 위 표의 고정 커밋 소스에 있다. 최신 제출은 `research/rerun_df_ffill_20261103/` 공용 재학습 러너(여러 recipe를 `--experiment`로 고름)이고, 이전 제출은 2026-10-31 레인 스크립트의 해당 줄이다. 여러 실험이 공유하는 러너라 여기에 복제하지 않는다.
- 재실행(최신): `python research/rerun_df_ffill_20261103/e2000fs_calib_tweedie15/experiment.py --experiment e2000fs_calib_tweedie15 --data <df_ffill.parquet> --out runs/e2000fs_calib_tweedie15 --threads N` (`CUDA_VISIBLE_DEVICES=''`).

Total S는 리더보드 store의 `competition.total.s`(공식 scorer, F1/F2/F3, 낮을수록 좋음).
