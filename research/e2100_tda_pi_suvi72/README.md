# e2100_tda_pi_suvi72

- 작성자: `seongeun`
- 리더보드 상태: 숨김(visibility.json)

## 설명 (최신 제출)

Fresh raw-only df.parquet + causal feature ffill rerun of e2100_tda_pi_suvi72. Input features and TRAINING y causally forward filled; official evaluation truth unchanged. Model architecture and seed preserved; current F1/F2/F3 144-hour maturity masks. Full training completed; no evaluation-target calibration or selection. Baseline submission: 20261031T182427407Z. Existing results retained.

## 제출과 아키텍처 소스

| 제출일 (UTC) | Total S | 아키텍처 소스 (고정 커밋) |
|---|---:|---|
| 2026-11-03 | 128.596 | [`research/rerun_df_ffill_20261103/e2100_tda_pi_suvi72/experiment.py`](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/b2d624b78e2032c613f1193b36d1ac635d78dac4/research/rerun_df_ffill_20261103/e2100_tda_pi_suvi72/experiment.py) @ `b2d624b78` |
| 2026-10-31 | 131.458 | [`research/e2100_tda_pi_suvi-20261031/run_all.py`](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/6b85acc0b00d731bf31b6885e0b847c18ee395d0/research/e2100_tda_pi_suvi-20261031/run_all.py) @ `6b85acc0b` |

## 이 폴더의 파일

- `experiment.py` — 최신 제출(fresh-df ffill 재학습)의 아키텍처 소스 사본(`research/rerun_df_ffill_20261103/e2100_tda_pi_suvi72/experiment.py`).

Total S는 리더보드 store의 `competition.total.s`(공식 scorer, F1/F2/F3, 낮을수록 좋음).
