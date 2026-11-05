# e2000fs_calib8_qxgb

- 작성자: `seongeun`
- 리더보드 상태: 숨김(visibility.json)

## 설명 (최신 제출)

Fresh raw-only df.parquet + causal feature ffill rerun of e2000fs_calib8_qxgb. Input features and TRAINING y causally forward filled; official evaluation truth unchanged. Model architecture and seed preserved; current F1/F2/F3 144-hour maturity masks. Full training completed; no evaluation-target calibration or selection. Baseline submission: 20261031T172626903Z. Existing results retained.

## 제출과 아키텍처 소스

| 제출일 (UTC) | Total S | 아키텍처 소스 (고정 커밋) |
|---|---:|---|
| 2026-11-04 | 105.509 | [`research/rerun_df_ffill/e2000fs_calib8_qxgb/experiment.py`](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/a759ffc922a93773bbf0ee0859352c68a4ed5279/research/rerun_df_ffill_20261103/e2000fs_calib8_qxgb/experiment.py) @ `a759ffc92` |
| 2026-10-31 | 106.866 | [`research/e2000fs_f123-20261031/qxgb.py`](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/6b85acc0b00d731bf31b6885e0b847c18ee395d0/research/e2000fs_f123-20261031/qxgb.py) @ `6b85acc0b` |

## 참고

- 코드는 제출 행마다 위 표의 고정 커밋 소스에 있다. 최신 제출은 `research/rerun_df_ffill/` 공용 재학습 러너(여러 recipe를 `--experiment`로 고름)이고, 이전 제출은 2026-10-31 레인 스크립트의 해당 줄이다. 여러 실험이 공유하는 러너라 여기에 복제하지 않는다.
- 재실행(최신): `python research/rerun_df_ffill/e2000fs_calib8_qxgb/experiment.py --experiment e2000fs_calib8_qxgb --data <df_ffill.parquet> --out runs/e2000fs_calib8_qxgb --threads N` (`CUDA_VISIBLE_DEVICES=''`).

Total S는 리더보드 store의 `competition.total.s`(공식 scorer, F1/F2/F3, 낮을수록 좋음).
