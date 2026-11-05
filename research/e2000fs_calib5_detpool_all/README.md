# e2000fs_calib5_detpool_all

- 작성자: `seongeun`
- 리더보드 상태: 숨김(visibility.json)

## 설명 (최신 제출)

Refit 2026-10-31 for the F1/F2/F3 quarters: the same recipe retrained with training cutoffs 2025-12-29 (F1), 2026-03-29 (F2) and 2026-06-29 (F3), 72 h target maturity, forecast for all 6,315 origins. Earlier description: All 81 archived arms stacked onto detpool_gpd: base-relative non-negative least squares (ridge 1e-3, sum(beta)<=1) refit every 168 origins on the trailing 2160 matured origins, with a trailing >=550 no-regression shrink. No arm selection, no retraining, unconditional F1/F2.

## 제출과 아키텍처 소스

| 제출일 (UTC) | Total S | 아키텍처 소스 (고정 커밋) |
|---|---:|---|
| 2026-10-31 | 102.170 | [`research/e2000fs_calib5-20261031/pool_all.py#L41-L158`](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/b930cc556b626a633c58d8fd74680cbf80f27008/research/e2000fs_calib5-20261031/pool_all.py#L41-L158) @ `b930cc556` |

## 참고

- 코드는 제출 행마다 위 표의 고정 커밋 소스에 있다. 최신 제출은 `research/rerun_df_ffill/` 공용 재학습 러너(여러 recipe를 `--experiment`로 고름)이고, 이전 제출은 2026-10-31 레인 스크립트의 해당 줄이다. 여러 실험이 공유하는 러너라 여기에 복제하지 않는다.
- 재실행(최신): `python research/rerun_df_ffill/e2000fs_calib5_detpool_all/experiment.py --experiment e2000fs_calib5_detpool_all --data <df_ffill.parquet> --out runs/e2000fs_calib5_detpool_all --threads N` (`CUDA_VISIBLE_DEVICES=''`).

Total S는 리더보드 store의 `competition.total.s`(공식 scorer, F1/F2/F3, 낮을수록 좋음).
