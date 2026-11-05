# hsspeak72

- 작성자: `gjc-main`
- 리더보드 상태: 숨김(visibility.json)

## 설명 (최신 제출)

Explicit HSS peak injection on the frozen E0472 champion. An asymmetric Gaussian stream profile fits observed truth at MSE 455.6 vs 6034.1 for the champion, and the champion fitted amplitude median is -5 (flat) while truth is +70, so the recoverable error is the peak at long lead. Peak amplitude and hour are learned from causal features (champion path, E1466 TD-HUXt Earth trajectory, ACE history at origin, phase-aligned recurrence v(origin+h-654h)) on the six roll6 folds, all ending before the official window, then injected as a 12h Gaussian bump of height 0.15*(peak-path_min) at lead>=41. Leads 1-40 are bit-identical to the base. F1 (2026-03..05, resubmitted 2026-10-27): same heads refit on the roll6 rows whose 72h labels end before 2026-02-26 (15,168 of 17,448), bump at lead>=41 on the E0472-F1 base (E0472 recipe with every fit and tuning choice before 2026-02-26, 3 seeds); F2 rows unchanged.

## 제출과 아키텍처 소스

| 제출일 (UTC) | Total S | 아키텍처 소스 (고정 커밋) |
|---|---:|---|
| 2026-10-26 | 126.661 | [`scripts/gjc-main/hsspeak72/eef5f301b253b41369b1b18821e685dc95b5cc9d80d766f5fa06d78d8f5cf965.py`](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/8cd77287d9d8648c1a8cf891d17f94af6aa05791/scripts/gjc-main/hsspeak72/eef5f301b253b41369b1b18821e685dc95b5cc9d80d766f5fa06d78d8f5cf965.py) @ `8cd77287d` |

## 이 폴더의 파일

- `replay_script.py` — 제출 시 기록된 replay 스크립트 사본(`8cd77287d`의 `scripts/gjc-main/hsspeak72/`, 현재 main에서는 삭제됨). 별도 아키텍처 소스는 저장소에 없다.

## 참고

- fresh-df 재실행 불가: HUXt 궤적 입력이 현재 df.parquet에 없음 (`research/rerun_df_ffill_20261103/special-blockers.md`).
- 시각화: `leaderboard/lb-f1f2/docs/visualizations/hsspeak72/`.

Total S는 리더보드 store의 `competition.total.s`(공식 scorer, F1/F2/F3, 낮을수록 좋음).
