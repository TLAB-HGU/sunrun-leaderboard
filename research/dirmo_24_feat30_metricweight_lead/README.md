# dirmo_24_feat30_metricweight_lead

- 작성자: `jsy301`
- 리더보드 상태: 숨김(visibility.json)

## 설명 (최신 제출)

DirMO-24 feat30 (team recipe, mean of seeds 42/1/2) trained with loss weights derived from the board score S: TIMSE decomposes into per-row weights, w = a + (1 - a) n c(y) per lead, a = TIMSE/(MSE+TIMSE) from 4-fold out-of-fold errors of the training rows; no hand-set 550 km/s / 4x threshold. Plus a per-lead factor S_l (1/MSE_l + 1/TIMSE_l), mean 1 per block. Lab check: pooled S -1.28 % vs 4x@550 [CI -2.65,-0.18], 5/5 windows; strict bar failed (2023-H2 +3.4 % vs unweighted). Informational entry (jsy301 round MW). Inference time measured on the sibling metricweight run (same predict work).

## 제출과 아키텍처 소스

| 제출일 (UTC) | Total S | 아키텍처 소스 (고정 커밋) |
|---|---:|---|
| 2026-11-02 | 103.846 | [`research/dirmo_24_feat30_metricweight/experiment.py`](https://github.com/TLAB-HGU/sunrun-leaderboard/blob/70d83427df449c8a0a4513b72a01aec2ba190db2/research/dirmo_24_feat30_metricweight/experiment.py) @ `70d83427d` |

## 참고

- 코드는 형제 실험 [`research/dirmo_24_feat30_metricweight/`](../dirmo_24_feat30_metricweight/)의 `experiment.py`가 lead 계수 옵션으로 함께 정의한다(그 폴더 README 표 참조). 코드를 복제하지 않는다.

Total S는 리더보드 store의 `competition.total.s`(공식 scorer, F1/F2/F3, 낮을수록 좋음).
