# plus7 + plus7-TIMSE 50:50 예측 평균

- 작성자: `seongeun`
- 상태: 2026-11-25 사용자 요청으로 채점했고, 결과와 관계없이 **탐색 행으로 게시**했다.

## 전략 (채점 전 고정)

- 질문: 1위 행 plus7의 손실함수(smooth-tail MSE 600 + S-cap 180)를 순수 TIMSE로 바꾼 모델을 기존 plus7과 같은 비율로 섞으면, 두 목적함수의 오차가 서로 상쇄되어 S가 좋아지는가?
- 기준선 A: `e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel_plus7` (mean3 Total S 90.17361022099254)
- 성분 B: 같은 345/345/350열 설계와 학습 구간을 쓰고, 순수 TIMSE 목적(gradient = w·error, w = 대회 TIMSE 계수 × N, 리드별 학습 정답으로 계산)으로 780라운드 학습했다. continuation과 S-cap은 쓰지 않는다. `…_plus7_mse_timse` 행의 TIMSE 성분과 같은 checkpoint다.
- 결합 A+B: seed별 원시 GBM 예측을 `0.5·A + 0.5·B`로 평균한 뒤 plus7과 같은 고정 후처리(seam xf8 → union9 ridge 5 % → 3 h 위상 전진 → seed 합의 uplift ≤ 40)를 적용하고, seed 42/43/44의 등가중 평균을 낸다.
- 비율 0.5는 채점 전에 고정했고 탐색하지 않았다. 특징, 후처리, ridge, offset은 plus7과 같다.
- 제출 규정: 모델 적합은 각 기간 cutoff 이전 자료만 쓴다. 다만 plus7 특징을 F1–F3 리더보드 점수로 골랐기 때문에 탐색 제출(`tuning_cutoff_exception`)이다. 누적 기준선으로 쓰지 않는다.

## 재현

```bash
CUDA_VISIBLE_DEVICES='' python experiment.py --component plus7 --seeds 42 --data TABLE --observations OBS --out RUNS/plus7/s42
CUDA_VISIBLE_DEVICES='' python experiment.py --component timse --seeds 42 --data TABLE --observations OBS --out RUNS/timse/s42
# … seed 43, 44
CUDA_VISIBLE_DEVICES='' python experiment.py --finalize --plus7-runs RUNS/plus7 --timse-runs RUNS/timse \
    --data TABLE --weights <union9_lbsel>/weights --observations OBS --out OUT
```

TABLE은 `model-table-union9-plus7.parquet`(plus7 입력 표)이다. 같은 후처리 경로에서 가중치를 1로 두면 plus7 점수(seed별, mean3)가 정확히 재현됨을 확인했다.

## 결과 (2026-11-25, mean3, 공식 scorer)

| 모델 | Total S | Total TIMSE | Total RMSE | bias | F1 S | F2 S | F3 S | seed S |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A plus7 | **90.174** | 13437 | **72.43** | **+28.8** | **107.346** | **73.823** | **59.843** | 90.033 / 90.128 / 90.778 |
| B TIMSE만 | 99.431 | 14181 | 85.31 | +46.1 | 117.197 | 82.066 | 68.173 | 99.039 / 99.680 / 100.153 |
| **A+B 50:50** | 93.337 | **13396** | 77.65 | +37.9 | 110.966 | 76.416 | 62.676 | 92.952 / 93.388 / 93.936 |
| 참고: MSE+TIMSE 50:50 | 94.405 | 15882 | 72.85 | +31.5 | 113.212 | 75.344 | 60.938 | 94.041 / 94.608 / 94.801 |

- A 대비 A+B: Total S **+3.164 (+3.51 %)**. seed별 +2.919 / +3.260 / +3.158로 세 seed 모두 나빠졌다. TIMSE는 −0.3 %로 거의 같고, RMSE는 +7.2 %, bias는 +9.1 km/s 늘었다. F1/F2/F3 모두 악화했다.
- 판정: **실패**. 기준선으로 채택하지 않는다.
- 원인: TIMSE 단독 모델은 TIMSE조차 A보다 나쁘고(14181 vs 13437) 과대예측이 크다(bias +46). TIMSE 가중치가 꼬리 셀에 몰려 있어 평균 수준을 위로 끌어올리기 때문이다. 평균하면 꼬리 오차는 일부 줄지만(TIMSE −0.3 %) 전 구간의 bias와 RMSE가 늘어 S가 나빠진다. A의 S-cap 단계가 이미 TIMSE를 S 안에서 MSE 상한과 함께 최적화하므로, 따로 학습한 TIMSE 성분이 더할 정보가 없다.
