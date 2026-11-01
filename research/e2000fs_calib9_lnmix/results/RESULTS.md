# e2000fs_calib9_lnmix — 결과 (2026-11-01)

`e2000fs_calib2_mixture2comp`의 휴리스틱 4종을 하나의 likelihood로 대체한 실행. 설계·문헌: `research/notes/lit-review-round8-20261101.md`.
적합: **XGBoostLSS**(Chevalier & Côté 2025, *Eur. Actuar. J.*, Algorithm 4) — 3K−1=5개 파라미터 그룹을 2 사이클 ×
그룹당 450 라운드로 순환 갱신, 나머지 그룹은 최근 예측함수에 고정. 동결 xgboost 레시피(900R 환산), K=2.

## 제거된 4종

| # | 기존 | 대체 | 상태 |
|---|---|---|---|
| ① | level head `y≥550` 4배 가중 | 전 행 비가중 NLL | 코드에서 사라짐 |
| ② | `clip(·,1,3000)` | 로그정규 지지집합 (0,∞) | 예측 최소 288.9 / 최대 896.4, 경계 도달 0셀 |
| ③ | `1{y≥550}` 이진 head | 잠재 혼합비 π(x) | 학습 라벨에 550 없음 |
| ④ | `(y≥550).any()` gap head | 성분별 (μ_k, σ_k) 전체행 추정 | 사건 블록 평균 상수 없음 |

학습된 성분(F1 b0): 평균 μ = 6.015 / 6.225 (410 / 505 km/s), σ = 0.081 / 0.059, π_fast = 0.650.
즉 550 km/s 컷이 근사하던 slow/fast 분할을 데이터가 스스로 잡았다.

## 점수 (보드, 제출 2026-11-01T09:48Z)

| 지표 | total | F1 | F2 | F3 |
|---|---:|---:|---:|---:|
| S (보드 키, 낮을수록 좋음) | **105.411** | 132.865 | 77.543 | 63.000 |
| TIMSE | 31,029.7 | 50,968.8 | 11,334.5 | 6,664.7 |
| RMSE | **64.735** | 80.899 | 58.573 | 51.622 |
| bias | −3.59 | −13.14 | −2.91 | +5.38 |

보드 순위 16/92. 같은 표에서 TOTAL RMSE는 상위 17개 중 최소급(1위 point4 66.441, 스택 detpool_all 64.432).
같은 실행의 구 채점기(F1/F2/F3 6,315 origin, 454,680행) 점수는 **MSE 4,229.4** (F1 6,522.1 / F2 3,525.6 / F3 2,666.5,
≥550 12,418.1)로, 구 규약 단일 arm 최고였던 `e2000fs_calib8_qxgb`(4,290.4)보다 1.4% 낮다.

## 판정

1. **①·②·③·④는 실제로 제거됐다.** 임계값·가중치·자르기가 학습·예측 경로 어디에도 없고, 예측은 양수 지지집합의
   혼합 기대값이다. `experiment.py --self-check`의 해석적 그래디언트는 유한차분과 5e-7까지 일치한다.
2. **사전등록 예상 1은 절반 채택/절반 반증.** total은 `point4`(구 규약 3,491)보다 좋아졌지만, ≥550(12,418)은
   `point4`(8,598)보다 나쁘다. 즉 "잠재혼합이 몸통 손실을 없애고 꼬리 회수는 유지한다"는 가설 중 **몸통 부분만 성립**했다.
   꼬리는 여전히 정보(500–600 밴드 판별력) 문제라는 7차 진단과 일치한다.
3. **남은 병목은 F1 편향(−13.1 km/s)이다.** 새 보드 키 S가 편향에 민감해, RMSE가 가장 낮은데도 순위는 16위다.
   4배 가중이 암묵적으로 보정하던 고활동기 편향이 비가중 NLL에서는 드러난 것으로, 다음 축은 가중이 아니라
   **편향을 만드는 정보**(F1 구간의 입력 분포 차이)다. 가중·게이팅으로 되돌리는 것은 이번 제거의 취지에 어긋난다.

## 재현

```bash
cd research/e2000fs_calib9_lnmix
python experiment.py --self-check                 # grad/Hessian 유한차분
python experiment.py --jobs 9 --threads 2 --k 2   # 9 블록 → results/predictions.parquet
python /home/t-lab01/sunrun/leaderboard/submit.py results/predictions.parquet results/meta.yaml --dry-run
```

산출: `results/{F1,F2,F3}_b{0,1,2}.npz|json`(블록별 학습 NLL 이력 포함), `results/predictions.parquet`(454,680행),
`results/check.json`, `results/meta.yaml`. 게시: `experiment.py` @ 565d53f, replay `scripts/seongeun/e2000fs_calib9_lnmix/755a3784….py`.
