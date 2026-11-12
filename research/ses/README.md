# ses

- 작성자: `baseline`
- statsforecast SimpleExponentialSmoothingOptimized를 원점마다 직전 648시간에 다시 적합합니다.

## 점수 (공식 scorer `lead-timse-6h-v1`)

| 구간 | S | TIMSE | RMSE |
|---|---:|---:|---:|
| Total | 149.106 | 57,377.2 | 97.61 |
| F1 | 171.544 | 77,889.2 | 115.06 |
| F2 | 122.814 | 30,427.0 | 92.41 |
| F3 | 106.169 | 21,633.6 | 82.52 |

## 폴더 구성

- `experiment.py` — 유일한 스크립트입니다. `data-v2/raw/ACE/monthly-txt`의 SWEPAM 시간별 속도(상태와 무관하게 속도 > 0인 기록, 시간 격자, ffill)와 `data-v2/input/df.parquet`의 공식 query origin(F1/F2/F3, 6,315개)을 읽고, 예측하고, 공식 scorer로 채점합니다.
- `results/predictions.parquet`, `results/score.json` — 이 스크립트가 만든 예측과 점수입니다.
- `logs/run.jsonl` — 실행 로그입니다.
- 학습된 가중치는 없습니다(고정 공식이거나 원점마다 다시 적합).

실행 환경: `leaderboard/requirements-baselines.txt`(statsforecast 2.0.3, numba 0.67.0). 공용 환경의 statsforecast 2.1.1에서는 예측이 달라집니다.

## 재현

```bash
python research/ses/experiment.py
```

2026-11-09 재작성: 예전 재작성본은 없는 경로(`data-v2/model-inputs/leaderboard-store/…`)와 MSE 시절 grid를 썼습니다. 리더보드에 저장된 예측과 그 점수 기록은 `results/leaderboard_*`에 있습니다.

## 리더보드 저장 예측과의 비교

셀 106,560개가 다릅니다(최대 0.003347 km/s). S는 이 스크립트 149.105747, 보드 149.105749(차이 2.2e-06)입니다. 입력 시계열은 같은 시계열을 쓰는 naive·seasonal_naive_648이 비트 단위로, mean_reversion이 1e-13 이내로 일치합니다. 고정 환경(statsforecast 2.0.3)에서 Python 3.12.3과 3.12.13 모두 같은 결과였으므로, 차이는 원 제출 당시 머신의 statsforecast 적합 수치 차이로 보입니다. 보드 점수는 `results/leaderboard_*`의 저장 예측 기준입니다.

## 2026-11-12 폴더 정리

사용자 지시로 repo를 보드 행 전용 레이아웃으로 정리했습니다. 이 폴더에는 `experiment.py`(유일한 스크립트), `README.md`, `logs/`, `results/`, `weights/`(로컬 전용)만 있고 하위 폴더는 없습니다. 결과는 `results/{tag}_predictions.parquet`·`results/{tag}_score.json`, 체크포인트는 `weights/F{n}_s{seed}.npz`입니다. 위 본문이 가리키는 캠페인 폴더(`research/*_2026…`, `analysis/…`, `runs/…`)는 삭제되어 이 폴더의 사본만 남아 있습니다.
