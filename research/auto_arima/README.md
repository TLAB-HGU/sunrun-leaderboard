# auto_arima

- 작성자: `baseline`
- statsforecast AutoARIMA(비계절, stepwise, AICc)를 원점마다 직전 648시간에 다시 적합합니다.

## 점수 (공식 scorer `lead-timse-6h-v1`)

| 구간 | S | TIMSE | RMSE |
|---|---:|---:|---:|
| Total | 148.872 | 56,804.7 | 97.76 |
| F1 | 170.857 | 77,249.1 | 114.68 |
| F2 | 122.358 | 30,233.8 | 92.03 |
| F3 | 106.978 | 21,452.3 | 84.01 |

## 폴더 구성

- `experiment.py` — 유일한 스크립트입니다. `data-v2/raw/ACE/monthly-txt`의 SWEPAM 시간별 속도(상태와 무관하게 속도 > 0인 기록, 시간 격자, ffill)와 `data-v2/input/df.parquet`의 공식 query origin(F1/F2/F3, 6,315개)을 읽고, 예측하고, 공식 scorer로 채점합니다.
- `results/predictions.parquet`, `results/score.json` — 이 스크립트가 만든 예측과 점수입니다.
- `logs/run.jsonl` — 실행 로그입니다.
- 학습된 가중치는 없습니다(고정 공식이거나 원점마다 다시 적합).

실행 환경: `leaderboard/requirements-baselines.txt`(statsforecast 2.0.3, numba 0.67.0). 공용 환경의 statsforecast 2.1.1에서는 예측이 달라집니다.

## 재현

```bash
python research/auto_arima/experiment.py
```

2026-11-09 재작성: 예전 재작성본은 없는 경로(`data-v2/model-inputs/leaderboard-store/…`)와 MSE 시절 grid를 썼습니다. 리더보드에 저장된 예측과 그 점수 기록은 `results/leaderboard_*`에 있습니다.

## 리더보드 저장 예측과의 비교

셀 432개가 다릅니다(최대 10.52 km/s). S는 이 스크립트 148.872091, 보드 148.863193(차이 0.0089)입니다. 입력 시계열은 같은 시계열을 쓰는 naive·seasonal_naive_648이 비트 단위로, mean_reversion이 1e-13 이내로 일치합니다. 고정 환경(statsforecast 2.0.3)에서 Python 3.12.3과 3.12.13 모두 같은 결과였으므로, 차이는 원 제출 당시 머신의 statsforecast 적합 수치 차이로 보입니다. 보드 점수는 `results/leaderboard_*`의 저장 예측 기준입니다.

## 2026-11-12 폴더 정리

사용자 지시로 repo를 보드 행 전용 레이아웃으로 정리했습니다. 이 폴더에는 `experiment.py`(유일한 스크립트), `README.md`, `logs/`, `results/`, `weights/`(로컬 전용)만 있고 하위 폴더는 없습니다. 결과는 `results/{tag}_predictions.parquet`·`results/{tag}_score.json`, 체크포인트는 `weights/F{n}/s{seed}.npz`입니다. 위 본문이 가리키는 캠페인 폴더(`research/*_2026…`, `analysis/…`, `runs/…`)는 삭제되어 이 폴더의 사본만 남아 있습니다.

## 2026-11-13 레이아웃 변경 (중복 제거)

- 공용 코드를 `research/_module/`로 옮겼습니다. 이 행이 쓰는 모듈: `baseline`.
- `results/`와 `logs/`는 `research/result/auto_arima/`로 옮겼습니다. 위 본문에 나오는 `results/…`, `logs/…` 경로는 이 위치 기준입니다. `weights/`와 `visualization/`은 이 폴더에 그대로 있습니다.
- `--out`을 주지 않으면 `research/result/auto_arima/`에 씁니다. `--out DIR`을 주면 예전과 같이 `DIR/results`, `DIR/logs`, `DIR/weights`에 씁니다(재현 스크립트 호환).
- 검증: 재실행한 `predictions.parquet`이 기존과 완전히 같고 S도 같습니다. 예전 생성본 `leaderboard_predictions.parquet`·`leaderboard_score.json`은 삭제했습니다. ses·auto_arima 재현 환경은 `docs/cleanup-20261113.md`에 기록했습니다.
