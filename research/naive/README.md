# naive

- 작성자: `baseline`
- 원점 시각의 속도를 72시간 그대로 유지합니다.

## 점수 (공식 scorer `lead-timse-6h-v1`)

| 구간 | S | TIMSE | RMSE |
|---|---:|---:|---:|
| Total | 149.444 | 57,419.9 | 98.01 |
| F1 | 171.565 | 77,907.0 | 115.07 |
| F2 | 122.915 | 30,448.7 | 92.50 |
| F3 | 107.161 | 21,778.2 | 83.71 |

## 폴더 구성

- `experiment.py` — 유일한 스크립트입니다. `data-v2/raw/ACE/monthly-txt`의 SWEPAM 시간별 속도(상태와 무관하게 속도 > 0인 기록, 시간 격자, ffill)와 `data-v2/input/df.parquet`의 공식 query origin(F1/F2/F3, 6,315개)을 읽고, 예측하고, 공식 scorer로 채점합니다.
- `results/predictions.parquet`, `results/score.json` — 이 스크립트가 만든 예측과 점수입니다.
- `logs/run.jsonl` — 실행 로그입니다.
- 학습된 가중치는 없습니다(고정 공식이거나 원점마다 다시 적합).

## 재현

```bash
python research/naive/experiment.py
```

2026-11-09 재작성: 예전 재작성본은 없는 경로(`data-v2/model-inputs/leaderboard-store/…`)와 MSE 시절 grid를 썼습니다. 리더보드에 저장된 예측과 그 점수 기록은 `results/leaderboard/`에 있습니다.

## 리더보드 저장 예측과의 비교

이 스크립트의 예측은 저장 예측과 비트 단위로 같습니다(S 149.444312).
