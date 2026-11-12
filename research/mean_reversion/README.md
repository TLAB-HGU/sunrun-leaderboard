# mean_reversion

- 작성자: `baseline`
- 마지막 값에서 648시간 이동평균으로 exp(-h/48)로 수렴합니다.

## 점수 (공식 scorer `lead-timse-6h-v1`)

| 구간 | S | TIMSE | RMSE |
|---|---:|---:|---:|
| Total | 134.733 | 52,648.6 | 81.87 |
| F1 | 160.206 | 76,994.4 | 97.33 |
| F2 | 107.340 | 24,733.3 | 76.84 |
| F3 | 89.683 | 15,259.6 | 68.87 |

## 폴더 구성

- `experiment.py` — 유일한 스크립트입니다. `data-v2/raw/ACE/monthly-txt`의 SWEPAM 시간별 속도(상태와 무관하게 속도 > 0인 기록, 시간 격자, ffill)와 `data-v2/input/df.parquet`의 공식 query origin(F1/F2/F3, 6,315개)을 읽고, 예측하고, 공식 scorer로 채점합니다.
- `results/predictions.parquet`, `results/score.json` — 이 스크립트가 만든 예측과 점수입니다.
- `logs/run.jsonl` — 실행 로그입니다.
- 학습된 가중치는 없습니다(고정 공식이거나 원점마다 다시 적합).

## 재현

```bash
python research/mean_reversion/experiment.py
```

2026-11-09 재작성: 예전 재작성본은 없는 경로(`data-v2/model-inputs/leaderboard-store/…`)와 MSE 시절 grid를 썼습니다. 리더보드에 저장된 예측과 그 점수 기록은 `results/leaderboard_*`에 있습니다.

## 리더보드 저장 예측과의 비교

rolling 평균의 부동소수 오차로 셀 88,300개가 최대 1.1e-13 km/s 다르고, S는 같습니다(134.732711).

## 2026-11-12 폴더 정리

사용자 지시로 repo를 보드 행 전용 레이아웃으로 정리했습니다. 이 폴더에는 `experiment.py`(유일한 스크립트), `README.md`, `logs/`, `results/`, `weights/`(로컬 전용)만 있고 하위 폴더는 없습니다. 결과는 `results/{tag}_predictions.parquet`·`results/{tag}_score.json`, 체크포인트는 `weights/F{n}_s{seed}.npz`입니다. 위 본문이 가리키는 캠페인 폴더(`research/*_2026…`, `analysis/…`, `runs/…`)는 삭제되어 이 폴더의 사본만 남아 있습니다.
