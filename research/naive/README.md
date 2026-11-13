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

2026-11-09 재작성: 예전 재작성본은 없는 경로(`data-v2/model-inputs/leaderboard-store/…`)와 MSE 시절 grid를 썼습니다. 리더보드에 저장된 예측과 그 점수 기록은 `results/leaderboard_*`에 있습니다.

## 리더보드 저장 예측과의 비교

이 스크립트의 예측은 저장 예측과 비트 단위로 같습니다(S 149.444312).

## 2026-11-12 폴더 정리

사용자 지시로 repo를 보드 행 전용 레이아웃으로 정리했습니다. 이 폴더에는 `experiment.py`(유일한 스크립트), `README.md`, `logs/`, `results/`, `weights/`(로컬 전용)만 있고 하위 폴더는 없습니다. 결과는 `results/{tag}_predictions.parquet`·`results/{tag}_score.json`, 체크포인트는 `weights/F{n}_s{seed}.npz`입니다. 위 본문이 가리키는 캠페인 폴더(`research/*_2026…`, `analysis/…`, `runs/…`)는 삭제되어 이 폴더의 사본만 남아 있습니다.

## 2026-11-13 레이아웃 변경 (중복 제거)

- 공용 코드를 `research/_module/`로 옮겼습니다. 이 행이 쓰는 모듈: `baseline`.
- `results/`와 `logs/`는 `research/result/naive/`로 옮겼습니다. 위 본문에 나오는 `results/…`, `logs/…` 경로는 이 위치 기준입니다. `weights/`와 `visualization/`은 이 폴더에 그대로 있습니다.
- `--out`을 주지 않으면 `research/result/naive/`에 씁니다. `--out DIR`을 주면 예전과 같이 `DIR/results`, `DIR/logs`, `DIR/weights`에 씁니다(재현 스크립트 호환).
- 검증: 재실행한 `predictions.parquet`이 기존과 완전히 같고 S도 같습니다. 예전 생성본 `leaderboard_predictions.parquet`·`leaderboard_score.json`은 삭제했습니다.
