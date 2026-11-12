# e2000fs_calib_offset_roll_drop_t035_v650_mean3

- 작성자: `seongeun`
- 리더보드 제출: 20261108T050039479Z (3 seed 등가중 평균, mean3)

calib_offset_roll 레시피에서 `suvi_t035_equatorial_area`, `strip_v650_dark`를 뺀 28개 특징 모델의 seed 42/43/44 평균입니다.

## 점수 (공식 scorer `lead-timse-6h-v1`, 낮을수록 좋음)

| 구간 | S | TIMSE | RMSE |
|---|---:|---:|---:|
| Total | 97.695 | 21,075.4 | 67.52 |
| F1 | 120.421 | 33,091.5 | 82.89 |
| F2 | 74.443 | 8,519.6 | 61.90 |
| F3 | 60.538 | 4,991.5 | 55.26 |

seed별 Total S: 42 = 97.492, 43 = 97.874, 44 = 98.075

## 폴더 구성

- `experiment.py` — 유일한 스크립트입니다. `data-v2/input/df.parquet` 읽기, 특징 ffill, 학습 라벨 ffill, 3 seed × F1/F2/F3 학습(prior + full, 24 lead 블록 3개 × 900 rounds), frozen offset, 추론, mean3, 공식 scorer 채점까지 모두 들어 있습니다.
- `weights/F{1,2,3}_s{42,43,44}.npz` — 시드×기간 체크포인트 하나씩입니다. prior/full 부스터 6개(UBJSON 바이트)와 frozen calibration을 담습니다. 로컬에만 두고 git에는 올리지 않습니다.
- `results/s{seed}_*`, `results/mean3_*` — 예측(`*_predictions.parquet`)과 점수(`*_score.json`). `mean3_score.json`은 리더보드 store 기록입니다.
- `logs/*.jsonl` — `s{seed}.jsonl`(launch, config, provenance, 블록별 fit, 기간별 metadata, 완료), `mean3.jsonl`(게시 기록). 이관분은 `"source": "migrated"`로 표시합니다.

## 재현

```bash
CUDA_VISIBLE_DEVICES= python research/e2000fs_calib_offset_roll_drop_t035_v650/experiment.py --threads 4                      # 전체 학습(seed당 약 1–2시간)
CUDA_VISIBLE_DEVICES= python research/e2000fs_calib_offset_roll_drop_t035_v650/experiment.py --predict-only --out /tmp/check  # weights/로 추론만
```

## 이관 기록 (2026-11-09)

산출물은 재학습 없이 원래 run에서 옮겼습니다: research/drop035_v650_confirm_20261108/runs/drop035_v650_s4{2,3,4} (s42 로컬, s43·s44 golab2). 확인한 내용:

- 새 데이터 경로(df.parquet + ffill)가 기존 입력 `df_ffill_trainy.parquet`의 학습·질의 배열과 F1–F3 모두 같습니다.
- 원본 코드와 이 `experiment.py`의 smoke 실행(F1, 2 rounds)에서 예측·offset·부스터 바이트가 같습니다(대표 레시피 3개).
- `--predict-only`로 `weights/`에서 추론한 seed 예측 3개와 mean3가 이관 예측과 비트 단위로 같고, mean3 S가 보드 값과 같습니다(차이 0).

## 2026-11-12 폴더 정리

사용자 지시로 repo를 보드 행 전용 레이아웃으로 정리했습니다. 이 폴더에는 `experiment.py`(유일한 스크립트), `README.md`, `logs/`, `results/`, `weights/`(로컬 전용)만 있고 하위 폴더는 없습니다. 결과는 `results/{tag}_predictions.parquet`·`results/{tag}_score.json`, 체크포인트는 `weights/F{n}_s{seed}.npz`입니다. 위 본문이 가리키는 캠페인 폴더(`research/*_2026…`, `analysis/…`, `runs/…`)는 삭제되어 이 폴더의 사본만 남아 있습니다.
폴더 이름은 보드 표시 이름(`e2000fs_calib_offset_roll_drop_t035_v650`)과 같게 바꿨고, 제출 ID는 `e2000fs_calib_offset_roll_drop_t035_v650_mean3` 그대로입니다(`experiment.py`의 `EXPERIMENT`).
