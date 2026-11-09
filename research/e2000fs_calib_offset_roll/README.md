# e2000fs_calib_offset_roll

- 작성자: `seongeun`
- 리더보드 제출: 20261109T095603839Z (3 seed 등가중 평균, mean3)

calib_offset_roll 레시피에서 `suvi_t035_equatorial_area`, `strip_v650_dark`를 빼고 `v_lag24`, `v_lag48`, `v_lag72`를 더한 모델의 seed 42/43/44 평균입니다. 사용자 요청으로 원래 이름 `e2000fs_calib_offset_roll`로 게시됐고(2026-11-09), 그 전의 같은 이름 기록(2026-11-04 원 레시피 재실행, S 98.76)을 대체했습니다.

## 점수 (공식 scorer `lead-timse-6h-v1`, 낮을수록 좋음)

| 구간 | S | TIMSE | RMSE |
|---|---:|---:|---:|
| Total | 96.311 | 20,032.4 | 67.29 |
| F1 | 117.402 | 30,088.0 | 82.70 |
| F2 | 75.178 | 8,879.2 | 61.86 |
| F3 | 60.935 | 5,176.2 | 54.79 |

seed별 Total S: 42 = 96.253, 43 = 96.906, 44 = 96.211

## 폴더 구성

- `experiment.py` — 유일한 스크립트입니다. `data-v2/input/df.parquet` 읽기, 특징 ffill, 학습 라벨 ffill, 3 seed × F1/F2/F3 학습(prior + full, 24 lead 블록 3개 × 900 rounds), frozen offset, 추론, mean3, 공식 scorer 채점까지 모두 들어 있습니다.
- `weights/F{1,2,3}/s{42,43,44}.npz` — 시드×기간 체크포인트 하나씩입니다. prior/full 부스터 6개(UBJSON 바이트)와 frozen calibration을 담습니다. 로컬에만 두고 git에는 올리지 않습니다.
- `results/s{seed}/`, `results/mean3/` — 예측(`predictions.parquet`)과 점수(`score.json`). mean3의 `score.json`은 리더보드 store 기록입니다.
- `logs/*.jsonl` — `s{seed}.jsonl`(launch, config, provenance, 블록별 fit, 기간별 metadata, 완료), `mean3.jsonl`(게시 기록). 이관분은 `"source": "migrated"`로 표시합니다.

## 재현

```bash
CUDA_VISIBLE_DEVICES= python research/e2000fs_calib_offset_roll/experiment.py --threads 4                      # 전체 학습(seed당 약 1–2시간)
CUDA_VISIBLE_DEVICES= python research/e2000fs_calib_offset_roll/experiment.py --predict-only --out /tmp/check  # weights/로 추론만
```

## 이관 기록 (2026-11-09)

산출물은 재학습 없이 원래 run에서 옮겼습니다: research/lag24_48_72_confirm_20261109/runs/lag24_48_72_s4{2,3,4} (로컬·golab2). 확인한 내용:

- 새 데이터 경로(df.parquet + ffill)가 기존 입력 `df_ffill_trainy.parquet`의 학습·질의 배열과 F1–F3 모두 같습니다.
- 원본 코드와 이 `experiment.py`의 smoke 실행(F1, 2 rounds)에서 예측·offset·부스터 바이트가 같습니다(대표 레시피 3개).
- `--predict-only`로 `weights/`에서 추론한 seed 예측 3개와 mean3가 이관 예측과 비트 단위로 같고, mean3 S가 보드 값과 같습니다(차이 0).
