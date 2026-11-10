# e2000fs_no_calib_offset_roll

- 작성자: `seongeun`
- 리더보드 제출: 20261110T054617020Z (3 seed 등가중 평균, mean3)

`e2000fs_calib_offset_roll` 레시피에서 **캘리브레이션(prior 모델 + frozen per-lead offset)만 제거**한 모델의 seed 42/43/44 평균입니다. 예측은 full 모델의 query 예측을 [1, 3000]으로 clip한 값입니다. 특징, 손실(tail weight 4, spectral 0.1), 하이퍼파라미터, seed, 학습 cutoff는 기준선과 같습니다.

## 점수 (공식 scorer `lead-timse-6h-v1`, 낮을수록 좋음)

| 구간 | S | 기준선 S | TIMSE | RMSE |
|---|---:|---:|---:|---:|
| Total | 97.254 | 96.311 | 21,063.1 | 66.96 |
| F1 | 118.326 | 117.402 | 32,499.5 | 80.70 |
| F2 | 74.957 | 75.178 | 8,554.2 | 62.60 |
| F3 | 60.853 | 60.935 | 4,999.8 | 55.64 |

seed별 Total S (기준선 대비): 42 = 97.148 (96.253, +0.895), 43 = 97.872 (96.906, +0.966), 44 = 97.181 (96.211, +0.970)

기준선 mean3은 S 96.311, TIMSE 20,032.4, RMSE 67.29입니다. 캘리브레이션을 빼면 Total S가 +0.943(+0.98%) 나빠집니다. 3 seed 모두 같은 방향이고 최악 seed 차이는 +0.970입니다. 악화는 F1에서 나옵니다(+0.924). F2, F3는 오히려 약간 좋아집니다(−0.221, −0.082). RMSE는 기준선보다 낮습니다(66.96 대 67.29). 즉 offset은 TIMSE, 특히 F1을 줄이는 효과입니다.

## 폴더 구성

- `experiment.py` — 유일한 스크립트입니다. `data-v2/input/df.parquet` 읽기, 특징 ffill, 학습 라벨 ffill, 3 seed × F1/F2/F3 학습(full 모델, 24 lead 블록 3개 × 900 rounds), clip, mean3, 공식 scorer 채점까지 모두 들어 있습니다.
- `strategy.md` — 사전등록(가설, 기준선, 변경, 판정 기준)과 결과 기록입니다.
- `weights/F{1,2,3}/s{42,43,44}.npz` — 시드×기간 체크포인트(full 부스터 3개, UBJSON 바이트)입니다. 로컬에만 두고 git에는 올리지 않습니다.
- `results/s{seed}/`, `results/mean3/` — 예측(`predictions.parquet`)과 점수(`score.json`)입니다.
- `logs/*.jsonl` — `s{seed}.jsonl`(launch, config, provenance, 블록별 fit, 기간별 metadata, 완료).

## 재현

```bash
CUDA_VISIBLE_DEVICES= python research/e2000fs_no_calib_offset_roll/experiment.py --threads 4                      # 전체 학습(seed당 약 40분)
CUDA_VISIBLE_DEVICES= python research/e2000fs_no_calib_offset_roll/experiment.py --predict-only --out /tmp/check  # weights/로 추론만
```

## 검증 기록 (2026-11-10)

- 3 seed 모두 `train.exit` 0, 기간별 메타데이터, 예측, 점수가 기록됐습니다. 학습은 `research/e2000fs_no_calib_offset_roll_20261110/runs/s{42,43,44}`에서 했고 산출물을 이 폴더로 옮겼습니다.
- 새로 학습한 full 부스터 18개(3 seed × 3 기간 × 3 블록)가 기준선 `weights/`의 `full_b*`와 바이트 단위로 같습니다. 따라서 이 예측은 기준선의 offset 이전 예측과 같고, 점수 차이는 offset 하나만으로 생깁니다.
- `--predict-only`로 weights에서 만든 seed 예측 3개가 학습 직후 예측과 같습니다.
