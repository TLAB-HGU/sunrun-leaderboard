# e2000fs_calib_offset_roll_splus

- 작성자: `seongeun`
- 리더보드: 2026-11-12 사용자 요청으로 게시 (3 seed 등가중 평균, mean3)

point4 + spectral 판 `e2000fs_calib_offset_roll` 레시피(A900, git e6dde5e1a; `suvi_t035_equatorial_area`·`strip_v650_dark` 제거, `v_lag24/48/72` 추가, prior 모델 + frozen per-lead offset)를 먼저 900 rounds 학습합니다. 그다음 각 부스터를 **학습셋 S + 같은 0.1 spectral 항**으로 180 tree 더 학습합니다(S+).

- 6 stage × 30 tree, stage seed = seed + stage
- S 계수(MSE·TIMSE 가중)는 매 round 다시 계산
- stage마다 정확한 Armijo line search(c = 1e-4, α ∈ 1, 1/2, …, 1/64, 아니면 0)로 수용
- offset은 continuation이 끝난 prior 모델로 같은 알고리즘(마지막 336 calibration origin, α 0.25)에서 다시 적합

모든 계산은 학습행만 씁니다. 공식 정답은 채점에만 씁니다.

## 점수 (공식 scorer `lead-timse-6h-v1`, 낮을수록 좋음)

| 구간 | S | A900 mean3 S | TIMSE | RMSE |
|---|---:|---:|---:|---:|
| Total | **95.516** | 96.311 | 18,280.3 | 69.46 |
| F1 | 116.781 | 117.402 | 27,388.7 | 85.97 |
| F2 | 74.187 | 75.178 | 8,095.3 | 63.14 |
| F3 | 61.149 | 60.935 | 4,950.6 | 56.52 |

seed별 Total S (A900 대비): 42 = 95.594 (96.253, −0.660), 43 = 96.161 (96.906, −0.746), 44 = 95.288 (96.211, −0.923)

- 세 seed 모두 좋아졌고 mean3 S는 0.83 % 낮아졌습니다.
- 대신 RMSE가 3.2 % 나빠졌습니다(67.29 → 69.46). TIMSE는 8.7 % 줄었습니다.
- 사전 게이트(3 seed 평균 1 % 개선, RMSE 악화 ≤ 1 %)는 통과하지 못했습니다. 사용자 요청으로 게시한 행입니다.
- 분석과 사전등록은 `research/loss_mechanism_ablation_20261112/strategy.md`, `analysis/report.md`에 있습니다.

## 폴더 구성

- `experiment.py` — 유일한 스크립트입니다. `data-v2/input/df.parquet` 읽기, 특징 ffill, 학습 라벨 ffill, 3 seed × F1/F2/F3의 A900 학습과 S continuation, frozen offset, clip, mean3, 공식 scorer 채점까지 모두 들어 있습니다.
- `weights/F{1,2,3}_s{42,43,44}.npz` — 시드×기간 체크포인트입니다. A900 부스터, stage 부스터, 수용된 α, frozen calibration이 들어 있습니다. 로컬에만 두고 git에는 올리지 않습니다.
- `results/s{seed}_*`, `results/mean3_*` — 예측(`*_predictions.parquet`)과 점수(`*_score.json`)입니다.
- `logs/train_s*.jsonl`, `logs/train_run_s*.jsonl` — 학습 run의 진행 기록, 기간별 metadata, 블록별 continuation 기록(stage별 α·목표값·학습 MSE), replay 결과입니다.
- `logs/{run,data,s*,mean3}.jsonl` — weights 추론과 mean3 기록입니다.

## 재현

```bash
CUDA_VISIBLE_DEVICES= python research/e2000fs_calib_offset_roll_splus/experiment.py --threads 5                      # 전체 학습
CUDA_VISIBLE_DEVICES= python research/e2000fs_calib_offset_roll_splus/experiment.py --predict-only --out /tmp/check  # weights/로 추론만
```

## 검증 기록 (2026-11-12)

- 학습은 `research/loss_mechanism_ablation_20261112/runs/splus_s{42,43,44}`에서 했습니다(exit 0, 18블록 replay 오차 0, 공식 채점 누락 대체 0). weights와 학습 로그를 이 폴더로 옮겼습니다.
- continuation의 시작점인 A900 부스터 54개는 이전 point4 + spectral 보드 행의 weights와 바이트 단위로 같습니다.
- 이 파일의 학습 경로로 `df.parquet`에서 seed 42 F1 full b0를 다시 continuation하면 stage 부스터 6개가 학습 run과 바이트 단위로 같고 α도 같습니다.
- `--smoke`(2 rounds) exit 0.
- `--predict-only`로 weights에서 만든 seed 예측 3개가 학습 run 예측과 비트 단위로 같고, mean3 S는 95.516입니다.

## 2026-11-12 폴더 정리

사용자 지시로 repo를 보드 행 전용 레이아웃으로 정리했습니다. 이 폴더에는 `experiment.py`(유일한 스크립트), `README.md`, `logs/`, `results/`, `weights/`(로컬 전용)만 있고 하위 폴더는 없습니다. 결과는 `results/{tag}_predictions.parquet`·`results/{tag}_score.json`, 체크포인트는 `weights/F{n}_s{seed}.npz`입니다. 위 본문이 가리키는 캠페인 폴더(`research/*_2026…`, `analysis/…`, `runs/…`)는 삭제되어 이 폴더의 사본만 남아 있습니다.

## 2026-11-13 레이아웃 변경 (중복 제거)

- 공용 코드를 `research/module/`로 옮겼습니다. 이 행이 쓰는 모듈: `continuation`, `gbm`, `objectives`. `experiment.py`에는 특징 목록, 목적함수, 보정 설정을 담은 `gbm.GBMSpec`만 남았습니다.
- `results/`와 `logs/`는 `research/result/e2000fs_calib_offset_roll_splus/`로 옮겼습니다. 위 본문에 나오는 `results/…`, `logs/…` 경로는 이 위치 기준입니다. `weights/`와 `visualization/`은 이 폴더에 그대로 있습니다.
- `--out`을 주지 않으면 `research/result/e2000fs_calib_offset_roll_splus/`에 씁니다. `--out DIR`을 주면 예전과 같이 `DIR/results`, `DIR/logs`, `DIR/weights`에 씁니다(재현 스크립트 호환).
- 검증: 새 코드의 `--predict-only`(weights/) 예측이 기존 `results/`의 s42·s43·s44·mean3와 완전히 같고 S도 같습니다. old/new `--smoke` 예측과 checkpoint 배열(부스터 바이트·calibration)도 모두 같습니다.
