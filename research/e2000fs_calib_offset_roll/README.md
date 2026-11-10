# e2000fs_calib_offset_roll

- 작성자: `seongeun`
- 리더보드: 2026-11-10 사용자 요청으로 spectral 제거판으로 교체 (3 seed 등가중 평균, mean3)

calib_offset_roll 레시피(`suvi_t035_equatorial_area`·`strip_v650_dark` 제거, `v_lag24/48/72` 추가, prior 모델 + frozen per-lead offset)을 **point4 고속 구간 가중 제곱오차만으로** 학습한 seed 42/43/44 평균입니다. 목적함수의 spectral(FFT) 항(계수 0.1, Hessian +0.1)을 제거했고, 나머지 특징·하이퍼파라미터·seed·학습 cutoff는 이전 행과 같습니다.

## 점수 (공식 scorer `lead-timse-6h-v1`, 낮을수록 좋음)

| 구간 | S | 이전 행 S (spectral 0.1) | TIMSE | RMSE |
|---|---:|---:|---:|---:|
| Total | 97.109 | 96.311 | 20,536.7 | 67.53 |
| F1 | 118.795 | 117.402 | 31,070.1 | 83.20 |
| F2 | 75.413 | 75.178 | 8,971.9 | 61.97 |
| F3 | 61.149 | 60.935 | 5,235.6 | 54.81 |

seed별 Total S (이전 행 대비): 42 = 97.104 (96.253, +0.851), 43 = 97.136 (96.906, +0.230), 44 = 97.572 (96.211, +1.361)

spectral 항을 빼면 3 seed 모두 나빠집니다(사전 기준 "spectral 유지 근거" 충족). 악화는 주로 F1에서 나옵니다. 이 행은 성능 개선이 아니라 사용자 요청에 따른 교체입니다. 이전 행의 코드는 git 기록에, 이전 weights는 `research/e2000fs_calib_offset_roll_no_spectral_20261110/replaced_board_row/weights/`(로컬)에 있습니다. 분석과 사전등록은 `research/e2000fs_calib_offset_roll_no_spectral_20261110/strategy.md`에 있습니다.

## 폴더 구성

- `experiment.py` — 유일한 스크립트입니다. `data-v2/input/df.parquet` 읽기, 특징 ffill, 학습 라벨 ffill, 3 seed × F1/F2/F3 학습, frozen offset, clip, mean3, 공식 scorer 채점까지 모두 들어 있습니다.
- `weights/F{1,2,3}/s{42,43,44}.npz` — 시드×기간 체크포인트입니다. 로컬에만 두고 git에는 올리지 않습니다.
- `results/s{seed}/`, `results/mean3/` — 예측(`predictions.parquet`)과 점수(`score.json`)입니다.
- `logs/train_*.jsonl` — 학습 run의 기록(launch, data, 블록별 fit, 기간별 metadata), `logs/{run,data,s*,mean3}.jsonl` — weights 추론과 mean3 기록입니다.

## 재현

```bash
CUDA_VISIBLE_DEVICES= python research/e2000fs_calib_offset_roll/experiment.py --threads 4                      # 전체 학습
CUDA_VISIBLE_DEVICES= python research/e2000fs_calib_offset_roll/experiment.py --predict-only --out /tmp/check  # weights/로 추론만
```

## 검증 기록 (2026-11-10)

- 학습은 `research/e2000fs_calib_offset_roll_no_spectral_20261110/runs/no_spectral_s{42,43,44}`에서 했고(exit 0, 기간별 metadata, 예측, 점수), weights와 학습 로그를 이 폴더로 옮겼습니다.
- 학습 전 smoke(F1, 2 rounds)에서 이전 코드에 `SPECTRAL_COEFFICIENT = 0.`만 준 것과 예측·부스터 바이트가 같았습니다.
- `--predict-only`로 weights에서 만든 seed 예측 3개가 학습 run 예측과 비트 단위로 같고, mean3 S는 97.109입니다.
