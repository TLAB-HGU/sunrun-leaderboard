# current_offset_roll_direct_s

- 작성자: `seongeun`
- 2026-11-12 사용자 요청에 따라 `current_offset_roll_direct_s`로 리더보드 게시 완료. 원 실험은 `current_offset_roll_direct_s_20261109`의 B군이다.
- seed 42/43/44의 고정 등가중 평균이며, 가중치를 적합하거나 최고 seed를 선택하지 않는다.

## 모델과 손실함수

31 source / 215 expanded 특징, 24-lead 블록 3개, 각 900-round XGBoost 모델을 **처음부터 direct S만으로** 학습한다. 손실은 학습행에서 lead별 `S_h = (MSE_h × TIMSE_h)^0.25`를 합한 값이다. TIMSE 계수는 공식 100-quantile 정의와 동일하게 학습 정답 분포에서 계산한다. 블록별 초기 상수 예측에서 평균 곡률 proxy가 1이 되도록 고정 배율을 정한다. 매 round S gradient와 양의 residual coefficient를 갱신하며, 후자는 결합 S의 정확한 Hessian이 아닌 대각 곡률 proxy다.

Tail-weighted MSE의 550 km/s·4배 규칙과 spectral 항은 사용하지 않는다. 원 B군의 legacy config에 남아 있던 `tail_weight=4`, `spectral=True`는 실제 callback을 표현하지 않았다. 이 스크립트에는 해당 분기와 상수를 넣지 않았다.

특징 ffill·학습정답 ffill, 학습 cutoff와 144h maturity, prior/full 분리, 학습행 마지막 336 calibration origin의 평균 잔차 ×0.25로 계산한 frozen per-lead offset, [1,3000] clip은 원 실험과 같다. 공식 평가 정답은 채점에만 사용한다. S+처럼 기존 모델에서 전환 학습하는 구조가 아니다.

## 결과

공식 scorer `lead-timse-6h-v1`. 아래는 세 seed의 **예측 평균을 채점한 값**이다. 원 분석 보고서의 seed 점수 평균 97.718929와 구분한다.

| 구간 | S | TIMSE | RMSE |
|---|---:|---:|---:|
| Total | 97.530229 | 19,317.045171 | 70.329355 |
| F1 | 120.054690 | 30,165.057070 | 86.371334 |
| F2 | 75.398603 | 8,191.422429 | 64.928419 |
| F3 | 62.472017 | 5,243.913263 | 57.089363 |

seed별 Total S: 42=97.994338, 43=97.786235, 44=97.376213. 기존 point4+spectral 기준선보다 악화된 실험이며, 개선을 검증해 채택한 모델이 아니라 사용자 요청에 따라 공개하는 결과다.

## 재현

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/current_offset_roll_direct_s/experiment.py --predict-only --threads 2
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/current_offset_roll_direct_s/experiment.py --threads 4 --out /tmp/direct-s-retrain
```

`--data` 기본값은 `data-v2/input/df.parquet`이다. 별도 출력으로 추론하려면 `--weights research/current_offset_roll_direct_s/weights --out /tmp/direct-s-replay`를 지정한다. 전체 학습 재현은 기존 데이터와 라이브러리 버전을 사용한다(원 실행: Python 3.12.13, NumPy 2.4.6, pandas 2.3.3, XGBoost 2.1.4).

- `experiment.py`: 데이터 준비, 순수 direct-S 학습, 저장 모델 추론, offset, mean3, 공식 채점을 포함하는 단일 architecture 스크립트.
- `weights/F*_s*.npz`: 기존 54개 prior/full booster와 9개 calibration state. 로컬 재현용이며 원 학습 weights를 다시 학습하지 않고 포장했다.
- `results/s*_*`, `results/mean3_*`: 예측 parquet와 공식 score JSON.
- `logs/provenance.json`: 원 실행 위치, 특징 목록, 기간, 하이퍼파라미터, 라이브러리 버전과 추론 시간.
- `logs/verification.json`: 목적함수 수치 일치, 예측·부스터·offset 일치와 최종 점수.

## 검증

순수 S callback의 값, float32 gradient와 곡률 proxy가 원본 B callback과 정확히 일치한다(독립 난수 입력 3개 크기). 원 raw df에서 `--predict-only` exit 0. seed별 454,680개 예측이 원 B군의 F1/F2/F3 NPZ와 비트 단위로 같고, 저장한 54개 booster byte 배열 및 9개 frozen calibration state도 원본과 정확히 같다. seed별 공식 S도 원 점수와 정확히 같으며 모든 점수의 persistence 대체 수는 0이다. `py_compile` 통과.

이번 게시 준비에서 새 학습은 실행하지 않았다. 전체 학습 경로의 새 end-to-end 실행은 검증하지 않았으며, 실제 게시 예측은 검증된 기존 B weights에서 재현했다. 처음 추론 시 observations 경로 오타로 채점 전에 실패했고, 원 캠페인의 고정 HF snapshot 관측 경로를 사용해 정상 완료했다.

## 전략 문서 (이전 strategy.md)

## Publication packaging plan

2026-11-12: user requested publication as `current_offset_roll_direct_s`.
The selected architecture is arm B (direct S only) from
`research/current_offset_roll_direct_s_20261109`, seeds 42/43/44.
This is packaging of completed runs, not a new hypothesis or training run.

Preserve all 31 source / 215 expanded features, data fill, cutoff, seed,
900-round model configuration, prior/full split, clipping and frozen offset.
Remove unrelated objective branches from the self-contained script. Reuse the
saved 54 boosters and nine frozen calibration states. Verify objective values,
gradients and curvature proxies against the original pure-S callback before
inference; then require bitwise equality to all nine saved prediction matrices.
Create the fixed arithmetic mean of all three seeds without fitting weights,
and score it with the official scorer and frozen official observations.
No new model fitting is authorized for this packaging task.

### Packaging outcome

Pure-S value/gradient/curvature parity passed on three independent finite matrices.
All 54 booster byte arrays and nine frozen calibration states are unchanged.
Raw-table inference reproduced all three seed forecasts exactly, with original
seed scores unchanged. Equal-weight mean3 Total S is 97.53022862821581,
with no persistence replacements. Full training was not rerun. See
verification.json and provenance.json for detailed evidence. Publication itself
is handled by the leader after this local package handoff.

## 2026-11-12 폴더 정리

사용자 지시로 repo를 보드 행 전용 레이아웃으로 정리했습니다. 이 폴더에는 `experiment.py`(유일한 스크립트), `README.md`, `logs/`, `results/`, `weights/`(로컬 전용)만 있고 하위 폴더는 없습니다. 결과는 `results/{tag}_predictions.parquet`·`results/{tag}_score.json`, 체크포인트는 `weights/F{n}_s{seed}.npz`입니다. 위 본문이 가리키는 캠페인 폴더(`research/*_2026…`, `analysis/…`, `runs/…`)는 삭제되어 이 폴더의 사본만 남아 있습니다.

## 2026-11-13 레이아웃 변경 (중복 제거)

- 공용 코드를 `research/_module/`로 옮겼습니다. 이 행이 쓰는 모듈: `gbm`, `objectives`. `experiment.py`에는 특징 목록, 목적함수, 보정 설정을 담은 `gbm.GBMSpec`만 남았습니다.
- `results/`와 `logs/`는 `research/result/current_offset_roll_direct_s/`로 옮겼습니다. 위 본문에 나오는 `results/…`, `logs/…` 경로는 이 위치 기준입니다. `weights/`와 `visualization/`은 이 폴더에 그대로 있습니다.
- `--out`을 주지 않으면 `research/result/current_offset_roll_direct_s/`에 씁니다. `--out DIR`을 주면 예전과 같이 `DIR/results`, `DIR/logs`, `DIR/weights`에 씁니다(재현 스크립트 호환).
- 검증: 새 코드의 `--predict-only`(weights/) 예측이 기존 `results/`의 s42·s43·s44·mean3와 완전히 같고 S도 같습니다. old/new `--smoke` 예측과 checkpoint 배열(부스터 바이트·calibration)도 모두 같습니다.
