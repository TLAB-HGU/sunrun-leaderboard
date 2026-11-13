# e2000fs_no_calib_offset_roll

- 작성자: `seongeun`
- 리더보드: 2026-11-10 사용자 요청으로 spectral 제거판으로 교체 (3 seed 등가중 평균, mean3)

`e2000fs_calib_offset_roll` 레시피에서 캘리브레이션(prior 모델 + frozen offset)을 뺀 모델(full 모델 예측을 [1, 3000]으로 clip)을 **point4 고속 구간 가중 제곱오차만으로** 학습한 seed 42/43/44 평균입니다. 목적함수의 spectral(FFT) 항(계수 0.1, Hessian +0.1)을 제거했고, 나머지 특징·하이퍼파라미터·seed·학습 cutoff는 이전 행과 같습니다.

## 점수 (공식 scorer `lead-timse-6h-v1`, 낮을수록 좋음)

| 구간 | S | 이전 행 S (spectral 0.1) | TIMSE | RMSE |
|---|---:|---:|---:|---:|
| Total | 98.108 | 97.254 | 21,665.0 | 67.14 |
| F1 | 119.796 | 118.326 | 33,679.1 | 81.13 |
| F2 | 75.171 | 74.957 | 8,643.9 | 62.66 |
| F3 | 61.041 | 60.853 | 5,050.6 | 55.64 |

seed별 Total S (이전 행 대비): 42 = 98.045 (97.148, +0.897), 43 = 98.155 (97.872, +0.283), 44 = 98.607 (97.181, +1.426)

spectral 항을 빼면 3 seed 모두 나빠집니다(사전 기준 "spectral 유지 근거" 충족). 악화는 주로 F1에서 나옵니다. 이 행은 성능 개선이 아니라 사용자 요청에 따른 교체입니다. 이전 행의 코드는 git 기록에, 이전 weights는 `research/e2000fs_no_calib_offset_roll_no_spectral_20261110/replaced_board_row/weights/`(로컬)에 있습니다. 분석과 사전등록은 `research/e2000fs_no_calib_offset_roll_no_spectral_20261110/strategy.md`에 있습니다.

## 폴더 구성

- `experiment.py` — 유일한 스크립트입니다. `data-v2/input/df.parquet` 읽기, 특징 ffill, 학습 라벨 ffill, 3 seed × F1/F2/F3 학습, clip, mean3, 공식 scorer 채점까지 모두 들어 있습니다.
- `weights/F{1,2,3}_s{42,43,44}.npz` — 시드×기간 체크포인트입니다. 로컬에만 두고 git에는 올리지 않습니다.
- `results/s{seed}_*`, `results/mean3_*` — 예측(`*_predictions.parquet`)과 점수(`*_score.json`)입니다.
- `logs/train_*.jsonl` — 학습 run의 기록(launch, data, 블록별 fit, 기간별 metadata), `logs/{run,data,s*,mean3}.jsonl` — weights 추론과 mean3 기록입니다.

## 재현

```bash
CUDA_VISIBLE_DEVICES= python research/e2000fs_no_calib_offset_roll/experiment.py --threads 4                      # 전체 학습
CUDA_VISIBLE_DEVICES= python research/e2000fs_no_calib_offset_roll/experiment.py --predict-only --out /tmp/check  # weights/로 추론만
```

## 검증 기록 (2026-11-10)

- 학습은 `research/e2000fs_no_calib_offset_roll_no_spectral_20261110/runs/nocalib_no_spectral_s{42,43,44}`에서 했고(exit 0, 기간별 metadata, 예측, 점수), weights와 학습 로그를 이 폴더로 옮겼습니다.
- 학습 전 smoke(F1, 2 rounds)에서 이전 코드에 `SPECTRAL_COEFFICIENT = 0.`만 준 것과 예측·부스터 바이트가 같았습니다.
- `--predict-only`로 weights에서 만든 seed 예측 3개가 학습 run 예측과 비트 단위로 같고, mean3 S는 98.108입니다.

## 전략 문서 (이전 strategy.md)

## e2000fs_no_calib_offset_roll — 캘리브레이션(frozen offset) 제거 ablation

> 2026-11-10 이후: 이 board 행은 사용자 요청으로 spectral 항을 제거한 버전(S 98.108)으로 교체됐습니다. 아래는 캘리브레이션 제거 실험(spectral 0.1, S 97.254)의 원래 사전등록과 결과 기록입니다. 교체 근거는 `research/e2000fs_no_calib_offset_roll_no_spectral_20261110/strategy.md`에 있습니다.

작성 2026-11-10. 사용자 요청: 현재 보드 행 `e2000fs_calib_offset_roll`에서 캘리브레이션을 제거한 모델을 3 seed 전부 학습해 `_mean3`로 리더보드에 올린다.

### 배경과 문헌
- 과거 기록: `e2000fs_calib_offset_roll_abl1_loss_t4_sp1_nooffset_s42`(seed 42 하나, 원 레시피, 같은 모델 예측 재사용)에서 offset 제거 시 Total S 98.763 → 99.751(+0.99). 단일 seed라 seed 노이즈(보드 행 seed 간 최대 약 1.4)와 구분되지 않았다.
- 이번 실험은 현재 보드 행(lag24/48/72 구성)에서 3 seed를 모두 새로 학습해 그 질문을 닫는다. 새 문헌 주장은 하지 않는다. 캘리브레이션 메커니즘은 기준선 레시피에 기록된 대로 학습 마지막 3개월에 대한 prior 모델의 잔차 평균(가중 0.25)을 지평별로 더하는 방식이다.

### 가설 (사전 확정)
H0: offset 제거는 Total S를 바꾸지 않는다(seed 노이즈 이내).
H1: offset 제거는 Total S를 악화시킨다(최근 3개월 잔차의 평균 편향 보정이 이득을 준다는 과거 단일 seed 증거).
효과의 크기와 방향은 3 seed 평균, 최악 seed, seed별 paired 차이로 보고한다.

### 기준선 / 변경 / 유지
- 기준선 A: 보드 행 `e2000fs_calib_offset_roll` (seed 42/43/44, mean3 Total S 96.311; seed별 96.253 / 96.906 / 96.211; F1 117.402 / F2 75.178 / F3 60.935).
- 변경 B(유일): 캘리브레이션 제거. prior 모델(reference window 학습)과 offset 추정·가산 단계를 없앤다. 예측 = full 모델 예측을 [1, 3000]으로 clip.
- 유지: 데이터 `data-v2/input/df.parquet`, 특징 ffill, 학습 라벨 ffill, 특징 목록 전부(상수 23 + 지평별 8), full 모델(XGBoost multi_output_tree, depth 3, lr 0.03, 900 rounds, 24-lead 블록 3개), tail weight 4(≥550 km/s), spectral 계수 0.1, seed, 학습 cutoff(F1 2025-12-29, F2 2026-03-29, F3 2026-06-29), purge 144h.
- 이것은 `removal ablation`이다. 다른 특징이나 구성요소로 대체하지 않는다.
- 같은 seed의 full 모델은 기준선과 동일 조건이므로, 학습된 full 부스터 바이트가 기준선 `weights/`의 `full_b*`와 같아야 한다(결정론 확인). 같다면 B 예측 = 기준선의 offset 이전 예측이다.

### 코드 diff
`diff research/e2000fs_calib_offset_roll/experiment.py research/e2000fs_no_calib_offset_roll/experiment.py`:
`prior_split`, `component_values`, `estimate_offset`, `apply_offset`, `ALPHA`, `OFFSET_WINDOW`, `POLICY` 삭제; `train_period`에서 prior 학습·offset 추정 제거; 체크포인트에서 `prior_b*`·`calibration` 제거; `forecast_values`(clip)만 추가; `F{n}_metadata.json` 기록 추가(진행 표시). 그 외 동일.

### 실행
- seed 42/43/44 각각 독립 프로세스, 로컬, CPU, `CUDA_VISIBLE_DEVICES=''`, `--threads 4`(기준선과 동일 thread 수). 캠페인 스테이징: `research/e2000fs_no_calib_offset_roll_20261110/runs/s{42,43,44}`.
- 학습 종료 후 `--predict-only`로 weights에서 mean3를 만들고, 공식 scorer(`leaderboard/scorer/competition.py`)로 채점한다.
- 평가: 리더보드 F1/F2/F3 기간만. 내부 검증 없음. 리더보드 정답은 채점에만 쓴다.

### 판정 기준 (사전 확정)
- 보고: Total/F별 S, TIMSE, RMSE, seed별 S, mean3 S, 기준선 대비 paired 차이(seed별, 평균, 최악).
- 채택 규칙: 이 실험은 제거가 이득인지 보는 진단이다. 제거 모델이 기준선 mean3 대비 S가 1% 이상 낮을 때만(제출 기준) 새 기준선 후보가 된다. 그렇지 않으면 기준선은 유지한다.
- 리더보드 게시는 사용자의 명시 요청에 따른다(게이트 통과 여부와 무관). 게시 시 `_mean3`만 보이고 seed 행은 숨긴다(board-seed-mean3 정책).

### 실행 상태 / 결과 / 원인 분석
#### 실행 상태 (2026-11-10)
- seed 42/43/44 각각 독립 프로세스(로컬 CPU, 4 threads), 3개 모두 `train.exit` 0. 학습 약 40분/seed. 산출물은 보드 폴더 형식으로 옮겼다.
- 결정론: 새 full 부스터 18개가 기준선 `weights/`의 `full_b*`와 바이트 동일. 예측은 기준선의 offset 이전 예측과 같다.
- 입력/scorer: `data-v2/input/df.parquet`, 관측 파일 `tlabtlab/sunrun-lb-store` revision `0df2354c00a35c516295e8641a995fc93c477753`, scorer `lead-timse-6h-v1`.

#### 결과
| | B(캘리브레이션 제거) | A(기준선) | B−A |
|---|---:|---:|---:|
| mean3 Total S | 97.254 | 96.311 | +0.943 (+0.98%) |
| F1 / F2 / F3 | 118.326 / 74.957 / 60.853 | 117.402 / 75.178 / 60.935 | +0.924 / −0.221 / −0.082 |
| TIMSE | 21,063.1 | 20,032.4 | +5.1% |
| RMSE | 66.96 | 67.29 | −0.5% |
| seed 42 / 43 / 44 | 97.148 / 97.872 / 97.181 | 96.253 / 96.906 / 96.211 | +0.895 / +0.966 / +0.970 |

#### 판정
- H1 지지: 3 seed 모두 같은 방향으로 악화(+0.90 ~ +0.97), 최악 +0.970. H0(노이즈 이내) 기각 쪽. 단 seed 3개이므로 정식 CI는 계산하지 않았다.
- 원인 분석: 악화는 F1(+0.92)에서만 나오고 F2, F3는 약간 좋아진다. RMSE는 오히려 낮으므로 offset의 이득은 평균 오차 감소가 아니라 TIMSE 쪽, 그중 F1에 집중된다(원인 메커니즘은 이번 실험으로 확인하지 않았다). 구현 오류는 아니다(부스터 바이트 동일). 검증 조건 불일치도 없다(같은 scorer, 같은 관측 revision). 가설 결과로 본다.
- 채택 여부: 제출 기준(기준선 대비 S 1% 이상 감소)을 충족하지 못하므로 새 기준선 후보가 아니다. 기준선은 `e2000fs_calib_offset_roll`(S 96.311) 유지. 리더보드 게시는 사용자 요청에 따른다.
- 다음 후보: offset 계수/창 조정은 이번 결과(F1 한 기간에 이득 집중)만으로 정하지 않는다. 그 가설은 별도로 사전등록한다.


#### 게시 (2026-11-10)
- 사용자 요청으로 `e2000fs_no_calib_offset_roll`(3 seed 등가중 mean3)을 리더보드에 게시했다(제출 20261110T054617020Z, Total S 97.254, 보드 순위 6위; 기준선 `e2000fs_calib_offset_roll` 96.311은 4위). 제출 기준(S 1% 이상 감소)은 충족하지 않았고, 게시는 사용자 요청에 따른 것이다.
- architecture 링크와 재현 스크립트는 GitHub main 커밋 `b42d0f880`에 고정했다. seed 단독 행은 게시하지 않았다.

## 2026-11-12 폴더 정리

사용자 지시로 repo를 보드 행 전용 레이아웃으로 정리했습니다. 이 폴더에는 `experiment.py`(유일한 스크립트), `README.md`, `logs/`, `results/`, `weights/`(로컬 전용)만 있고 하위 폴더는 없습니다. 결과는 `results/{tag}_predictions.parquet`·`results/{tag}_score.json`, 체크포인트는 `weights/F{n}_s{seed}.npz`입니다. 위 본문이 가리키는 캠페인 폴더(`research/*_2026…`, `analysis/…`, `runs/…`)는 삭제되어 이 폴더의 사본만 남아 있습니다.

## 2026-11-13 레이아웃 변경 (중복 제거)

- 공용 코드를 `research/module/`로 옮겼습니다. 이 행이 쓰는 모듈: `gbm`, `objectives`. `experiment.py`에는 특징 목록, 목적함수, 보정 설정을 담은 `gbm.GBMSpec`만 남았습니다.
- `results/`와 `logs/`는 `research/result/e2000fs_no_calib_offset_roll/`로 옮겼습니다. 위 본문에 나오는 `results/…`, `logs/…` 경로는 이 위치 기준입니다. `weights/`와 `visualization/`은 이 폴더에 그대로 있습니다.
- `--out`을 주지 않으면 `research/result/e2000fs_no_calib_offset_roll/`에 씁니다. `--out DIR`을 주면 예전과 같이 `DIR/results`, `DIR/logs`, `DIR/weights`에 씁니다(재현 스크립트 호환).
- 검증: 새 코드의 `--predict-only`(weights/) 예측이 기존 `results/`의 s42·s43·s44·mean3와 완전히 같고 S도 같습니다. old/new `--smoke` 예측과 checkpoint 배열(부스터 바이트·calibration)도 모두 같습니다.
