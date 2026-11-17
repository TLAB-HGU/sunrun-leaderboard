# e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec

- 작성자: `seongeun`
- 원래 실험: `~/sunrun-continuous-loss-suite/`, `smooth_tail_scap_nospec`, seeds 42/43/44
- 게시: 2026-11-17 사용자 명시 요청. 연구 게이트 탈락 기록을 유지하는 **예외 게시**이며 누적 기준선으로 채택하지 않는다.
- 예측: seed 42/43/44 전체의 고정 등가중 평균(mean3). 최고 seed 선택이나 평가 정답을 이용한 보정은 하지 않았다.

## 사전 전략 (원본 등록: 2026-11-16, 학습 전)

기준선은 `e2000fs_calib_offset_roll_scap_arrival_cme`이다. 가설은 성능의 핵심이 hard point4 자체보다 tail 강조 A단계와 MSE 제한 S-cap 후속 학습의 결합에 있다는 것이다. 원본 사전 명세는 캠페인 `specs/smooth_tail_scap_nospec/README.md`와 `amendments/scap-composition-v1/README.md`에 보존한다.

이 행은 **두 요소 replacement ablation**이다. 기준선 대비:

1. A단계의 `point4 + 0.1 spectral`을 연속 smooth-tail MSE로 대체한다.
2. S-cap 단계의 spectral 계수도 0으로 바꾼다. 따라서 두 단계 어디에도 spectral 항이 없다.

실제 적합 대상 행렬에서만 `d = max(std(y), 10)`, lead별 `q80`을 계산한다. 셀 가중치는 `w = 1 + 3 sigmoid((y-q80)/(0.2d))`이며 lead별 평균으로 정규화한다. 손실은 `0.5 w (prediction-y)^2`이고, 초기 평균 Hessian의 역수로 한 번 더 정규화한다. 미분은 정확한 대각 Hessian을 사용한다.

유지 요소: 원래 31개 상수 특징과 8개 lead-varying 특징, 24-lead 블록별 **223개 확장 특징의 이름·순서 전부**, 입력·학습 마감·144시간 maturity, 3개 24-lead 블록, A900 + S-cap180 tree, XGBoost 파라미터(`subsample=0.8` 포함), seeds 42/43/44, 336-origin 학습 내 보정 구간, alpha=0.25 고정 offset, [1,3000] clipping. 추가·제거 특징은 없다.

S-cap은 학습 타깃에서 계산한 S를 최소화하며, 30-tree 단계 6개와 Armijo line search를 쓴다. 각 lead의 학습 MSE를 해당 A단계 MSE의 1.01배 이하로 제한한다. 공식 F1/F2/F3 정답은 학습이나 loss 통계 계산에 사용하지 않고 적합 완료 후 채점에만 사용한다. 별도 내부 평가 fold는 없다.

근거 문헌은 [Density-based weighting for imbalanced regression](https://link.springer.com/article/10.1007/s10994-021-06023-5)과 [XGBoost custom objective 문서](https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html)를 참조했다. 이 문헌은 본 sigmoid 가중식이나 이 태양풍 자료에서의 개선을 직접 입증하지 않는다. 해당 조합과 기대 효과는 프로젝트의 사전 가설이다.

## 완료 결과

공식 scorer `lead-timse-6h-v1`, 낮을수록 좋음. 세 seed 모두 exit 0, F1/F2/F3 예측·checkpoint·metadata 완료. 평균 예측은 454,680행이며 공식 채점에서 persistence 대체는 0건이다.

| 구간 | mean3 S | 기준선 S | RMSE | TIMSE |
|---|---:|---:|---:|---:|
| Total | **93.628737** | 93.864258 | 67.515807 | 17749.426888 |
| F1 | 113.749462 | 114.306 | 82.414114 | 26672.640804 |
| F2 | 74.321202 | 73.939 | 63.341852 | 8135.897286 |
| F3 | 59.232524 | 59.865 | 54.389146 | 4647.307912 |

| Seed | S | 동일 seed 기준선 대비 |
|---|---:|---:|
| 42 | 93.477473 | −0.420280% |
| 43 | 93.748627 | −0.395139% |
| 44 | 94.063873 | +0.074203% |

mean3 S는 기준선보다 **0.250917% 감소**했다. 그러나 seed44가 악화되어 사용자 지정 순차 seed 게이트(한 seed라도 개선 없으면 discard)를 통과하지 못했고, 최종 mean3 1% 개선 기준도 충족하지 못했다. F2 S는 악화했다. 따라서 일관된 loss 개선이나 통계적 우위를 주장하지 않는다. 관측 결과는 작은 seed 의존 효과와 양립하지만 원인을 입증하지는 않는다. 실패한 구성을 다음 누적 기준선에 편입하지 않는다. SOTA 대비 2% 목표 달성 여부는 이 비교로 판정하지 않는다.

## 산출물과 재현

- `experiment.py`: 이 아키텍처만 실행. 기존 Arrival-CME 특징과 공용 GBM/continuation 모듈을 재사용하며 smooth-tail callback만 이 행에 정의한다.
- `weights/F{1,2,3}/s{42,43,44}.npz`: 원래 완료 run의 checkpoint, 로컬 전용.
- `research/result/<이 행 이름>/results/`: seed별 및 mean3 예측·공식 점수.
- `research/result/<이 행 이름>/logs/`: 원본 학습 로그, 기간·seed별 특징 목록/데이터 범위/파라미터/버전/commit metadata, seed-gate 판정, 저장 DONKI 원문과 게시 검증 영수증.
- 원래 학습 source commit: `c32b76cc3a488ecec600ed1627f9c70a014f42e7`; 실제 공용 모듈과 loss는 캠페인 frozen source에 보존. XGBoost 2.1.4, GBM CPU 전용.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec/experiment.py --threads 6
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec/experiment.py --predict-only --threads 1 \
  --donki research/result/e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec/logs/donki_notifications.json
```

`--out DIR` 실행은 `DIR/results`, `DIR/logs`, `DIR/weights` 계약을 유지한다. 공개 재현 스크립트는 `scripts/seongeun/<이 행 이름>/`에서 이 파일을 호출한다.

## 게시 전 검증 (2026-11-17)

- `--predict-only`: 세 seed 각각 454,680행이 원본 학습 run 예측과 비트 단위로 일치한다. mean3도 원본 세 예측의 고정 평균과 완전 일치한다.
- 구 campaign / 새 단일 아키텍처 `--smoke`: F1/F2/F3 seed42, CPU 1 thread, 둘 다 exit 0. 예측 1,728행 및 모든 checkpoint 배열(booster bytes와 calibration 포함)이 완전 일치한다.
- 추출한 loss callback의 가중치·정규화·gradient·Hessian 및 223개 특징 순서가 frozen 구현과 일치한다.
- `ruff check`, `py_compile`, `--help` 통과. 공용 모듈은 수정하지 않았다.
