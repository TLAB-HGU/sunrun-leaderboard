# union9_lbsel_plus4에 리드별 신호 3개 추가

## 전략 및 사전 명세

기준선 A는 완료된 `e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel_plus4`다. Mean3 Total S는 **91.17845947919972**이며, 최초 union9_lbsel의 **92.10851479473993**과 구분한다. 사용자 지정에 따라 세 신호 B를 함께 추가한 A+B를 평가한다. 세 신호의 개별 기여를 분리하는 실험은 아니다.

| 추가 특징 | 적용 리드 | 가설 |
|---|---|---|
| `enlil_earth_speed` | 1–72h | 예보된 지구 풍속의 시간 구조로 고속 도착을 구별 |
| `onset700_hazard` | 1–72h | 학습 자료로 적합한 고속 진입 확률로 진입 시점 표현을 보강 |
| `rec54_consensus` | 1–72h | 54일 반복 신호의 합의로 반복류의 지속성을 보강 |

기존 특징 전체를 합집합으로 유지한다. plus4의 `cme_phase_rec27`, `rec27_ramp`는 모든 리드에 유지하며, `suvi_t055_lowlat_top1_lon_width_cap67_deg`, `suvi_ballistic_dark_contrast`는 기존과 동일하게 49–72h에 유지한다. 각 블록의 기존 입력 273/273/278열을 같은 순서의 prefix로 보존하고 세 신호를 리드 우선 순서로 72열씩 추가하여 **345/345/350열**을 사용한다. 결측 sentinel -999는 추가 입력에서 NaN으로 변환한다.

학습 라운드, 목적함수, S-cap, offset 및 원래 seam xf8 → ridge 5% → 3시간 위상 전진 → 시드 합의 uplift ≤40 → 고정 동일 가중 Mean3를 유지한다. 시드 합의 보정은 세 시드가 필요하므로 단일 시드 raw 결과는 최종 모델의 동등한 판정이 아니다. 공식 F1/F2/F3 cutoff 이전 자료만 적합에 사용하며, 공식 정답은 채점에만 사용한다. 보조 hazard 생성의 학습/예측 시점 적합성은 입력 생성 영수증으로 확인해야 한다.

동일 시드 42/43/44와 Mean3의 공식 S, W, MAE, RMSE, TIMSE 및 기간별 결과를 비교한다. 제출 후보 기준은 동일 기준선 대비 W 1% 이상 감소와 보조 지표의 중대한 악화 없음이다. SOTA 대비 W 2% 감소는 별도로 판정한다. 공식 기간 결과를 참고해 후보를 선택한 탐색적 결합이므로 독립적인 일반화 검증으로 주장하지 않는다. 실패한 결합은 누적 기준선에 반영하지 않는다.

## 근거와 입력 출처

기존 후보의 기전과 생성 절차를 재사용하며 새 문헌 효과를 검증된 개선으로 주장하지 않는다.

- [plus4 기준선](../e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel_plus4/README.md)
- [기존 CME·반복류 캠페인 명세](/home/t-lab01/sunrun-discrimination10/README.md)
- 입력: `/home/t-lab01/sunrun-discrimination10/input/{enlil_earth_speed,onset700_hazard,rec54_consensus}.parquet`
- 시점/생성 영수증: 같은 디렉터리의 `enlil_earth_speed.ready.json`, `enlil_earth_speed.run_selection.json`, `onset700_hazard.json`, `onset700_hazard-validation.json`, `rec54_consensus.receipt.json`.

## 실행 상태

실행 전 사전 등록. `experiment.py`는 기존 plus4를 읽어 디자인 prefix를 보존하고 세 입력만 추가한다. 기존 실행기와 공용 모듈은 변경하지 않는다. 학습·채점·실패 분석·채택 여부는 캠페인 완료 후 기록한다.

## 재현 (2026-11-25)

입력 51열은 모두 `data-v2/input/df.parquet`에 들어 있다. `data-v2/fetch.py`가 원문을 `data-v2/raw/`에 받고, `data-v2/notebooks/raw_to_input_df.ipynb`의 6b절이 446열 밖에 있던 20열을 만든다.

```bash
.venv/bin/python data-v2/fetch.py                 # 원문
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 data-v2/notebooks/raw_to_input_df.ipynb
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel_plus7/experiment.py --threads 6
```

- 2026-11-25 이전 기록: NCEI 포함 df는 캠페인 입력 표 `~/sunrun-union9-plus7/input/model-table-union9-plus7.parquet`와 466열·2,816,784행이 모두 같았고, mean3 S는 90.17361022099254였다.
- 지금 df(NCEI 제거판)와 checkpoint로 `--predict-only`(`--data` 없음)를 실행하면 seed 42/43/44와 mean3 예측이 아래 재학습 결과와 같고, mean3 S는 91.56316183549588이다.

## NCEI 제거 재학습 (2026-11-25)

NCEI WSA-Enlil archive(실행 후 약 25 h 공개, tarball 개당 약 145 MB)는 라이브에서 쓸 수 없어 데이터 파이프라인에서 뺐다.
`enlil_earth_speed`는 이제 NOAA open-data S3 `noaa-wsa-enlil-pds` 실행(597개)만으로 만든다. 그 밖의 코드·설정·후처리는 같고,
frozen ridge(`union9_lbsel`의 `ridge.npz`)를 그대로 쓰며 GBM 3기간 × 3 seed만 다시 학습했다. 기록은 `~/sunrun-plus7-noncei/`.

- 입력 변화: 노트북 6b만 다시 실행했고 `enlil_earth_speed` 밖의 465열은 이전 df와 값이 완전히 같다.
  `enlil_earth_speed`가 바뀐 칸(origin × lead 기준)은 다음과 같다. 원래 S3 값이던 칸은 변하지 않았다.

  | 기간 | 값 있는 비율 (이전 → 지금) | 바뀐 칸 | -999가 된 칸 | 다른 S3 실행으로 바뀐 칸 |
  |---|---|---|---|---|
  | 학습 | 18.1 % → 14.8 % | | | |
  | F1 | 78.7 % → 37.7 % | 50.8 % | 41.0 % | 9.8 % |
  | F2 | 90.0 % → 81.4 % | 20.1 % | 8.6 % | 11.5 % |
  | F3 | 98.8 % → 72.1 % | 39.1 % | 26.7 % | 12.4 % |

- 결과(공식 scorer, mean3): **S 91.563 (이전 90.174, ΔS +1.390, +1.54 %)**. seed 42/43/44는 91.419 / 91.727 / 91.899(이전 90.033 / 90.128 / 90.778)로 세 seed 모두 나빠졌다.

  | | Total | F1 | F2 | F3 |
  |---|---|---|---|---|
  | S (지금) | 91.563 | 109.703 | 73.929 | 60.121 |
  | S (NCEI 포함) | 90.174 | 107.346 | 73.823 | 59.843 |
  | TIMSE (지금 / 이전) | 14295 / 13437 | 22059 / 20056 | 6391 / 6578 | 4150 / 3977 |
  | RMSE (지금 / 이전) | 72.08 / 72.43 | 84.70 / 86.05 | 70.68 / 69.46 | 59.06 / 59.87 |

- 해석: 악화는 커버리지가 가장 많이 줄어든 F1(ΔS +2.36)에 몰려 있고(F2 +0.11, F3 +0.28), TIMSE가 커졌다. RMSE는 거의 같다.
  ENLIL 지구 풍속 예보가 고속류 도달 시각 표현에 기여했던 부분이 S3 실행의 불규칙한 공개(월 8–22일, 최대 13일 공백)로 줄었다.
  NCEI 포함 점수는 라이브에서 재현할 수 없는 입력에 기댄 값이므로 이 행은 NCEI 제거판으로 교체한다.
- 검증: 세 seed `train.exit` 0, 기간별 checkpoint·metadata 9개, 학습 입력 `data-v2/input/df.parquet`(220,093,500 bytes),
  `--predict-only` 재실행 예측이 seed별·mean3 모두 완전히 같다.


## ridge 제거 ablation (2026-11-25, 채점 후 진단, NCEI 포함 df 기준)

- 후처리에서 ridge 5 % 결합 단계 하나만 빼고 GBM raw 예측과 나머지 단계는 그대로 두었다. 기록은 `~/sunrun-plus7-noridge/`.
  A를 재조립한 결과는 저장된 mean3와 비트 단위로 같았다.
- mean3 S 90.174 → 90.094(ΔS −0.080, −0.089 %), CI [−0.22, +0.33]이고 seed별 ΔS는 −0.096 / −0.062 / −0.037이다.
  RMSE +0.38 %, TIMSE −1.12 %, F1 / F2 / F3 S +0.06 / −0.09 / +0.14 %다.
- 판정은 ridge 기여 미검출(중립)이다. 이 행에서 ridge는 RMSE를 조금 낮추고 TIMSE를 높이는 수축 효과만 낸다. 구성은 바꾸지 않는다.
