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
