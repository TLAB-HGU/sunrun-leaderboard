# offset_roll 전체 ablation — 2026-11-06 사전 고정 계획

## 목적과 비교 기준

사용자 승인: `e2000fs_calib_offset_roll`의 미검증 구성 요소를 모두 검증하고 유지 근거가 확인된 요소만 남긴다. 미확정 요소도 제거 대상으로 삼되, 특징군의 기여가 확인되면 개별 기여가 미확정인 구성원도 묶음으로 유지한다. golab은 현재 로컬 Tlab01이며 golab2는 Windows OpenSSH를 통한 Ubuntu WSL 장비다.

주 대조 A는 training-only frozen calibration의 원래 `e2000fs_calib_offset_roll`이다. 30 원천 특징, 3×24시간 공동 출력, 900 rounds, point4, spectral coefficient .1/scale50, horizon별 마지막 336행 잔차 offset α=.25, component/final clipping [1,3000]을 사용한다. 후속 B1 persistence blend는 포함하지 않는다. 비교 seed는 42/43/44로 고정한다. 현재 기록된 A의 Total S는 98.7626868541542 / 99.93218855943535 / 100.21277355950158이며 수집 시 같은 관측 revision으로 다시 검증한다. 기존 최고 단일 모델·A·외부 SOTA는 별도 항목으로 보고한다.

관측 revision `920588c2a716130e19b145acbbd91ace4758faf2`, 공식 `competition.py` scorer, `df_ffill_trainy.parquet`, F1/F2/F3 cutoff 2025-12-29/2026-03-29/2026-06-29, origin+144h 성숙 조건을 고정한다. 마지막 학습 3개월은 보정 적합용이며 내부 평가·선별 기간이 아니다. 공식 정답은 학습·특징 생성·보정 적합으로 전달하지 않는다.

## 사전 확정 실험과 가설

역사적 `calib_offset_roll_ablation_20261104/assignment.full-superseded.json`의 162조건을 다시 등록하고 spectral Hessian-only 3 seed를 추가해 총 165조건의 증거를 완성한다. 이전 명세는 실행 증거가 아니며, 유효하게 완료된 동일 조건만 재사용한다. 범위 축소 전 옛 큐를 자동 재개하지 않는다.

| 축 | 조건과 가설 | 유지되는 조건 |
|---|---|---|
| 핵심 구성 | tail1/4 × spectral0/1 × offset0/1. 제거 시 성능 악화 여부와 상호작용 확인 | 특징·출력 구조·학습 설정 |
| 개별 특징 | 30개 각각 LOCO. 특정 원천 특징의 조건부 기여 | 해당 특징의 모든 확장열만 제거; prior/full 및 offset 재적합 |
| 특징군 | speed1, ACE10, SUVI7, freshness1, ENLIL2, recurrence4, strip5 각각 제거 | 다른 군을 명시적으로 유지 |
| 보정 | horizon/global/24h-block, 최근336/전체 보정행, α.25/1의 사전 고정 대조 | 원천 모형·입력·seed |
| 출력 구조 | 24→12, 24→72 | 같은 원천 특징·학습 설정; 입력폭 및 spectral 결합폭 변화가 교락되는 구조 대체로 분류 |
| clipping | component/final 각각 제거 및 동시 제거 | 같은 raw prior/full 예측; 해당 조건으로 offset 재추정 |
| spectral 기전 | spectral_hessian_only=True: gradient=w·error, Hessian=w+.1 | point4·보정·특징·출력 구조 |

Hessian-only는 gradient 결합과 추가 diagonal Hessian의 효과를 분리하는 진단이다. 이 결과에서 새 목적함수를 선택하지 않는다. baseline A, Hessian-only, spectral 전체 제거를 비교한다. 일반 XGBoost 설정(depth, learning rate, sampling, regularization), seed 탐색 및 새 아키텍처는 범위 밖이다.

기존 offset 제거와 spectral 제거의 3-seed 결과는 완료했지만 기여 미확정이다. tail 제거는 seed42 완료분을 검증하고 43/44를 보충한다. strip군·block12 완료분과 단일 특징/단일원소군의 중복은 재사용한다. 대체 목적함수 실패 실험을 원래 구성의 제거 검증으로 간주하지 않는다.

## 증거 재사용 및 실행 게이트

중앙 `assignment.global.json`은 전체 165조건과 소유 호스트를 기록하며, 호스트 `assignment.json`은 해당 실행 목록이다. 중앙 `runs/<experiment>/`에 수집한다. 재사용 시 seed, 특징·손실·블록 설정, 입력 identity, cutoff, 라이브러리, 원시 예측 및 정상 종료·채점 증거를 확인한다. 불일치하면 같은 조건을 재학습하거나 오류를 해결하고, 미검증 결과를 완료로 올리지 않는다.

각 호스트에서 기준선 smoke 및 저장 예측 재현을 통과해야 신규 실험을 실행한다. 예측 최대 절대차와 공식 S 차는 각각 ≤1e-6이다. 실패 시 해당 호스트 후속 학습을 보류하고 구현·환경·데이터 차이를 진단한다. 특징 유지/제거 목록, 확장열, cutoff, replay 키, spectral gradient/Hessian 테스트를 실행한다. 코드 diff, Git commit/dirty 상태, 버전·seed·데이터 범위·파라미터를 보존한다.

## 자원과 실행 수명

두 장비 모두 24 CPU·약31 GiB RAM이며 실행 전에 현재 작업과 여유를 다시 확인한다. GBM은 `CUDA_VISIBLE_DEVICES=''`, `--device cpu --threads N`만 사용한다. 초기 최대 호스트당 5작업×4threads, 전체 10작업·40threads다. 기존 부하를 포함해 호스트별 20threads 예산과 가용 메모리 6 GiB를 보존한다. 실측 작업 시간·최대 메모리를 근거로 동시 실행 수를 줄이거나 늘리되 예산을 넘지 않는다. replay·scoring·bootstrap도 같은 예산에 포함한다.

동일 실험 ID는 하나의 호스트만 소유한다. 미시작 작업을 이동할 때 원 소유권을 해제하고 새 소유권을 기록하며, 실행 중 작업을 다른 호스트에 중복 시작하지 않는다. 각 run은 독립 tmux·로그·출력 폴더를 사용한다. `launch.json`, JSON 진행률(period/block/unit/units), 기간별 예측·메타데이터, `train.exit`, 공식 점수 및 `complete.json`을 남긴다. 신규 세션만 정상 종료 후 정리하며 기존 작업은 보존한다. `sunrun --once`로 실행·최근 결과·실제 호스트 표시를 확인한다. 외부 게시는 하지 않고 `publish: not_required`와 사유를 기록한다.

## 통계 판정과 최종 제거

ΔS는 제거/변경 모델−같은 seed A이다. 모든 165조건 완료와 아래 추론 완료 전에는 유지·제거 판정을 보류한다.

- paired circular bootstrap 2,000회, 기간별 층화, 168h origin 블록, RNG seed20261104. 모든 seed와 조건에 같은 origin draw를 사용한다. 매 draw 공식 TIMSE의 경험적 분위수를 다시 계산한다.
- 유지 근거는 3 seed 모두 ΔS>0, 평균 ΔS≥.10, bootstrap95% CI 하한>0, 검정군 BH q<.05를 모두 충족하는 것이다.
- 검정군은 개별 특징, 특징군, 기타(손실·보정·구조·clipping·Hessian 진단)로 분리한다. 기타에는 등록된 baseline 대비 대조 전체를 포함한다. offset-effect 및 factorial 진단은 이 BH 가족에 중복 넣지 않는다.
- 검증된 특징군의 합집합과 독립적으로 검증된 개별 특징을 유지한다. 묶음 유지와 개별 기여 확인은 구분한다.
- tail·spectral·offset·component clip·final clip는 각각 원래 A의 단일 제거 대조만으로 유지 여부를 정한다. 기전·구조 대체 대조에서 최고 점수 설정을 고르지 않는다. 숫자 설정과 24h 블록은 고정 대조 조건이다.
- 유지 조건을 충족하지 못한 선택적 요소는 제거한다. 이는 사용자가 정한 단순화 규칙이며, “효과 없음” 또는 “동등 성능”의 증명이 아니다.
- 특징이 하나도 남지 않으면 임의 특징을 복원하지 않고 `blocked_no_retained_features`를 기록한다. 학습 불가능한 0열 XGBoost를 실행하지 않는다.

`analysis/removal_manifest.json`에 제거 목록과 `final_config`, `final_pruned_s42/43/44`를 한 번 고정한다. 세 seed 최종 조합을 새로 학습하고 `--final` 분석으로 A와 비교한다. 결과가 악화해도 제거 목록을 점수에 맞춰 반복 변경하지 않는다. 이 단계는 이미 사용한 F1/F2/F3에 대한 조합 확인이며 독립적인 미관측 일반화 검증은 아니다. 기존 최고 모델의 자동 교체·SOTA 주장·외부 제출은 하지 않는다.

## 산출물과 완료 조건

`analyze_results.py --root ROOT --partial --bootstrap 0`은 완료/누락 목록과 기술통계를 갱신한다. 전수완료 후 `--bootstrap 2000 --block-hours 168`이 공식 재채점, CI·BH, 상호작용, 유지/제거 manifest를 작성한다. `--final`은 고정 manifest와 최종 조합 결과만 읽어 `final-analysis/`에 별도 보고한다. 호스트 자체 완료는 전체 캠페인 완료가 아니다.

Total/F별 S, RMSE, TRMSE, TIMSE, MAE, bias, seed별 값·평균·표준편차·최악값, 속도 구간·offset 분해·clipping 활성 셀과 유지/제거/묶음 유지 이유를 보고한다. W는 현재 공식 scorer에 정의가 없으므로 S를 W로 바꾸어 부르거나 임의 제출 기준을 만들지 않는다.

완료는 모든 등록 대조의 정상 종료·예측·메타데이터·공식점수·분석과 최종 조합의 검증까지다. 실행 명령 제출 또는 tmux 생성은 완료가 아니다. 실패는 가설 미지지, 구현 결함, 조건 불일치, 자원/최적화 실패로 나누고 로그를 보존한다.

## 방법 근거와 한계

기존 캠페인의 문헌 근거를 재사용한다. 다음 문헌은 방법 선택의 근거이며 현 태양풍 성능 개선을 보장하지 않는다.

- XGBoost 2.1 Multiple Outputs: https://xgboost.readthedocs.io/en/release_2.1.0/tutorials/multioutput.html — vector-leaf joint-output 구현.
- XGBoost advanced custom objectives: https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj.html — diagonal Hessian 근사의 역할; gradient와 Hessian 분리 진단의 근거.
- Künsch (1989), https://doi.org/10.1214/aos/1176347265 — 시간 의존성을 보존하는 block resampling. 168h/circular/F층화의 특정 선택을 직접 검증한 논문은 아니며 정상성 가정·선택한 seed에 조건부인 CI 한계를 보고한다.
- 기존 `offset_roll_ablation2_20261105/literature/evidence.md` 및 5차 strategy — LOCO는 재학습 비교이며 spectral 전체 제거는 gradient와 Hessian이 함께 변한다는 교락을 명시했다.

새로운 성능 가설을 공식 점수로 반복 발굴하지 않는다. 현재 계획은 사전 고정 전체 제거 감사를 한 번 수행하는 것이며 제거 후 성능 하락도 최종 산출물이다.
