# e2000fs_calib_point4_144h — 누출(leakage) 검토

검토일 2026-11-01, 대상 `e2000fs_calib_point4_144h`(리더보드 MSE 4,452.152081, F1 6,426.86 / F2 3,896.40 / F3 3,055.62, 게시 기준 10위).
재현 스크립트는 [`code/leakage_audit.py`](code/leakage_audit.py), 원본 수치는 [`leakage_audit.json`](leakage_audit.json)입니다. 14개 검사 **전부 통과**했고, 아래 "남는 위험" 절의 **특징 선택 누출 1건은 실제로 존재**하며 이 arm에도 그대로 적용됩니다.

## 결론

- **학습/평가 겹침 없음.** 학습 행은 `origin + 144 h ≤ cutoff`로 잘리고, 마지막 학습 목표와 첫 채점 목표 사이에 F1·F2 **144 h**, F3 **120 h** 간격이 있습니다. 학습 origin과 채점 origin의 교집합은 0, 학습·채점 설계행렬 사이의 **완전 중복 행도 0**입니다.
- **미래 정보 의존 없음(재계산 가능한 범위).** ACE·플라스마·속도 특징은 모두 origin 시점까지의 rolling/shift/ffill이고, 재계산값이 번들과 1e-4 이내로 일치합니다. 순환 특징 `rec26/27/28`은 정확히 **목표시각 − 26/27/28일**의 속도이며, 이는 채점 목표보다 26일 이상 앞선 과거입니다.
- **정답 복사 없음.** 채점 격자 237개 열과 목표의 상관을 전수 스캔했을 때 0.9를 넘는 열은 `v_lag0`·`ace_speed_kms_last` 두 개뿐이고, 둘 다 **origin 시각의 속도**(t0 규칙이 허용하는 입력)입니다. horizon별 특징(`rec*`, `strip_*`, `enlil*`)의 최대 상관은 **0.54**입니다.
- **예측 출처 확인.** archive 예측은 마스크된 학습 행만 쓴 재학습으로 **비트 단위(최대 차이 0.0)** 재현됩니다. held-out loss가 학습 loss보다 크고(평균 72.1 vs 51.6 km/s), 점수도 리더보드 레코드와 1e-9 이내로 일치합니다.

## 세부 검사

| # | 검사 | 결과 | 근거 수치 |
| ---: | --- | --- | --- |
| 1 | 학습/평가 origin 분리 | 통과 | F1/F2/F3 각각 교집합 0, 학습 행 9,433 / 10,153 / 10,889(리핏 로그와 일치) |
| 2 | 마지막 학습 목표 ↔ 첫 채점 목표 간격 | 통과 | F1 144 h(2025-12-26 → 2026-01-01), F2 144 h, F3 120 h(2026-06-26 → 2026-07-01) |
| 3 | 학습·채점 설계행렬 완전 중복 행 | 통과 | 9개 (period, block) 조합 모두 0 |
| 4 | 목표 시계열 결측 충전 방식 | 통과 | 7,269시간(20.2%)이 채워져 있고, 전부 **직전 관측값 유지(ffill)**, ffill과 불일치 0건 |
| 5 | 입력 속도 시계열 = 리더보드 이력 시계열 | 통과 | 인덱스·값 동일(`e2000fs-lag0/speed.parquet` == `ch-v1/ace.parquet` filled) |
| 6 | 속도 가족 특징 재계산 | 통과 | `v_lag0`·`v_m6`·`v_m24`·`v_sd6` 최대 차이 4e-6 ~ 3e-5 |
| 7 | 순환 특징 시점 | 통과 | `rec26/27/28` = 목표 − 26/27/28일 (최대 차이 2.4e-5, float32 저장 오차) |
| 8 | SUVI 영상 나이 | 통과 | 사용된 모든 영상의 `img_age_h` > 0 (최소 1.00 h, 중앙값 1.01 h, 최대 65.0 h) |
| 9 | 코로나홀·Enlil 특징의 인과 규칙 | 통과 | CH: `slot ≤ origin`, `ffill` 6 h 한도 / Enlil: `modelCompletionTime ≤ origin` 런만 사용([headroom.py](../../../leaderboard/experiments/headroom_audit/headroom.py)) |
| 10 | 설계 열 ↔ 목표 상관 스캔 | 통과 | 0.9 초과 열은 `v_lag0`·`ace_speed_kms_last`(둘 다 origin 속도)뿐, 다음이 ACE 온도 0.83, 나머지 상수 특징은 0.59 이하, horizon별 특징은 0.54 이하 |
| 11 | 예측 재현 | 통과 | 9개 booster 재학습이 archive 예측을 최대 차이 0.0으로 재현 |
| 12 | 학습 vs held-out loss | 통과 | point4 RMSE 평균 51.6(학습) vs 72.1(held-out) |
| 13 | 점수 재현 | 통과 | 채점 재계산 4,452.152081 == 리더보드 레코드 |
| 14 | (참고) 공식 채점 구간을 보지 않는 설계 상수 | 통과 | 900 rounds·depth 3·eta 0.03·4배 고속 가중·144 h 마스크는 이전 e2000fs 연구에서 상속된 고정값 |

## 남는 위험 (통과했지만 알고 있어야 하는 것)

1. **특징 선택 누출 (실제 존재, 문서화됨).** 입력 30개는 `e2000fs_feat30_extreme` 단계에서 **공식 F1/F2 구간의 permutation importance**를 보고 골랐습니다. 그래서 이 arm의 점수는 "선택까지 포함한 엄격한 홀드아웃"이 아닙니다. 모든 e2000fs arm(및 그 위에 얹힌 calib4/calib5 스택)이 같은 특징 집합을 쓰므로 공통 한계이며, 없애려면 내부 창에서 특징을 다시 고르고 전부 재학습해야 합니다.
2. **SUVI 특징의 재계산 미검증.** 원본 SUVI 영상에서 `strip_*`·CH 특징을 이 자리에서 다시 만들지는 않았습니다(원자료 파이프라인 재구동 필요). 대신 (a) 사용 영상 나이가 항상 양수(검사 8), (b) CH·Enlil의 인과 규칙 코드(검사 9), (c) 상류 v3 materialization 감사 영수증(`leaderboard/experiments/cumulative_lane12_20261024/materialized_v3/audit-v3-local.json`, future/past perturbation 포함)을 근거로 삼았습니다.
3. **F3 간격이 120 h.** 다른 기간(144 h)보다 짧은 것은 F3 cutoff가 2026-06-29로, 첫 채점 목표 2026-07-01보다 48 h 앞이기 때문입니다. 학습 데이터는 여전히 72 h 성숙 규칙을 만족하고(마지막 학습 목표 06-26), 겹침은 없습니다.
4. **t0 규칙 자체.** `v_lag0`과 `ace_speed_kms_last`는 origin 시각의 속도이므로 목표와 상관이 0.97–0.98입니다. 리더보드 규칙(`timestamp ≤ t0` 입력 허용)이 명시적으로 허용하는 입력이며, 미래값이 아닙니다.
5. **충전된 목표.** 리더보드 정답 자체가 ACE 결측을 ffill로 채운 시계열입니다(20.2%). 학습 목표와 채점 정답이 같은 규칙을 쓰므로 일관되지만, 결측 시간대의 점수는 관측값이 아니라 유지값에 대한 오차입니다(문서에도 명시).

## 재현

```bash
cd /home/t-lab01/sunrun
/home/t-lab01/sunrun/.venv/bin/python docs/visualizations/e2000fs_calib_point4_144h/code/leakage_audit.py
```

스크립트는 `research/e2000fs_f123-20261031/{common,arms}.py`의 실제 학습 코드로 학습/채점 구간을 다시 만들고, 동결 입력 번들(`data-v2/model-inputs/leaderboard-store/e2000fs-lag0`)과 리더보드 `folds/truth`(pinned sha256)를 읽습니다. 어느 검사든 실패하면 `assert`로 멈추고 `leakage_audit.json`의 `verdict`가 바뀝니다.
