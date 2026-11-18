# e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam

- 작성자: `seongeun`
- 기반 행: `e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec` (mean3 S 93.628737). 재학습하지 않았다.
- 원래 진단 실험: `~/sunrun-block-smooth/` (`block_smooth`, 2026-11-18).
- 게시: 2026-11-18 사용자 명시 요청에 따른 **예외 게시**다. 1 % 게이트를 통과하지 못했고 누적 기준선으로 채택하지 않는다.
- 현재 버전(v2, 2026-11-18): 모든 기간에 **xf8**을 쓴다. k = 8은 사용자 요청으로 **리더보드 sweep(xf1–xf24)에서 고른 값**이다. LB 정답으로 하이퍼파라미터를 고른 것이므로 이 점수는 LB에 맞춰진 값이고, 깨끗한 일반화 추정이 아니다.
- 예측: 기반 행 seed 42/43/44 예측에 각각 xf8을 적용한 뒤 고정 등가중으로 평균(mean3)했다.

## 변환

기반 행은 24-lead 블록 3개를 따로 학습한다. 그래서 h24→h25와 h48→h49 경계의 예측 점프가 블록 안쪽 한 시간 변화의 약 10–15배다. xf{k}는 다음 순서로 계산한다.

1. 경계 b(24, 48)에서 점프 J = (p₍b+1₎ − p_b) − ½[(p_b − p₍b−1₎) + (p₍b+2₎ − p₍b+1₎)]를 구한다.
2. J의 절반을 양쪽에서 없앤다. 왼쪽 lead b−i에는 +½J·(k−i)/k, 오른쪽 lead b+1+i에는 −½J·(k−i)/k를 더한다(i = 0…k−1). 이렇게 하면 k lead에 걸쳐 선형으로 줄어든다.
3. 보정은 원래 곡선에서 계산하고, 마지막에 clip [1, 3000]을 적용한다.

유지 요소: 기반 행의 223개 특징, A900 + S-cap180 모델, checkpoint, offset, seed를 모두 그대로 쓴다. 바꾼 것은 예측 뒤 경계 변환 하나다.

## k 선택 경위

1. v1(사전 등록): 변환 9개(none, ma1–6, xf3/6/12) 중에서 기간마다 학습 내부 calibration 구간(학습 마감 직전 3개월, prior 모델 + frozen offset, 학습 라벨)의 proxy S로 골랐다(F1 xf12, F2 xf6, F3 xf12). 결과는 mean3 S 93.422191(−0.221 %)이고 submit 20261118T043451683Z다. 기록은 `~/sunrun-block-smooth/publish/replaced_v1/`에 있다.
2. sweep: 모든 기간에 같은 xf{k}를 적용하고 k = 1…24를 LB에 채점했다(`~/sunrun-block-smooth/runs/block_smooth/xf_sweep_lb.csv`).
   - 최소는 k = 8(93.3726)과 k = 9(93.3728)이고, k = 6–10이 −0.26 ~ −0.27 %로 평평하다.
   - k를 키울수록 RMSE는 계속 줄어든다. TIMSE는 k = 5에서 최소이고 그 뒤로 커진다.
   - 기간별 최적은 F1 6, F2 13, F3 14이고, seed별 최적은 8 / 9 / 9다.
3. v2: 사용자 요청에 따라 k = 8로 바꿔 다시 게시하고 v1 record를 교체했다.

## 결과 (v2, xf8)

공식 scorer `lead-timse-6h-v1` 기준이고 낮을수록 좋다. 454,680행이며 persistence 대체는 0건이다.

| 구간 | mean3 S | 기반 행 S | RMSE | TIMSE (기반 대비) |
|---|---:|---:|---:|---:|
| Total | **93.372603** | 93.628737 | 67.223979 | 17672.513 (−0.43 %) |
| F1 | 113.520740 | 113.749462 | 82.121614 | 26616.186 (−0.21 %) |
| F2 | 74.084100 | 74.321202 | 63.026628 | 8081.728 (−0.67 %) |
| F3 | 58.883400 | 59.232524 | 54.110600 | 4565.642 (−1.76 %) |

| Seed | S | 같은 seed 기반 행 대비 |
|---|---:|---:|
| 42 | 93.225176 | −0.27 % |
| 43 | 93.451351 | −0.32 % |
| 44 | 93.769501 | −0.31 % |

- mean3 S는 **0.274 % 감소**했다. 세 seed, 세 기간 모두 개선됐고 RMSE와 TIMSE도 모두 개선됐다.
- mean3 예측의 경계 점프 비율은 0.80× / 0.75×다(기반 11× / 14×).
- 1 % 제출 기준과 SOTA 대비 2 % 목표에는 미달한다.
- v1(학습 구간 선택, 93.4222)보다 0.05 S 낮다. 다만 이 차이는 LB를 보고 k를 고른 효과를 포함한다.

## 산출물과 재현

- `experiment.py`: 기반 행 seed 예측에 xf8을 적용해 seed별·mean3 예측과 점수를 쓴다. 학습과 calibration 선택은 하지 않는다.
- 기본 입력은 `research/result/<기반 행>/results/s{42,43,44}_predictions.parquet`이다. `--base DIR`이면 `DIR/results`를 쓴다. 두 경로의 seed별·mean3 예측이 완전히 같음을 확인했다.
- 산출: `research/result/<이 행>/results/{s42,s43,s44,mean3}_predictions.parquet`·`_score.json`, `logs/run.jsonl`(경계 통계 포함).
- 공개 재현 스크립트 `scripts/seongeun/<이 행>/`는 기반 행을 `--out`으로 새로 학습한 뒤, 이 파일을 `--base`로 호출한다.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam/experiment.py
```
