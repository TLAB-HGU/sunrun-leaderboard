# e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam

- 작성자: `seongeun`
- 기반 행: `e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec` (mean3 S 93.628737). 재학습 없음.
- 원래 진단 실험: `~/sunrun-block-smooth/` (`block_smooth`, 2026-11-18)
- 게시: 2026-11-18 사용자 명시 요청. 1 % 게이트를 통과하지 못한 **예외 게시**이며 누적 기준선으로 채택하지 않는다.
- 예측: 기간별로 고른 경계 스무딩을 seed 42/43/44 예측에 각각 적용한 뒤 고정 등가중으로 평균(mean3)했다.

## 가설 (진단 실험 실행 전 고정)

기반 행은 24-lead 블록 3개를 따로 학습한다. 그래서 h24→h25, h48→h49 경계의 예측 점프가 블록 안쪽 한 시간 변화의 약 10–15배다(학습 내부 calibration 구간: seam24 9.8–11.0×, seam48 10.7–14.7×). 실제 태양풍 변화에는 이런 경계가 없다. 따라서 경계 점프는 블록 분할이 만든 잡음이고, 점프를 줄이면 경계 근처 lead의 오차가 줄어든다고 가정했다.

변환 9개를 미리 고정했다.

- `none`: 그대로(clip만).
- `ma{1,2,3,4,6}`: 경계 양쪽 m lead에 중심 이동평균(창 2m+1)을 적용한다.
- `xf{3,6,12}`: 경계 점프에서 국소 기울기를 뺀 몫을 양쪽에 반씩 나눈다. 이를 k lead에 걸쳐 선형으로 줄여 없앤다.

선택 규칙: 기간마다 학습 내부 calibration 구간(학습 마감 직전 3개월)에서 고른다. 이 구간에서 기반 행의 prior 모델과 frozen offset으로 seed 3개 평균 예측을 만들고, 학습 라벨에 대해 공식 lead별 S 공식으로 proxy S를 계산한다. proxy S가 가장 낮은 변환을 쓴다. 리더보드 F1/F2/F3 정답은 선택에 쓰지 않고 채점에만 쓴다.

| 기간 | calibration 구간 | 채점 origin | proxy S none | 선택 | proxy S 선택 |
|---|---|---:|---:|---|---:|
| F1 | 2025-09-23 – 2025-12-23 | 365 | 125.8511 | xf12 | 125.6426 |
| F2 | 2025-12-23 – 2026-03-23 | 361 | 115.1929 | xf6 | 115.0505 |
| F3 | 2026-03-23 – 2026-06-23 | 369 | 70.2753 | xf12 | 69.9374 |

유지 요소: 기반 행의 223개 특징, A900 + S-cap180 모델, checkpoint, offset, seed를 모두 그대로 쓴다. 변경 요소는 예측 뒤 경계 변환 하나다. prior checkpoint로 다시 계산한 offset은 9개 모두 저장값과 정확히 같아야 하며, 다르면 실행이 멈춘다.

## 결과

공식 scorer `lead-timse-6h-v1`, 낮을수록 좋음. 454,680행이며 persistence 대체는 0건이다.

| 구간 | mean3 S | 기반 행 S | RMSE | TIMSE |
|---|---:|---:|---:|---:|
| Total | **93.422191** | 93.628737 | 67.184627 | 17724.651 |
| F1 | 113.622766 | 113.749462 | 82.028171 | 26756.980 |
| F2 | 74.117163 | 74.321202 | 63.086935 | 8087.351 |
| F3 | 58.840016 | 59.232524 | 54.032651 | 4564.166 |

| Seed | S | 같은 seed 기반 행 대비 |
|---|---:|---:|
| 42 | 93.289733 | −0.20 % |
| 43 | 93.491560 | −0.27 % |
| 44 | 93.796293 | −0.28 % |

- mean3 S는 **0.221 % 감소**했다. 세 seed와 세 기간 모두 개선됐고, RMSE도 개선됐다.
- TIMSE는 F1에서만 +0.3 % 악화했다.
- mean3 예측의 경계 점프 비율은 0.81×/0.76×로 블록 안쪽 수준이 됐다.
- 1 % 제출 기준과 SOTA 대비 2 % 목표에는 미달한다.

**진단 실험과의 차이:** `~/sunrun-block-smooth`의 진단 스크립트(93.4367)는 예측 행의 기간을 날짜 규칙(`origin < 다음 cutoff − 144h`)으로 나눴다. 이 규칙 때문에 F1 마지막 144개 origin(2026-03-23 – 03-28)에는 F2 변환이, F2 마지막 24개에는 F3 변환이 적용됐다. 이 행은 공식 query 개수(F1 2089 / F2 2113 / F3 2113, 예측 파일 순서)로 기간을 나눈다. 고른 변환은 동일하며, 결과 차이는 이 기간 배정 수정에서만 나온다.

## 산출물과 재현

- `experiment.py`: 기반 행의 `SPEC`과 prior checkpoint로 calibration 구간 예측을 만들어 변환을 고른다. 기반 행 seed 예측에 변환을 적용해 seed별·mean3 예측과 점수를 쓴다.
- 기본 입력: `research/result/<기반 행>/results/` 예측과 `research/<기반 행>/weights/F{n}/s{seed}.npz`(로컬 전용). `--base DIR`이면 `DIR/results`, `DIR/weights`를 쓴다.
- 산출: `research/result/<이 행>/results/{s42,s43,s44,mean3}_predictions.parquet`·`_score.json`, `logs/`(`run.jsonl`, 기간별 `F{n}_metadata.json`: calibration 범위, 변환별 proxy S, 선택, 경계 통계).
- 공개 재현 스크립트 `scripts/seongeun/<이 행>/`는 기반 행을 `--out`으로 새로 학습한 뒤 이 파일을 `--base`로 호출한다.
- `--base` 경로 실행이 기본 경로 실행과 seed별·mean3 예측이 완전히 같음을 확인했다.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam/experiment.py --threads 4 \
  --donki research/result/e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec/logs/donki_notifications.json
```
