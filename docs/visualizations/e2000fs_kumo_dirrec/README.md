# E2000FS Kumo DirRec — F1+F2 월별 예측 GIF

2026-10-27 게시 기준 리더보드 **18위**, 작성자 **seongeun**인 `e2000fs_kumo_dirrec`입니다. NVIDIA KumoTabular(v1.0.1, large, 8 estimators, 999 quantile 평균)로 horizon마다 앞선 horizon 예측을 넣어 질의하는 DirRec 전략입니다. 입력은 E2000FS와 같은 lag-0 ACE·prefill 캐시이며, 최근 5,000개 scalar 행을 context로 씁니다. gradient 학습이 없는 in-context 모델입니다. 제출 설정 SHA256은 `e39be4ab896b5f8eb24ba207999739fa9eba29a89d1e5705afe41959e60ce8dd`이며, 재현 소스는 [releases/e2000fs_kumo/dirrec](../../../releases/e2000fs_kumo/dirrec/)에 있습니다.

| 지표 | 값 |
| --- | ---: |
| MSE (F1 + F2, 순위 기준) | **4259.032058** |
| F1 MSE (2026-03..05, 2,137 folds) | 4499.560388 |
| F2 MSE (2026-06..09, 2,833 folds) | 4077.595757 |

## Train·validation loss

KumoTabular는 사전학습 가중치로 context 행을 읽어 바로 예측하는 in-context 모델이라 학습 round가 없습니다. 그래서 **per-round loss 기록이 존재하지 않으며**, 만들어 내지 않았습니다.

## Feature importance

만들지 않았습니다. Kumo는 tree split이 없어 split 횟수가 정의되지 않습니다. permutation은 134개 특징 × 5회 × 2기간, 즉 1,340번의 공식 기간 재추론이 필요한데, 공식 기간 추론 한 번이 약 29분(Recursive 기준)이라 현실적으로 계산할 수 없습니다.

## 월별 예측 GIF (F1 + F2)

| 기간 | 월 | 프레임 | GIF |
| --- | --- | ---: | --- |
| F1 | 2026-03 | 744 | [validation_2026-03.gif](validation_2026-03.gif) |
| F1 | 2026-04 | 720 | [validation_2026-04.gif](validation_2026-04.gif) |
| F1 | 2026-05 | 673 | [validation_2026-05.gif](validation_2026-05.gif) |
| F2 | 2026-06 | 720 | [validation_2026-06.gif](validation_2026-06.gif) |
| F2 | 2026-07 | 744 | [validation_2026-07.gif](validation_2026-07.gif) |
| F2 | 2026-08 | 744 | [validation_2026-08.gif](validation_2026-08.gif) |
| F2 | 2026-09 | 625 | [validation_2026-09.gif](validation_2026-09.gif) |

리더보드에 실제 제출한 예측 파일(단일 seed 42)을 그대로 읽습니다. 각 프레임은 공식 평가 시작점 하나이며, 파란 선은 시작점까지 최근 **168시간 관측**, 검은 선은 이후 **72시간 실제값**, 주황 선은 제출 예측입니다. 빈 원은 원 관측이 없어 채워진 지점입니다. 제목에 전체 폴드 번호(4,970개 중), UTC 시작 시각, 해당 72시간 RMSE를 표시합니다.

첫 예측 시각(`origin+1h`)의 UTC 월로 나누고 시간순으로 재생합니다. 1000×500, 200~850 km/s 고정 Y축, 100ms/프레임, 무한 반복입니다. 2026-05-29 ~ 05-31은 72시간 창이 F1·F2 경계에 걸쳐 어느 기간에도 속하지 않으므로 5월 프레임이 673개입니다.

## 검증 ([manifest.json](manifest.json))

`scripts/visualize_enlil_validation.py --experiment E2000FS_KUMO_DIRREC`로 렌더링했습니다. 렌더러는 예측·truth가 4,970 origin × h1..72에 정확히 1:1로 대응하는지, 다시 계산한 MSE가 리더보드 값 4259.032058051013와 1e-7 이내로 일치하는지, 월별 프레임 수와 GIF 디코딩이 맞는지 확인하며, 하나라도 어긋나면 멈춥니다. 입력 파일과 GIF의 SHA256은 manifest에 기록했습니다. 학습이나 예측 변경은 하지 않았습니다.
