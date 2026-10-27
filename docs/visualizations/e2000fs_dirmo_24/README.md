# E2000FS DirMO-24 — train·validation loss와 F1+F2 월별 예측 GIF

2026-10-27 기준 리더보드 **1위**, 작성자 **seongeun**인 `e2000fs_dirmo_24`입니다. E2000 입력(111 features, ACE lag 0)과 고정 XGBoost recipe(900 rounds, depth 3, lr 0.03, seed 42)로 24시간 블록별 독립 multi-output 모델 3개(h1–24 / h25–48 / h49–72)를 학습한 DirMO(σ=24) 전략입니다. 제출 설정 SHA256은 `6fc76b1836f3a22669108823c6bf6833082b4d9a74bedf80edd8199f66183037`이며, 재현 소스는 [releases/e2000fs/dirmo_24](../../../releases/e2000fs/dirmo_24/)에 있습니다.

| 지표 | 값 |
| --- | ---: |
| MSE (F1 + F2, 순위 기준) | **3480.270151** |
| F1 MSE (2026-03..05, 2,137 folds) | 3878.876144 |
| F2 MSE (2026-06..09, 2,833 folds) | 3179.592068 |

## Train·validation loss

![E2000FS DirMO-24 loss history](loss_history.png)

실제 제출 예측을 만든 두 모델의 **학습 중 기록한 XGBoost per-round loss**(`evals_result`)입니다. 사후 재구성이 아니며, smoothing 없이 원래 값을 표시합니다.

- 위 행 **F1 모델**: target 시점 < 2026-02-26 데이터로 학습 → F1 예측.
- 아래 행 **F2 모델**: target 시점 < 2026-05-29 데이터로 학습 → F2 예측.
- 열은 24시간 블록(h1–24 / h25–48 / h49–72), 파란색은 train MSE, 주황색은 validation MSE, 점선은 사용한 900 round입니다.
- **주의: validation은 held-out이 아닙니다.** validation 구간(origin 2025-12-04 ~ 2026-02-22, 타깃은 ffill된 ACE 속도)은 두 모델의 학습 구간 안에 들어 있습니다. 학습 데이터 일부를 따로 평가한 **진단용 곡선**이므로 일반화 성능이나 조기 종료 근거로 읽으면 안 됩니다. 900 rounds는 고정 recipe이며 이 곡선으로 고르지 않았습니다.

| 모델 | 블록 | train MSE (900) | validation MSE (900) |
| --- | --- | ---: | ---: |
| F1 | h1–24 | 1280.7 | 1728.8 |
| F1 | h25–48 | 2201.0 | 3035.0 |
| F1 | h49–72 | 2622.5 | 3274.2 |
| F2 | h1–24 | 1304.6 | 1812.5 |
| F2 | h25–48 | 2253.9 | 3188.2 |
| F2 | h49–72 | 2683.9 | 3413.4 |

원본 CSV(`step,train_mse,validation_mse`, (km/s)²)는 [loss_history/](loss_history/)에 있고, `analysis/forecast_strategies_e2000/outputs/loss_history/{f1,final}/DirMO-24/`와 바이트 단위로 같습니다. 그림은 [code/plot_loss.py](code/plot_loss.py)로 다시 만들 수 있습니다.

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

리더보드에 실제 제출한 예측 파일(단일 seed 42)을 그대로 읽습니다. 각 프레임은 공식 평가 시작점 하나이며, 파란 선은 시작점까지 최근 **168시간 관측**, 검은 선은 이후 **72시간 실제값**, 주황 선은 제출 예측입니다. 빈 원은 원 관측이 없어 채워진 지점입니다. 제목에 전체 폴드 번호(4,970개 중), UTC 시작 시각, 해당 72시간 RMSE를 표시합니다. h24·h48 경계의 계단은 블록별 독립 모델(DirMO-24)이기 때문입니다.

첫 예측 시각(`origin+1h`)의 UTC 월로 나누고 시간순으로 재생합니다. 1000×500, 200~850 km/s 고정 Y축, 100ms/프레임, 무한 반복입니다. 2026-05-29 ~ 05-31은 72시간 창이 F1·F2 경계에 걸쳐 어느 기간에도 속하지 않으므로 5월 프레임이 673개입니다. 관측 7일은 표시 구간일 뿐 모델의 전체 입력 특성을 나타내지는 않습니다.

## 검증 ([manifest.json](manifest.json))

`scripts/visualize_enlil_validation.py --experiment E2000FS_DIRMO24`로 렌더링했습니다. 렌더러는 예측·truth가 4,970 origin × h1..72에 정확히 1:1로 대응하는지, 다시 계산한 MSE가 리더보드 값 3480.2701508981454와 1e-7 이내로 일치하는지, 월별 프레임 수와 GIF 디코딩이 맞는지 확인하며, 하나라도 어긋나면 멈춥니다. 입력 파일과 GIF의 SHA256은 manifest에 기록했습니다. 학습이나 예측 변경은 하지 않았습니다.
