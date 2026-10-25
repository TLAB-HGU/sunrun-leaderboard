# E1466 TD-WSA HUXt72 — official-window forecast GIFs

E1466 `e1466_td_wsa_huxt72`의 공식 리더보드 제출 예측(3-seed equal mean)을 고정 folds/truth 스냅샷과 비교해 월별 GIF로 렌더링했습니다. 예측은 원래 제출본을 그대로 읽었으며, 재학습·재추론·점수 변경은 없습니다.

## 기록

- 공식 점수: **MSE 3,246.255** (2833 origins × 72 h, 203,976 rows, seed-averaged prediction).
- 제출 당시 전체 **rank 3**, 비앙상블 단일 모형 중 최고 기록.
- 내부 lbeval은 seed-mean MSE **3,356.99**로 E0472의 3,281.31을 넘지 못했습니다. `high_high_high` 셀도 4,569.3으로 4,174.4보다 나빴습니다. 이 GIF는 그 판정을 바꾸지 않습니다.
- lead band 변화: 25–48 h **−2.30%**, 49–72 h **−0.59%**, 1–6 h **+2.62%**, 7–24 h **+0.73%**.
- 2026-10-20 제출된 historical official-exposed record입니다. 새 holdout이나 현재 계약의 비교 챔피언(E1813)이 아닙니다.

## GIF

| 월 | 파일 | 프레임 | 크기 |
|---|---|---:|---:|
| 2026-06 | `validation_2026-06.gif` | 720 | 23,374,478 B |
| 2026-07 | `validation_2026-07.gif` | 744 | 24,587,401 B |
| 2026-08 | `validation_2026-08.gif` | 744 | 23,948,901 B |
| 2026-09 | `validation_2026-09.gif` | 625 | 20,894,600 B |

각 프레임은 origin 시점의 최근 168시간 ACE 관측(ACE+2h 규칙)과 이후 72시간 실제값·예측을 함께 보여줍니다. GIF는 100 ms/frame, 무한 루프이며 모든 프레임 디코딩을 검증했습니다. 프레임별 SHA256과 월별 첫/마지막 fold는 [manifest.json](manifest.json)에 있습니다.

## Train/validation loss 그래프에 대한 기록

**E1466의 XGBoost backbone은 고정 900 rounds로 학습되어 eval_set·early stopping·per-round train/validation loss history가 없습니다.** 유일한 epoch 단위 곡선은 TD-WSA field encoder의 self-supervised reconstruction loss(`curves_s0/s1/s2.json`)였는데, 해당 파일은 원 run 디렉터리(`research/expos/runs/E1466`)와 함께 삭제되어 현재 남아 있지 않습니다. 등록된 SHA256과 학습 시간은 [loss_history_audit.json](loss_history_audit.json)에 보존했습니다. **곡선을 재구성하거나 합성하지 않았습니다.**

대신 저장된 예측과 고정 truth로 계산한 validation 오차 진단을 게시합니다.

- [validation_mse_by_lead.png](validation_mse_by_lead.png) — lead 1–72별 MSE
- [validation_mse_by_month.png](validation_mse_by_month.png) — 2026-06~09 월별 MSE
- [validation_mse_by_lead.csv](validation_mse_by_lead.csv), [validation_mse_by_month.csv](validation_mse_by_month.csv), [validation_mse_by_band.csv](validation_mse_by_band.csv)

## Sources

| 입력 | SHA256 |
|---|---|
| predictions: `research/submissions/top18-20261020/E1466/predictions.parquet` | `df20b7fc540a4f8f9b1e48dac5203cfbf7d6c4f00934e6ae99c6ecce87bbede3` |
| folds snapshot `bcf5d541…` | `9c8f24f2cd82009cd7d071e99f45c9e15179326f7cccb4f46eb6805788332daf` |
| truth snapshot `bcf5d541…` | `ed02280b78f0ffe0512adb2a51cc92da79058e9abfa6882d5a2e736c62b1d430` |
| history `leaderboard/store/ch-v1/ace.parquet` | `700d7f54c31f2499f30780207c8706d8185d37af0fa2d37268f0eabdf150d8af` |

## 재현

```bash
python scripts/visualize_enlil_validation.py \
  --experiment E1466 \
  --predictions /path/to/e1466_submitted_predictions.parquet \
  --folds /path/to/bcf5d541/folds.parquet \
  --truth /path/to/bcf5d541/truth.parquet \
  --history /path/to/ch-v1/ace.parquet \
  --output-dir docs/visualizations/e1466_td_wsa_huxt72 \
  --ace-latency-hours 2
```

원본 정답·관측·중간 예측 parquet는 이 폴더에 게시하지 않습니다. 이 시각화는 공식창에 이미 노출된 예측의 사후 진단이며, 새로운 일반화 증거가 아닙니다.
