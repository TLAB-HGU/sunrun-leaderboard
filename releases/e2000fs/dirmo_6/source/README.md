# forecast_strategies_e2000: E2000 계열 다중 horizon 예측 전략 비교 (Direct 제외)

이 스터디는 E2000 입력(111 features)과 frozen XGBoost recipe(`experiments/e2000/model.py:ModelConfig`: 900 rounds, depth 3, lr 0.03, seed 42)를 그대로 두고, multi-step 예측 전략만 바꿔 비교합니다.
평가는 리더보드 공식 구간에서 합니다. 타깃은 2026-06-01 ~ 09-30, origin은 2,833개 × h 1..72입니다.
결과는 `forecast_strategies_e2000.ipynb`의 리더보드 셀에 있으며, 공식 scorer 지표에 **MSE(truth ≥ 550 km/s)** 열을 추가했습니다.

**보고 전용 분석입니다.** lineage 후보도 아니고 업로드도 하지 않습니다.
σ와 Stratify 조합은 내부 검증으로만 골랐고, 공식 점수는 선택에 쓰지 않았습니다.
Rectify 계열의 residual head는 CUMULATIVE-WORKER-CONTRACT의 lineage 규칙(single direct only)과 맞지 않으므로 분석 목적으로만 학습했습니다.

## 전략
| 전략 | 구현 |
|---|---|
| Recursive | RecMO(σ=1): 1-step 공유 모델을 72회 반복 |
| DirRec | DirRecMO(σ=1): h별 모델 72개, 입력에 1..h−1 궤적 포함 |
| MIMO | DirMO(σ=72): XGBoost joint multi-output tree(`multi_strategy="multi_output_tree"`) |
| RecMO / DirMO / DirRecMO (σ=6, 12, 24) | 블록 공유 반복 / 블록별 독립 / 블록별 + 이전 블록 궤적 |
| Rectify | Recursive base + h별 residual |
| RectifyMO(σ) | RecMO base + DirMO residual. σ는 내부 검증으로 선택 |
| Stratify | base·residual ∈ {RecMO, DirMO, DirRecMO} × σ ∈ {6, 24}의 18개 조합 중 내부 검증으로 선택 |

- 블록 입력은 다음으로 구성됩니다.
  - origin 특징 90개
  - 최근 속도 24h
  - horizon 특징 20개 × 블록 길이(rec26–28, rec_mean, SUVI strip 15개, enlil_dt_target)
  - 공유 모델(Recursive, RecMO)에는 `lead`를 추가합니다.
- 학습은 teacher forcing입니다. 학습 시 lag와 궤적에는 실측값(ffill)을 넣고, 추론 시에는 자기 예측만 씁니다.
- residual 모델은 base의 purged out-of-fold 예측으로 학습합니다(시간 블록 K=4, 72h purge).

## E2000과 다른 점 (사용자 결정 2026-10-26)
- **2시간 지연 없음.** `ace_*`, `v_*`, `speed_d24`, `epam_p1_d6`는 같은 production 함수(`headroom.ace_features`, `info_ablation.speed_features`)로 origin 시각(lag 0)에서 다시 계산했습니다.
  - 같은 함수를 lag 2h로 돌리면 frozen parquet가 그대로 재현됩니다. train과 official 각각 256 origin에서 불일치 0건입니다.
  - 따라서 E1983, E1987, E1988(lag 2h)과는 입력이 다르고, 이 run들은 참고 행으로만 표기합니다.
- **Prefill.** 남은 특징 결측은 origin 시간순으로 ffill했습니다. 학습 타깃과 궤적은 ffill된 시간별 ACE 속도(`filled_speed_kms`)입니다. 이는 공식 truth와 같은 시계열이며, 모든 셀에서 일치하는 것을 확인했습니다.
  - E2000은 관측 라벨만 썼지만, multi-output 학습에는 결측 없는 타깃 벡터가 필요하므로 모든 전략에 같은 규칙을 적용했습니다.
- 모든 전략이 같은 학습 origin 집합을 씁니다. 내부 단계는 72개 타깃이 모두 2026-03-01 이전인 origin, final 단계는 2026-05-29 이전인 origin입니다.

## 지표
- `mse`: 공식 순위 지표입니다. ffill 행을 포함한 203,976행으로 계산하며, `lbeval.score_file`과 자체 계산이 일치하는지 assert로 확인합니다.
- `mse_obs`: 관측 행만으로 계산한 MSE입니다.
- **`mse550`**: 공식 truth `target_kms ≥ 550`인 셀의 MSE입니다(ffill 포함, n=23,976, 전체의 11.75%).
  - `bias550`은 mean(pred − truth)입니다.
  - `mse550_obs`는 관측 행 중 ≥550인 셀로 계산합니다(n=19,440).
- LLH, LHH, HLH, HHH는 고난도 regime 4셀이고, `f1`은 unique-event F1입니다.
- horizon 밴드별 MSE와 E1813(3277.12), E1789(3180.20) 대비 Δ%도 함께 냅니다.
- `val_mse_obs`는 내부 검증 MSE입니다(origin 2026-03-04 ~ 05-25, 관측 행). 선택 근거로 쓴 값입니다.

## 실행
```bash
cd analysis/forecast_strategies_e2000
uv venv --python 3.12 .venv && VIRTUAL_ENV=.venv uv pip install -r requirements.txt
.venv/bin/python -m pytest -q test_strategies.py          # 17 tests
# smoke (5 rounds, origin 1/8):
FS_ROUNDS=5 FS_SUBSAMPLE=8 FS_NJOBS=4 FS_OUT=/tmp/smoke_out .venv/bin/jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 forecast_strategies_e2000.ipynb --output /tmp/smoke.ipynb
# full (E1987/E1988가 실행 중이면 FS_IGNORE_BUSY=1 필요):
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 forecast_strategies_e2000.ipynb
```
- `make_notebook.py`는 노트북 생성기입니다(`python make_notebook.py forecast_strategies_e2000.ipynb`).
- 예측은 전략별로 `outputs/predictions/<stage>/`에 체크포인트로 저장되므로, 다시 실행하면 이어서 진행합니다.
- 데이터 캐시는 `outputs/cache/`에 있으며, receipt에 입력 해시와 parity 결과가 들어 있습니다.

## 파일
- `data.py`: lag-0 재계산, prefill, Bundle 구성, 분할
- `strategies.py`: RecMO, DirMO, DirRecMO, Rectify와 round별 loss 로깅
- `run.py`: stage 드라이버, 체크포인트, 선택
- `evaluate.py`: 학습 코드와 분리된 채점 모듈. leaderboard snapshot을 읽는 곳은 여기뿐입니다.
- `outputs/`(커밋하지 않음)
  - `leaderboard.csv`, `leaderboard.md`
  - `predictions/official/*.parquet`
  - `loss_history/<stage>/<strategy>/*_s42.csv`(`step,train_mse,validation_mse`)
  - 그림

## 검증
- `test_strategies.py` 17개 테스트가 통과합니다. 다루는 내용:
  - lag-0 future-poison과 positive control
  - prefill 인과성
  - 블록 폭과 정렬
  - 추론 입력에 미래 truth나 NaN이 들어가지 않음
  - purged fold
  - 분할 cutoff
  - mse550 손계산
- smoke 실행에서 노트북 전체 셀이 오류 없이 돌았습니다(2026-10-26).
- 전체 실행 결과는 아래 결과 섹션에 기록합니다.

## 결과
(전체 실행 후 기록)
