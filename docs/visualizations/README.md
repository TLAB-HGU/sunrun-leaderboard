# 리더보드 예측 GIF

## 2026-10-25 추가

[E1466 TD-WSA HUXt72 GIF와 공식창 검증 오차 진단](e1466_td_wsa_huxt72/)을 추가했습니다. 공식 MSE **3246.254994**, 제출 당시 비앙상블 단일 모형 최고 기록입니다. XGBoost backbone은 고정 900 rounds로 train/validation loss history가 없고, field encoder의 `curves_s*.json`은 원 run 디렉터리 삭제로 남아 있지 않습니다. 등록 해시와 학습 시간은 [loss_history_audit.json](e1466_td_wsa_huxt72/loss_history_audit.json)에 보존했으며 곡선을 합성하지 않았습니다.

[Exa 기반 실험의 실제 train·validation 곡선](exa_earlystop_20261025/)을 추가했습니다. E1901의 9개 실제 학습 이력, 272회 선택 과정과 E1900/E1904 외부 비교를 포함합니다. 내부 검증 개선은 외부 MSE 개선으로 이어지지 않아 후보는 미채택입니다.

[Train·validation 두 곡선](e1789_core5_prediction_mean/component_losses/paired_train_validation/)을 추가했습니다. E1789 계보의 E1725·E1711 저장 모델을 고정한 과거 fold 3의 단계별 재평가 결과입니다.

[E1789 구성 모델별 loss 기록](e1789_core5_prediction_mean/component_losses/)과 [E1769 하위 E1725의 실제 3시드 학습 곡선](e1789_core5_prediction_mean/component_losses/upstream_E1725/)도 추가했습니다. 각 자료에서 학습 objective, OOF 검증 MSE, 공식/test MSE를 구분했습니다.

[E1789 Core5 · 공식 평가 오차 그래프 및 월별 GIF](e1789_core5_prediction_mean/)를 추가했습니다. 게시 기준 리더보드 1위, 공식 MSE **3177.232884**이며 실제 제출한 세 시드 예측 평균을 사용합니다. 고정 예측 결합이므로 E1789 자체의 train/validation 학습 곡선은 없습니다. 대신 공식/test 오차를 명확히 구분해 표시했습니다.

## 기존 시각화

공식 평가 기간인 2026년 6~9월의 예측을 월별 GIF로 확인할 수 있습니다. 아래 순위는 **2026-10-20** 게시 상태 기준이며, 각 그림은 리더보드에 실제 제출한 예측을 사용합니다.

| 당시 순위 | 모델 | 작성자 | 공식 MSE | GIF |
| ---: | --- | --- | ---: | --- |
| 1 | E1604 · HSSPEAK25 | seongeun | 3222.882998 | [월별 GIF](e1604_hsspeak25/) |
| 2 | E1593 · HSSPEAK72 | gjc-main | 3225.384369 | [월별 GIF](hsspeak72/) |
| 3 | E1466 · TD-WSA HUXt72 | dispatch-w0 | 3246.254994 | [월별 GIF](e1466_td_wsa_huxt72/) |

각 모델의 세 시드 예측 평균을 표시합니다. 최근 168시간 관측, 이후 72시간 실제값과 예측을 동일한 200~800 km/s Y축으로 비교합니다. 월별 프레임 수는 720 / 744 / 744 / 625로 모델당 전체 2,833개 시작점을 포함하며, 프레임당 100ms로 반복 재생합니다. 시각화는 저장 예측을 읽으며 학습이나 예측 변경을 수행하지 않습니다. 기존 입력 계약과 현재 연구 기준의 차이는 각 모델 설명에 기재했습니다.

기존 시각화: [E0472](xgb_d76_lead_moe_resid_dv_enlil/), [E0471](xgb_d76_lead_moe_enlil/), [E1149 ACE+2h](e1149_l72cond_ace2h/), [E1149 ACE+0h 비교](e1149_l72cond_ace0h/), [E1466 TD-WSA HUXt72](e1466_td_wsa_huxt72/).
