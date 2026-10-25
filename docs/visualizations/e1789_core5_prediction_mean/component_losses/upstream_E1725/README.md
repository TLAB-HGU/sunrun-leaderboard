# E1769 하위 모델 E1725 — 실제 공식 학습 loss

E1789 → E1769 → E1725 계보의 **실제 공식 예측에 사용된 학습 모델**에서 `trace`를 읽었습니다. Seed 0·1은 E1745에서 학습 후 E1759가 검증해 재사용했고, seed 2는 E1759에서 학습했습니다. 각 시드 **300 rounds**, 학습 행 수 **230,626**입니다. 모델 재학습이나 추론은 하지 않았습니다.

![저장 모델에서 추출한 실제 training objective](training_loss.png)

왼쪽은 원본 custom objective를 학습 행 수로 나눈 값, 오른쪽은 초기값 대비 비율입니다. 0번은 학습 전 원본 `before`, 1~300번은 각 round의 원본 `after`입니다. **MSE가 아니며 E1789 전체의 loss도 아닙니다.** 반복별 validation loss는 기록되지 않아 validation 선은 없습니다.

## 원본 이력

| Seed | 실제 학습 실행 | JSON | CSV |
| --- | --- | --- | --- |
| 0 | E1745/E1725 → E1759 재사용 | [trace JSON](training_trace_s0.json) | [trace CSV](training_trace_s0.csv) |
| 1 | E1745/E1725 → E1759 재사용 | [trace JSON](training_trace_s1.json) | [trace CSV](training_trace_s1.csv) |
| 2 | E1759/E1725 | [trace JSON](training_trace_s2.json) | [trace CSV](training_trace_s2.csv) |

원본 필드 `round`, `before`, `after`, `leaves`, `leaf_gradient_norm`을 그대로 보존했습니다. 각 시드의 300개 기록과 round 연속성, `after`→다음 `before` 일치, 목적함수 감소를 확인했습니다.

## 목적함수

원본 [engine.py](original_objective_engine.py)의 정의는 다음과 같습니다.

```text
L = 0.5 × [Σᵢ rᵢ² + (γ−1) × Σg n_g × mean(r_g)²]
γ = 2
group = (source_fold, forecast lead)
r = prediction − training residual target
```

학습 목표 자체가 residual-target에 대한 group-mean anchor objective입니다. 단위는 비정규화 합산 목적함수이며, 그림의 행 수 정규화도 이를 표준 MSE로 바꾸지는 않습니다.

## 공식 제출 계보 및 해시

[provenance.json](provenance.json)에 모델·영수증·trace 해시와 실제 학습 실행을 기록했습니다. 원본 영수증의 모델 SHA256을 확인한 뒤 저장 모델을 읽고, 결과 예측의 SHA256이 E1769 공식 입력 manifest의 E1725 멤버와 일치하는지도 확인했습니다.

- [공식 입력 manifest](original-official-input-manifest.json)
- [E1759 공식 멤버 영수증](official_member_receipt.json)
- [Seed 0 재사용 영수증](reuse_s0.json), [Seed 1 재사용 영수증](reuse_s1.json)
- 실제 학습 영수증: [Seed 0](receipt_s0.json), [Seed 1](receipt_s1.json), [Seed 2](receipt_s2.json)

모델 파일, 학습 원본 행과 target은 이 폴더에 게시하지 않았습니다. 그래프는 게시된 trace만으로 다시 생성할 수 있습니다.

```bash
python scripts/export_e1789_training_trace.py --mode plot \
  --output docs/visualizations/e1789_core5_prediction_mean/component_losses/upstream_E1725
```

원본 모델에서 다시 추출하려면 원래 작업 공간과 joblib/XGBoost 환경에서 같은 스크립트에 `--mode extract --root /path/to/sunrun`을 지정합니다.
