# vanila_mse

2026-11-12 사용자 요청에 따른 게시 준비. 31 source / 215 expanded 특징의 offset-roll 구조에서
손실만 순수 MSE(gradient=prediction-target, Hessian=1)로 바꿔 새로 학습했다.
Tail 가중치, spectral 항, direct-S 항은 없다. 각 블록 900 rounds, seed 42/43/44,
학습 cutoff·144h maturity·prior/full 분리·training-only frozen offset·clip은 유지했다.

세 seed 예측의 고정 등가중 평균이며 공식 평가 정답으로 계수나 가중치를 적합하지 않았다.

| 구간 | S | TIMSE | RMSE |
|---|---:|---:|---:|
| Total | 100.775874 | 24932.356726 | 65.881206 |
| F1 | 122.112963 | 35421.477619 | 82.263806 |
| F2 | 81.017198 | 12936.267801 | 60.053956 |
| F3 | 63.373792 | 6592.104452 | 52.324697 |

개별 seed 점수 평균 100.877281, 표본 SD 0.276684,
최악 101.061462. 위 표의 mean3 예측 점수와 구분한다.
사용자 요청 게시이며, 게시 자체는 성능 개선 판정이나 기준선 승격이 아니다.

모든 seed 학습 exit 0, 9개 seed×기간 저장 모델 예측 재생성 오차 0,
원 parquet와 비트 단위 동일, 공식 scorer 재채점 일치, 누락 대체 0을 확인했다.
상세 검증은 verification.json, 실행 metadata와 로그는 runs/mse_s{42,43,44}에 있다.
weights는 로컬에 보관하고 Git에는 올리지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/vanila_mse/experiment.py --threads 6 --out /tmp/vanila-mse-retrain
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/vanila_mse/experiment.py --predict-only --weights research/vanila_mse/weights --out /tmp/vanila-mse-replay
```
