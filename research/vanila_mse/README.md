# vanila_mse

사용자 요청(2026-11-12)에 따른 순수 MSE replacement ablation. 현재 31-source/215-expanded offset-roll 구조, 900 rounds, prior/full·frozen offset을 유지하고, tail 가중치와 spectral 항을 모두 제거한다. seed42/43/44를 새로 학습한 후 고정 등가중 예측 평균을 `vanila_mse`로 게시한다. 점수를 보고 seed를 선택하거나 보정 가중치를 적합하지 않는다.

사전 가설·비교대상·평가조건은 strategy.md, 원본 대비 변경은 baseline-vs-mse.diff에 기록한다. 동일 데이터의 native XGBoost reg:squarederror와 축소 학습 부스터·예측·offset이 정확히 일치한다. 학습 및 게시 완료 여부는 training-complete.json, submission.json, publication-verification.json으로 확인한다. 단순 실행 요청이나 tmux 생성은 완료가 아니다.

CPU 실행: `CUDA_VISIBLE_DEVICES='' .venv/bin/python research/vanila_mse/experiment.py --seeds 42 --threads 6 --device cpu --out research/vanila_mse/runs/mse_s42`.
