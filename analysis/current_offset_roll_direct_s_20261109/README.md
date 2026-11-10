# 현재 offset-roll 직접 S 최적화 ablation

현재 31개 특징 모델의 A(기존 손실), B(직접 S), C(직접 S+spectral)를
seeds42/43/44에서 새로 학습하는 replacement ablation이다.

아직 성능 결과를 의미하지 않는다. `status.json`은 실제 실행 상태,
`research/current_offset_roll_direct_s_20261109/strategy.md`는 사전 확정 조건이다.
기존 실험의 CPU 자원을 침범하지 않는 대기열로 smoke→9개 학습→채점을 수행한다.
모든 예측 동결과 검증 이후 `report.md`와 `report.json`이 자동 생성된다.
리더보드 게시는 하지 않는다.
