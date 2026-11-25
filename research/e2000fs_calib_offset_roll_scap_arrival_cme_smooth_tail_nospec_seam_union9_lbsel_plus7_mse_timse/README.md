# plus7의 MSE·TIMSE 분리 학습 앙상블

기준선은 plus7의 Smooth-tail MSE 600라운드와 S-cap 180라운드다. 새 실험은 같은 345/345/350열과 학습 구간으로 순수 MSE 모델과 순수 TIMSE 모델을 각각 780라운드 학습한다. TIMSE의 100분위 계수는 각 적합의 학습 정답에서 리드별로 계산하며, 평균 헤시안이 1이 되도록 정규화한다. 모델별 offset은 각각 학습 자료로 산출한다. 총 부스팅 연산량은 두 배다.

같은 시드의 두 예측을 50:50으로 평균한 뒤 plus7의 seam, ridge, 위상 전진 및 시드 합의 보정을 적용한다. 혼합 비율은 공식 평가로 튜닝하지 않는다. 시드 42·43·44를 모두 사용해 기존 plus7과 Mean3 및 같은 시드별 공식 S를 비교한다. 기존 입력의 후보 선정 이력 때문에 탐색적 결과에 해당한다.

[TIMSE 정의](../../leaderboard/scorer/competition.py) · [기존 TIMSE 구현](../vanila_timse/experiment.py)
