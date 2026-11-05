# 조건부 노이즈 손실의 세 축 — 실행 전 확정

## 목적과 비교 범위

2026-11-03 사용자 요청: 문헌에 근거해 세 실험 축을 설계하고 실행한다.
목표는 주변 타깃의 2성분 적합을 조건부 오차에 바로 적용하지 않고, 이분산성,
로그정규 분포군, 혼합성분의 추가 가치를 통제 비교하는 것이다.

현재 확인된 최고 **단일 모델**은 `e2000fs_feat30_pointweight4_tcn_lb120`:
공식 S=100.002051, RMSE=78.571052. 최고 **GBM**은
`e2000fs_calib11g_serarounds2`: S=100.228038, RMSE=68.186872.
근거는 `research/rerun_df_ffill/runs/{experiment}/submission.json`.
TCN은 모델 구조가 다르므로 GBM 손실 실험과의 인과적 대조군으로 사용하지 않는다.
TCN 최고 성능·공식 SOTA·내부 검증 점수는 혼합하지 않는다.

이 실험은 **loss replacement ablation**이다. 최고 GBM의 모든 30개 특징과
237열 블록 입력을 유지하되, SERA 가중 L2를 다른 손실로 교체하는 효과를 평가한다.
비가중 L2 대조군을 추가해, 꼬리 비용 제거와 분포군 변경의 효과를 구분한다.
새 특징·특징 대체·앙상블·사후 보정·검증 타깃 fit은 없다.

## 사전 확정한 세 축과 공통 대조군

| 축 | 신규 arm | 주 대조군 | 단일 검증 질문 |
|---|---|---|---|
| 1 이분산 | gaussian | plain_l2 | 입력/horizon 의존 분산을 학습하면 오차와 불확실성을 더 적절히 표현하는가? |
| 2 양수/곱셈적 분포 | lognormal | gaussian | 동일한 2모수 조건부 모델에서 로그정규가 정규보다 적합한가? |
| 3 조건부 혼합 | mixture (K=2 고정) | lognormal | 단일 로그정규 대비 2성분이 held-out NLL/calibration을 개선하는가? |

모든 arm은 `sera_l2` 최고 GBM 레시피의 내부 재실행과도 비교한다.
plain_l2는 추가 구조적 가설이 아닌 공통 손실 대조군이다.
분포 arm은 **비가중** NLL이다. 다만 학습 y가 보간되어 있으므로 직접 적합하는 것은
보간 학습 타깃의 조건부 분포다. 실제 관측의 조건부 분포에 대한 적합성은 raw truth의
calibration으로 별도 판단한다. SERA 가중 NLL로 실제 분포를 추정했다고 주장하지 않는다.

## 고정 계약

- 입력: `research/rerun_df_ffill/input/df_ffill_trainy.parquet`.
  기존 정책대로 피처와 학습 y ffill을 유지한다. 보간 자체는 이번 변경 요소가 아니다.
- 채점 truth: `data-v2/input/df.parquet`의 status-0 관측 y만 사용. 보간 truth 채점 금지.
- 유지 특징: `calib11.py`의 CONSTANT 21개 + VARYING 9개 전부.
  블록별 21+24*9=237열, 3개 블록 h1–24/25–48/49–72. 정확한 목록을 metadata에 기록.
  소스 스냅샷과 입력 bytes/mtime으로 실행 정체성을 기록한다.
- 하이퍼파라미터: depth3, lr .03, subsample .8, colsample .7,
  min_child_weight20, lambda5, multi_output_tree, CPU hist.
- seed 42,43,44 모두 실행. 단일 최고 seed 선택 금지.
- rounds=300 고정: 현재 최고 GBM F1/F2 예산. 이번 창으로 탐색하지 않는다.
  분포 모수 수는 2/2/5로 달라지며 공유 vector-leaf tree의 용량 차이는 해석 한계다.
- 분포 최적화: training-only 모수 초기화, 고정 대각 Fisher 전처리,
  양의 scale floor, 공통 distribution-arm step bound. 정확한 식/값은 코드·metadata에 기록.
  이는 분포 손실 구현에 필요한 solver 변경이며 L2와 동일 최적화라고 주장하지 않는다.
  실행 전 고정값: y/500 단위변환, sigma floor=.02, sigma ceiling=2 초과 시 실패,
  max_delta_step=.5. 혼합 초기화는 각 학습 horizon의 log(y/500)에 GMM K=2,
  n_init=3, max_iter=300, random_state=42, reg_covar=.02²이며 수렴 검사를 통과해야 한다.
- 점예측: Gaussian의 학습된 평균, Lognormal의 exp(mu+sigma²/2),
  mixture의 성분 기대값 가중합. 검증값을 이용한 평균 보정 금지.

## 누수 없는 내부 검증 (실행 전 고정)

- 학습: set=train, `origin+144h <= 2025-06-29T00:00Z`.
- 평가: target 범위 `[2025-07-01,2025-10-01)`;
  UTC 00/06/12/18 origin 중 미래 72시간 전체가 그 범위에 있는 창.
- 원래 공식 2026 F1/F2/F3의 정답은 선택·학습·보정에 사용하지 않는다.
- 2025 Q3는 이번 세 축의 사전 지정 내부 평가 창이다. 이 결과를 공식 S나
  SOTA 개선으로 표시하지 않는다. 실패하면 설정을 반복 조정하지 않고 원인을 기록한다.
- 기존 예약 `calib9/10` 및 gaussian/lognormal legacy 재실행은 공식 기간의
  서로 다른 레시피 재현이다. 이 캠페인은 공통 solver/동일 창/복수 seed의 loss ablation으로
  분리한다. 그 대기열·예약·출력을 수정하거나 중복으로 가져와 실행하지 않는다.

## 검증 지표와 채택 조건

- 공식과 동일한 lead별 S = sqrt(RMSE*sqrt(TIMSE))를 계산하고 72 lead 평균을 W로 보고한다.
  함께 MAE, RMSE, bias, y>=550/700 구간 오차, h1–24/25–48/49–72 지표를 기록.
- 확률 arm: 원 단위 NLL(+log(500) 단위복원 포함), PIT 십분위,
  중심 50/90/95% 포함률, 성분 비중·분산 경계 도달 여부.
- 축1은 plain_l2에 밀도가 없으므로 NLL 상대 개선을 판정하지 않는다.
  seed 평균 RMSE가 plain_l2보다 악화되지 않고 90/95% 포함률 절대 오차가 각각
  5 percentage point 이내일 때 가설을 지지한다.
- 축2/3 분포 가설: 직접 부모 대비 seed 평균 NLL 감소 및 90/95% 포함률 오차가
  각 5 percentage point 이내이고 부모보다 절대 포함률 오차가 2 percentage point
  넘게 악화되지 않을 때 지지한다.
- 성능 채택 후보: sera_l2 대비 seed 평균 W 1% 이상 감소, 평균 MAE/RMSE 각각
  2% 넘게 악화되지 않음. 최악 seed·seed 표준편차를 반드시 공개한다.
- 개선 후보에는 7일 시간 블록 paired bootstrap (고정 seed, 1000회)으로 W 차이의
  95% 구간을 보고한다. 구간이 0을 포함하면 통계적 개선으로 확정하지 않는다.
  7일은 겹치는72시간 예측의 의존성을 다루는 작업 가정이며 27일 회귀 등 더 긴
  의존성을 완전히 반영한 불확실성이라고 주장하지 않는다.
- 내부 후보 통과는 공식 제출 승인이 아니다. 공식 동일 조건 검증과 복수 seed,
  기존 최고 단일 모델 비교 전에는 기준선 갱신/리더보드 제출하지 않는다.

## 문헌과 적용 한계 (Exa.ai 검색, 2026-11-03)

1. Collin et al. 2026 PROSWIN https://arxiv.org/html/2609.26683
   태양풍의 조건부 단일 Lognormal NLL. 입력별 sigma 근거, K=2 근거는 아님.
   원 논문 결측 0.5%와 현재 학습 y 결측 원비율 약32.5%는 큰 차이가 있다.
2. Cobos-Maestre et al. 2025 BaLNet
   https://www.swsc-journal.org/articles/swsc/full_html/2025/01/swsc240076/swsc240076.html
   조건부 이분산 Gaussian/ELBO. 정규 비교군의 도메인 근거.
3. Bishop 1994 https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf
   조건부 혼합의 모수 예측. 주변분포의 혼합이 조건부 혼합을 증명하지 않는다.
4. Seitzer et al. 2022 https://arxiv.org/abs/2203.09168
   이분산 NLL의 분산 팽창/평균 학습 저하 가능성. 불확실성 지표와 평균오차 동시 확인.
5. Gneiting 2011 https://doi.org/10.1198/jasa.2011.r10138
   MSE는 정규성 없이 조건부 평균에 일관적. target 가중은 목표 통계량을 바꾼다.
6. Gneiting & Ranjan 2011 https://doi.org/10.1198/jbes.2010.08110
   꼬리 확률과 전체 분포 적합의 목표 분리. NLL 개선이 W 개선을 보장하지 않는다.

## 기존 실패 진단

이전 입력의 calib9/10은 RMSE 개선과 달리 고속 꼬리 및 S 악화가 있었다.
calib10은 95% coverage 81.3%, y>=550 MSE가 point4 대비 약45% 악화.
이는 새 입력에서 검증된 결론이 아니며 수치를 직접 비교하지 않는다.
현재 가설은 '혼합이면 꼬리 점수도 좋아진다'가 아니다.
이번에도 NLL만 개선되면 분포 가설과 운영 성능 채택을 분리한다.

## 자원·완료 계약

GBM은 반드시 CUDA_VISIBLE_DEVICES="" 및 device=cpu로 실행한다.
로컬 가용 CPU/메모리 확인 후 제한된 동시 실행. 각 실행은 독립 tmux/log/out을 사용한다.
golab2는 기존 마지막27개 예약만 예외 허용되어 새 실험을 원격 배정하지 않는다.
colab CLI 가용 여부도 확인하되 GBM CPU 실험을 GPU에 배정하지 않는다.
메모리 부족 시 실행을 축소하며 타 작업을 종료하지 않는다.
종료코드0, 3블록 모델·예측·metadata·평가가 모두 있어야 정상 완료다.
정상 완료 세션은 정리하고 실패 로그는 보존한다.

## 실행 후 결과

### 실행 전 검증과 배치

- 수학/데이터/채점 회귀검사 22개 통과. 세 분포의 해석적 gradient 및 전처리 chain rule을
  유한차분과 대조했고, TIMSE/W는 공식 scorer와 1e-12 수준에서 일치했다.
- 다섯 arm 모두 2-round smoke의 3블록·예측·채점을 exit0으로 완료했다.
- Colab CPU 런타임은 CPU2개, RAM12GiB 중 available11GiB, GPU 없음 확인.
  XGBoost는 로컬 기준선과 같은2.1.4로 고정했다.
- Colab 전체 파일 업로드 제한으로, 필요한 행/열만 **값 변경 없이** 투영한33MB 입력과
  원 관측 truth를 준비했다. parquet 왕복 dataframe 완전일치를 확인했다.
  `input/projection.json`에 원본 정체성과 투영 규칙을 기록한다.
- 로컬 admission이 회복된 뒤 SERA/L2 대조군을 각각 독립 tmux로 시작했다.
  분포 arm은 별도 tmux와 독립 출력으로 실행하며 타 캠페인 예약은 수정하지 않는다.

최종 결과·실패 원인·채택 여부는 검증 후 추가한다.

### 장시간 실행의 완료 검증

`noise-watch` tmux의 `controller.py`가60초마다 로컬/Colab 진행 상태를
`status.json`에 기록한다. 종료코드·300회 학습 곡선·237열 특징·356×72 예측 격자·모수·평가를
검사하고, Colab 산출물은 회수·검증한 뒤 해당 CPU 런타임만 해제한다.
15회가 모두 유효하게 완료되면 `finalize.py`가 원본 관측 truth와 저장 truth의 일치 및
점수 재계산을 확인하고 `summary.json`, `seed_results.csv`, `RESULTS.md`를 생성한다.
조건을 통과한 내부 후보에만 사전 고정 bootstrap을 수행한다.
실패 시 로그를 보존하고 실패 상태를 기록하며 파라미터를 바꾸어 재실행하지 않는다.
수학·데이터·채점·컨트롤러 합계40개 테스트 및 Ruff가 통과했다.
실행 명령을 제출했다는 이유로 실험 완료를 선언하지 않으며, 현재 상태는 `status.json`을 따른다.
