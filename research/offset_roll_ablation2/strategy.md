# offset_roll 고도화 2차 ablation — 8개 축 × 3 seed (golab2 CPU)

작성 2026-11-05, 실행 전 사전 확정. 사용자 요청: e2000fs_calib_offset_roll 고도화에 필요한 ablation을 dsh(DeepSeek Harness)에 Exa 문헌조사로 요청하고, 그 결과로 8개 실험 축을 설계해 sunrun 큐에 반영.

## 1. 비교 기준 (분리)

| 구분 | 실험 | Total S | 비고 |
|---|---|---:|---|
| 기준선 A (고정) | `e2000fs_calib_offset_roll` (= 1차 ablation `baseline_s42`) | 98.762687 | 리더보드 1위, seed42 |
| 리더보드 SOTA | 동일 (현재 1위) | 98.762687 | |
| 최종 목표 (SOTA 대비 2%) | — | ≤ 96.787 | 별도 판정 |
| 제출 후보 기준 (1%) | — | 3-seed 평균 ΔS ≤ −0.988 | 아래 5절 |

기준선 소스: `research/rerun_df_ffill/runs/e2000fs_calib_offset_roll/experiment.py`, 실행 코드는 1차 ablation `research/calib_offset_roll_ablation/experiment.py`에서 파생. 공식 관측 revision `920588c2a716130e19b145acbbd91ace4758faf2`, scorer `competition.py` 고정.

## 2. 1차 ablation 결과(입력 근거, seed42 단일)

offset 제거 +0.99 (CI [−0.62, 2.17]), spectral 제거 +0.79 (CI [−0.47, 1.75]), tail 가중 4→1 +4.49 (CI [1.07, 7.16]). offset은 y<550을 악화시키고 y≥550(놓친 초과, bias −108)을 개선 → 지평당 스칼라 하나로는 두 구간을 동시에 보정하지 못한다. 단일 seed라 어떤 기여도 확정되지 않았다 → 이번 라운드는 모든 축을 seed 42/43/44로 실행하고 기준선도 3 seed로 재현한다.

## 3. 문헌조사 (dsh + Exa)

- 실행: `literature/run_dsh_literature.sh` (dsh headless 프로필 + 실행별 Exa MCP patch `literature/exa.patch.yml`), 요청문 `literature/dsh-prompt.md`. 최종 쓰기 단계가 inference timeout으로 끊겨 같은 세션을 `resume_dsh_literature.sh`로 재개해 기록. 결과 `literature/evidence.md`(29개 Exa 쿼리, HTTP 429 0건, 출처별 [V]/[I]/[S]/[U] 검증 등급).
- 직접 spot-check: Ben Taieb et al. arXiv:1108.3259 초록의 "Multiple-Output strategies are the best performing approaches" 일치 확인. "s is critical" 문장은 본문이며 dsh가 [V]로 표기(초록에는 없음).
- 어떤 문헌도 이 대회의 S 지표나 이 레시피에서의 개선을 입증하지 않는다(evidence.md "What the corpus does not establish").

### dsh 순위와 이번 채택의 차이 (근거)

| dsh 순위 | 제안 | 채택 | 이유 |
|---|---|---|---|
| 1 | 예측값 조건부 isotonic 보정 + blend | **채택 (O1)** | 지평당 스칼라의 구조적 한계를 직접 겨냥. Gneiting & Resin 2023 (isotonic/PAV, [V]). |
| 2 | 상단 꼬리 일측 확장 변환 (Collin 2024) | **기각** | 로컬 근거: feat30 진단(2026-10-28)에서 variance-matching de-shrink +3.1%, quantile mapping +2.7% MSE 악화. Collin 계열도 MSE/RMSE 악화 보고. |
| 3 | 꼬리 가중 smooth ramp | **채택 (W1)** | SERA (Ribeiro & Moniz 2020, Silva et al. 2022 [V]), LDS (Yang 2021 [V]). |
| 4 | 지수감쇠 평균 + cap | **채택 (O2, cap 제외)** | Cui et al. 2012 decaying average [V], Guan(AMS 초록, 비심사) 전환기 실패 사례. 단일 변경 원칙상 cap은 분리(보류). |
| 5 | block 24→12 | **채택 (M1)** | Ben Taieb 2012/2016 DIRMO s 선택 [V]. 72는 bias 증가 방향이라 12를 검정. |
| 6 | spectral 0.1→0.3 | **채택 (L1)** | FreDF (Wang et al. 2024 [V]) α 증가 방향. 규모 이전성은 미확립. |
| 7 | LOCO 특징 그룹 | **채택 (F1: strip 그룹)** | Hooker et al. 2021 [V]: 재학습형 LOCO만 허용. strip은 5개 lead-varying 특징 → 블록당 237열 중 120열로 차원 최대 그룹이며, 과거 "차원 증가 손해" 기록(closed-axes round9)과 직접 연결. |
| 9 | 2–3 fold cross-fit offset | 기각 | prior 모형 잔차가 이미 144h purge 밖 out-of-sample → DML 동기 부재(dsh 자체 판단). |
| (d) | rounds 탐색 | **단일 사전등록 1개 (C1: 900→300)** | dsh는 탐색 금지·고정 1회 허용. 300은 로컬 round9 walk-forward(3분기 모두 300 우세, 다른 부모) 근거. |
| — | tail 가중 dose 4→6 | **채택 (W2)** | 1차에서 최대 효과 성분(+4.49)의 용량 반응 방향 확인. 사후 LB 튜닝 아님: 사전 고정 1개 점, 결과로 추가 점 탐색하지 않음. |
| (d) | hurdle/게이팅, 무제약 quantile map, lead별 고정 가중 | 기각 | evidence.md (d)절. |

## 4. 8개 실험 축 (각각 기준선 A 대비 단일 변경)

| ID | 단일 변경 | 분류 | 재학습 | 가설 / 기대 메커니즘 | 실패 시 해석 |
|---|---|---|---|---|---|
| O1 `o1_offset_isotonic` | offset `α·mean(r_h)` → `α·(iso_h(f)−f)`, iso는 prior 예측 f→정답의 단조 회귀(마지막 336행), α=0.25 유지, 보정 범위 밖은 끝 knot 보정 고정 | 보정 형태 replacement | 아니오(기준선 모형 replay) | 몸통 과대(f 낮은 구간 음수 보정)와 꼬리 과소(f 높은 구간 양수 보정)를 분리 | 예측값 조건부 편향이 없거나(feat30에서 slope≈1.01–1.05) 336행에서 iso가 잡음 |
| O2 `o2_offset_ewma` | 최근 336행 단순평균 → 3개월 전체 지수감쇠 평균(반감기 336행) | 보정 추정량 replacement | 아니오 | 더 긴 기억으로 offset 분산 감소, 기간 간 부호 뒤집힘 완화 | 편향이 비정상적이라 긴 기억이 오래된 편향을 끌어옴(Guan) |
| W1 `w1_tail_ramp` | 가중 `4·1[y≥550]` → `1+3·clip((y−480)/140,0,1)` (최대 4 동일, lead별 평균 1 정규화 동일) | 가중 형태 replacement | 예 | 550 절벽 제거, 450–620 전이 구간 gradient 연속 | 절벽 자체가 놓친 초과를 강하게 밀어주는 역할 |
| W2 `w2_tail_weight6` | 꼬리 가중 4 → 6 | 용량 반응 | 예 | 놓친 초과(꼬리 SE 70%) 과소예측 축소 | 몸통 편향(+14 계열) 증가가 상쇄 |
| L1 `l1_spectral_c03` | spectral 계수 0.1 → 0.3 (Hessian = w + 0.3) | 손실 계수 | 예 | lead 간 결합 오차(저주파=블록 내 수준 편향) 억제 | 계수 증가가 Hessian을 키워 학습 속도만 낮춤 |
| M1 `m1_block12` | 출력 블록 24 → 12 (6개 multi-output booster) | 출력 구조 | 예 | 블록 내 bias↓(Ben Taieb & Atiya 2016) | 공유 분할 정규화 손실로 분산↑ |
| C1 `c1_rounds300` | boosting rounds 900 → 300 | 용량 | 예 | 기간 drift 하 과적합 감소 | 과소적합(TIMSE↑) |
| F1 `f1_drop_strip` | strip 5개(`strip_v350_logmed, strip_v450_logmed, strip_v550_dark, strip_v650_dark, strip_v750_logmed`) 제거, 블록당 237→117열 | feature-group LOCO(제거) | 예 | 차원 축소가 n≈수만 origin에서 일반화 개선 | strip이 HSS 도착 정보를 담음(제거 시 꼬리 악화) |

모든 축: 나머지 30개 특징(F1만 25개), XGBoost 파라미터(depth3, lr .03, subsample .8, colsample .7, mcw 20, λ5, multi_output_tree), 학습 cutoff, 144h maturity, 3개월 prior/calibration 분할, clipping은 기준선 그대로. O1/O2는 동일 seed 기준선의 prior/full 모형 예측을 재사용하므로 모형 차이가 0이고 보정만 다르다.

특징 유지 검증: `test_experiment.py::test_all_preregistered_configs_are_accepted`가 모든 비기준선 실행이 DEFAULTS 대비 정확히 1개 키만 바꾸는지, replay가 학습 키를 바꾸지 않는지 검사한다. 각 실행 `F{n}_metadata.json`에 특징 이름·개수·확장열·학습/질의 범위·파라미터·seed·버전을 기록한다. 코드 diff: `implementation-vs-parent-ablation.diff`.

## 5. 평가와 판정 (사전 고정)

- 평가: 리더보드 F1/F2/F3만, 공식 scorer, 내부 검증 없음(2026-11-04 규칙). 학습 holdout(prior/calibration 3개월)은 적합의 일부. 공식 정답은 채점에만 사용.
- 재현 게이트: `baseline_s42`가 저장 참조 예측과 최대 절대차 ≤1e-6, S 차 ≤1e-6이어야 다른 실행 진행(기본값 = 기준선 정확 재현 확인).
- 대조: 같은 seed 기준선 대비 ΔS(변경−기준선), seed별·평균·최악, 168h 원형 블록 bootstrap 2000회 95% CI, RMSE/TRMSE/TIMSE/MAE/bias, F별 ΔS, offset 구간 분해.
- **채택(다음 누적 기준선 후보)**: 3-seed 평균 ΔS ≤ −0.988(기준선의 1%), bootstrap 95% CI 상한 < 0, 3개 seed 모두 ΔS < 0, RMSE·MAE 악화 ≤1%, 어느 F 기간도 +2% 초과 악화 없음.
- **기각**: 위 조건 중 하나라도 불충족. CI가 0을 포함하는 것만으로 "효과 없음"을 확정하지 않고 "미확정"으로 기록.
- 다중비교: 8개 축 동시 검정 → BH 보정 q 함께 보고(analyze_results.py). 리더보드 점수는 탐색 신호로 쓰지 않으며 결과에 따라 계수·임계값을 재탐색하지 않는다. 채택 축이 둘 이상이면 다음 라운드에서 A, B, A+B(명시적 합집합)를 별도 실행해 상호작용을 확인한다.
- 제출: 이 캠페인은 진단용, `complete.json`에 `publish: not_required`. 채택 축이 나오면 별도 승인 후 제출.

## 6. 실행 계획 (golab2 CPU)

- 원격 루트 `/home/yehwan/sunrun-offset-roll-ablation2-20261105`, scheduler `ablation_scheduler.py`(tmux `srab2-controller-20261105`), 실험 tmux `srab2-<id>-<name>`, `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=threads`.
- 27개 실행: 기준선 3(seed 42/43/44) → 각 축 seed42 8개 → seed43 8개 → seed44 8개. 재학습 21개(약 1h25m/4 threads), replay 6개(수십 초).
- 자원: max_jobs 4, 4 threads/job, CPU 예산 20(타 작업 load 포함 측정), 메모리 여유 4 GiB + 작업당 3 GiB 추정. 예상 소요 약 10–11 h.
- TUI: golab2 scheduler 레이아웃(`~/sunrun-*/assignment.json` + `scheduler-state.json`), 대기 항목은 대기열, 실행은 `train.log` JSON 진행 레코드, 종료 `train.exit`, `complete.json`.

## 7. 실행 상태 / 결과 (실행 후 누적 기록)

- 2026-11-05 01:43 KST: deploy 완료(입력 225,755,615 bytes 확인), 원격 smoke 3종(F1, 2 rounds, 부분표본) exit 0 — isotonic offset, ramp+block12(확장열 129), strip 제거+300r+spectral .3+ewma+tail6(특징 25, 확장열 117). `ablation_scheduler.py --check` 27개 유효. 로컬 pytest 34 passed(원격 venv에는 pytest 없음, 동일 버전 sklearn 1.9.1/xgboost 2.1.4).
- controller `srab2-controller-20261105` 시작, `baseline_s42` 실행 중, 26개 대기. `sunrun --once`에서 golab2 실행 중/대기열 표시 확인.
- 결과·원인 분석·채택 여부: 완료 후 기록.

- 2026-11-05 09:4x KST: **사용자 지시로 게시 정책 변경** — "로컬 채점(게시 안 함) 항목 전부 게시". 5절의 `publish: not_required`를 무시하고 `publish_lb.py`로 1차(abl1) 3개 + 2차(abl2) 25개(+ f1_drop_strip_s44 완료 시 자동, tmux `srab2-publish-s44`)를 `seongeun/e2000fs_calib_offset_roll_<abl1|abl2>_<run>`으로 제출. 두 `baseline_s42`는 게시된 `e2000fs_calib_offset_roll`과 예측이 비트 단위로 같아 중복 행이 되므로 제외. 공식 dry-run S = 로컬 S(|ΔS| ≤ 1e-6) 확인 후 제출, 영수증 `lb_publish/receipts.json`, golab2 run의 `submission.json`은 실제 영수증으로 교체(원본 `submission.local.json`).
