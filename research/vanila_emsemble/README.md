# vanila_emsemble — fixed 50:50 prediction ensemble

2026-11-12 user-requested name preserved exactly (emsemble). State: runnable; dispatch follows validated TIMSE completion.
Dependencies: validated completed vanila_mse AND vanila_timse cohorts, seeds 42/43/44. The experiment.py runner verifies the source cohorts and assembles all predictions before loading official observations.

Hypothesis: equal averaging of separately trained MSE and TIMSE predictors may reduce complementary errors; improvement is unproven. A = pure MSE mean3, B = pure TIMSE mean3, A+B = 0.5*A + 0.5*B. This is prediction averaging, not mixed-loss training. No retraining, feature replacement, target-dependent correction, seed selection, ratio search, or fitted ensemble coefficients.

Freeze prediction ratio at 50:50 before seeing TIMSE results. Match origin/horizon keys exactly, reject missing/duplicate rows and nonfinite values; verify identical feature/data/cutoff protocol and seed cohorts. Reuse the completed prediction artifacts and record source byte sizes, modification times, row counts, columns and time ranges, versions, git commit, source experiment IDs and arithmetic verification. No leaderboard target is used in assembly. The fixed equal-seed means imply six source predictions each weighted 1/6.

Score once assembled on official F1/F2/F3 observations using leaderboard/scorer/competition.py; report A, B, A+B separately, seed-paired ensembles and mean/SD/worst plus ensemble mean3, S/W/MAE/RMSE/TIMSE. No internal folds or leaderboard-score tuning. Candidate gate: W at least 1% better than the matched best single model with no serious auxiliary deterioration; SOTA 2% goal is assessed separately. No automatic baseline adoption or unconditional publication.

Run on CPU with CUDA_VISIBLE_DEVICES='', explicit --experiment vanila_emsemble --out ... --device cpu --threads 2, after checking resources. Independent tmux/log and standard runs/{name}/launch.json, JSON progress, period artifacts, train.exit and complete.json; mark publish=not_required with reason if not eligible. Confirm sunrun visibility and preserve errors. Completion requires validated source runs, exact fixed arithmetic, official scores and complete provenance. Reference: the official scorer defines evaluation; the arithmetic averaging hypothesis makes no literature-backed performance guarantee.


## 실행 및 산출물

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/vanila_emsemble/experiment.py \
  --mse-root research/vanila_mse \
  --timse-root /home/t-lab01/sunrun-vanila-followups/runs/vanila_timse \
  --out /home/t-lab01/sunrun-vanila-followups/runs/vanila_emsemble \
  --observations /path/to/frozen/competition-observations.parquet --device cpu --threads 2
```

Source roots require `results/s{42,43,44}_predictions.parquet` and `logs/F{1,2,3}_s{seed}_metadata.json`. Inputs must have unique matching hourly origin/integer lead keys, positive finite forecasts, identical features, training ranges/cutoffs, 900 rounds, XGBoost parameters and frozen-offset protocol. Different offsets and objectives are expected. Metadata for smoke runs is rejected.

Outputs: flat `results/{s42,s43,s44,mean3}_predictions.parquet` and `_score.json`; `logs/run.jsonl`, `logs/provenance.json`, `logs/verification.json`; campaign-only `F*_metadata.json`, `train.log`, `complete.json`. The launcher owns `launch.json` and `train.exit`. Repository collection copies only allowed final artifacts. Publication is not performed by this runner; completion marks it as not required pending a separate candidate decision.

All source predictions are averaged before official observations are loaded. Both the two-model and three-seed means are fixed; no weights are fit. Official scorer replacement count must be zero. Seed mean/SD/worst and the score of mean3 forecasts are reported separately. The official scorer does not expose W or MAE; no improvement gate is asserted from unavailable metrics.

## 검증

External campaign tests cover equal-weight arithmetic, row ordering, mismatched keys, duplicate keys, nonfinite/invalid values, source protocol mismatch, smoke rejection, and all-six-source cohort assembly without observation or score files.

## Dispatch 2026-11-12

External campaign: `/home/t-lab01/sunrun-vanila-followups/`. Controller: `followup_scheduler.py`; assignment and scheduler-state record lifecycle. TIMSE seeds 42/43/44 use independent CPU tmux sessions (6 threads each). Fixed 50:50 ensemble automatically runs only after every TIMSE seed exits zero and passes artifact/replay checks. No leaderboard publication is scheduled. Results are archived here when complete. Local resource admission requires >9 GiB available memory per launch, max three jobs; golab2 unreachable, Colab has no active sessions. Official-objective reference audit and 14 passing synthetic tests are recorded in the external campaign.

## Execution result

Completed UTC: 2026-11-12T13:49:47.907628+00:00

| Scope | S | TIMSE | RMSE |
|---|---:|---:|---:|
| total | 97.602247 | 19651.129186 | 69.705242 |
| f1 | 119.238059 | 31027.781101 | 83.761778 |
| f2 | 75.308604 | 7991.263628 | 65.464030 |
| f3 | 61.893623 | 4890.219018 | 57.905131 |

Seed S mean=97.697705, SD=0.116692, worst=97.773574. Fixed mean3 is scored separately above.
Execution requested; publication later done on user request (see below), no baseline promotion. Failure/adoption diagnosis remains a separate review after all fixed experiments finish. Campaign: /home/t-lab01/sunrun-vanila-followups/.

## 리더보드 게시 (2026-11-13, 사용자 요청)

사용자 요청으로 `vanila_mse`·`vanila_timse`·`vanila_emsemble` 세 행을 보드에 함께 올렸다. 게시 자체는
성능 개선 판정이나 기준선 승격이 아니다(제출 게이트·SOTA 판정과 분리).

| 실험 (mean3) | S | TIMSE | RMSE | Bias | F1 S | F2 S | F3 S |
|---|---:|---:|---:|---:|---:|---:|---:|
| vanila_mse | 100.776 | 24932 | 65.88 | +4.55 | 122.11 | 81.02 | 63.37 |
| vanila_timse | 102.061 | 16638 | 83.01 | +42.53 | 123.00 | 78.84 | 69.62 |
| vanila_emsemble | 97.602 | 19651 | 69.71 | +23.54 | 119.24 | 75.31 | 61.89 |

Seed S (42/43/44): MSE 101.061/100.559/101.011, TIMSE 102.139/102.381/102.384,
emsemble 97.756/97.563/97.774. 앙상블은 세 seed와 세 기간 모두에서 두 단일 모델보다 낮다.

해석: 순수 TIMSE는 tail 오차를 줄이지만(TIMSE −33 %) 전체 예측을 위로 끌어올려(bias +42.5 km/s)
RMSE가 +26 % 나빠졌다. 고정 50:50 평균은 bias를 +23.5로 절반만 남기고 TIMSE 개선(−21 %)을 대부분 유지해
S = (MSE·TIMSE)^¼ 맞교환에서 이득(−3.15 % vs MSE)을 얻는다. 남은 +23.5 km/s bias가 다음 병목이다.
현재 보드 1위(S-cap 95.498) 대비 +2.2 %, `e2000fs_calib_offset_roll`(97.109) 대비 +0.5 %로 SOTA는 아니다.

dry-run 공식 scorer 점수는 저장된 mean3 점수와 1e-10 이내로 일치했다. 제출 meta·기록·라이브 행 검증은
캠페인 폴더 `~/sunrun-vanila-followups/{이름}-submission*.json`, `*-publication-verification.json`에 있다.

## 2026-11-13 레이아웃 변경 (중복 제거)

- 공용 코드를 `research/_module/`로 옮겼습니다. 이 행이 쓰는 모듈: `common`.
- `results/`와 `logs/`는 `research/result/vanila_emsemble/`로 옮겼습니다. 위 본문에 나오는 `results/…`, `logs/…` 경로는 이 위치 기준입니다. `weights/`와 `visualization/`은 이 폴더에 그대로 있습니다.
- `--out`을 주지 않으면 `research/result/vanila_emsemble/`에 씁니다. `--out DIR`을 주면 예전과 같이 `DIR/results`, `DIR/logs`, `DIR/weights`에 씁니다(재현 스크립트 호환).
- 입력 기본값이 `research/result/vanila_mse`, `research/result/vanila_timse`로 바뀌었습니다. 검증: 재실행한 s42·s43·s44·mean3 예측과 S가 기존과 같습니다.
