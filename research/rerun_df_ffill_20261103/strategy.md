# seongeun 전체 제출 모델: 새 df + causal ffill 재실험

## 실행 전 고정 계약

- 요청: seongeun의 제출된 96개 실험을 새 `data-v2/input/df.parquet`로 다시 실행하고 리더보드에 제출한다. 기존 숨김 모델도 대상에 포함한다.
- 사용자 최종 범위 수정: 원본 구현이 손상된 `e2000fs_blk0prev`만 재현 불가로 제외하고 **95개**를 재실행한다. HSSPEAK용 GONG 원본 추가도 사용자가 명시적으로 허용했다.
- 데이터: `MANIFEST.json`의 `built_utc=2026-11-02T12:09:07Z`, 2,816,784행, 446열. `old_df.parquet`를 사용하지 않는다. 원본 입력과 사용자의 작업 파일은 수정하지 않는다.
- 가설: 모델 구조를 유지한 상태에서 새 원문 기반 입력과 causal forward fill을 적용했을 때의 성능을 측정한다. 데이터/라벨/시간 정렬이 기존과 달라졌으므로 단일 특징의 개선 ablation 또는 SOTA 개선 증거로 해석하지 않는다.
- 기준선: `manifest.json`에 고정한 각 실험의 기존 제출 설정 및 지표. 실험별 피처 목록, 순서, 구조, seed, 학습 파라미터를 보존한다. 하나의 공통 Top30 또는 generic XGBoost로 대체하지 않는다.
- 허용 변경: 새 df 열 이름 연결(`ch_ch_` -> `suvi_`), 시간순 입력 특징 ffill, 새 df의 결측 학습 target에 대한 관측 마스크, F1/F2/F3의 기존 공식 학습 cutoff 준수.
- 입력 ffill은 `(origin,h)` 시간축에서 h별로 적용한다. train/query 중복 입력의 일치를 검사하고 같은 과거 입력 이력을 공유한다. period/fit mask/키/시간은 채우지 않는다. 선두 결측은 NaN으로 남긴다. bfill, 미래 입력, 전체 기간 평균 대체를 사용하지 않는다.
- 사용자 추가 확정: **학습 y도 ffill**한다. y는 물리적 `target_time` 순서로 원래 관측 라벨을 앞값 채움하고 train 행에만 적용한다. query y와 공식 채점 truth는 그대로 보존한다. 763,854개 학습 라벨 셀의 결측이 0개가 되었다. 입력만 채운 초기 8개 실행은 종료 코드 130으로 중단하여 별도 보관했고 제출하지 않았다.
- 최종 학습 입력은 `input/df_ffill_trainy.parquet`, 처리 보고서는 `input/ffill_trainy_report.json`이다. `input/df_ffill.parquet`는 입력만 채운 중간 산출물이다.
- 학습은 `fit_F1/F2/F3`의 origin+144h maturity masks를 적용한다. 관측 없는 target은 손실에서 제외하고 공식 평가 truth를 변경하지 않는다.
- 현재 CPU/RAM을 확인하여 실행 개수를 제한한다. 기존 다른 연구 작업/프로세스를 중단하지 않는다. 각 실험은 독립 tmux 세션, 로그, 출력 디렉터리, 종료 코드 및 메타데이터를 가진다.

## 근거 및 원본 검증

- `data-v2/README.md`, `data-v2/input/MANIFEST.json`: 새 raw-only 데이터 스키마와 가용 시각/관측 라벨 규칙.
- `analysis/A_oldvsnew/report.md`: 구형 표와 새 표의 시간 정렬, SUVI 계산, 학습 격자 및 결측 target 차이.
- `source_archive/`: 현재 연구 소스 또는 commit-pinned 과거 원본을 보관한 읽기 전용 사본. 출처는 `manifest.json`에 기록한다.
- 새 구조나 문헌상 성능 가설을 추가하는 실험이 아니므로 외부 문헌을 근거로 개선을 주장하지 않는다.
- 새 F1 평가기간과 겹치는 과거 검증/epoch 선택 구간은 각 fit 기간 내부의 과거 구간으로 이동하고 purge를 적용한다. 과거 평가 스트림으로 보정하던 모델은 동일 보정 공식을 학습기간 내 별도 사전 구간에 적합한 뒤 고정하는 검증정책 수정을 명시한다. 이 결과를 원래 누수된 점수의 정확한 재현 또는 독립적 구조 개선으로 주장하지 않는다.

## 검증과 완료 조건

1. 미래값 변조, horizon 간 오염, train/query 중복, metadata/target 불변 검사를 통과한 ffill 입력을 만든다.
2. 원본과 포팅 코드의 diff 및 모델별 실제 입력 feature list, 차원, 학습/예측 시간 범위를 기록한다.
3. smoke 검사는 파이프라인 검증만 한다. 본 실험은 원본 rounds/epochs/seed로 실제 학습을 완료해야 한다.
4. 모든 공식 F1/F2/F3 예측 키가 유일하고 유한한 양수이며 예측 대체율이 0인지 확인한다. 학습 모델 파일, 실행 종료 코드, 학습 로그, 예측 parquet, 평가 결과 및 메타데이터가 있어야 완료다.
5. `submit.py --dry-run` 후 실제 리더보드에 제출하고 원격 결과를 재조회한다. 기존 원격 visibility/강조 정책은 격리된 제출 클라이언트에 반영하여 보존한다.
6. 실패/입력 부재/구조 보존 불가 실험은 개별 원인과 필요한 복구를 기록한다. 실행 제출만으로 완료라 하지 않으며, 기존 예측 파일 재조립을 재학습으로 세지 않는다.
7. 모델별 새 데이터 설정은 사실대로 기록한다. 구형 입력으로 얻은 점수와 새 점수를 혼동하거나 새 입력을 구형 config라고 표시하지 않는다.

## 현재 발견한 의존성

- TDA는 df에 없는 원본 SUVI 이미지의 persistence-image 특징 392개를 사용한다. 사용자가 새 df와 원본 SUVI의 결합 사용을 명시적으로 허용했으므로 새 이미지에서 재계산한다.
- e1604_hsspeak25 및 xgb_d76_lead_moe_resid_dv_enlil은 하위 기반 모델/OOF 의존성을 새 입력으로 재구축해야 한다.
- HSSPEAK는 GONG 자기장 지도와 공개 시각에서 PFSS/WSA/HUXt 물리 입력을 재구성하고, 실제 원본 L72-COND roll6 기반 모델과 세 seed의 학습/추론 특징 차이를 보존한다. 기존 학습 모델 예측을 새 결과에 재사용하지 않는다.
- 일반 111-feature 전략은 df의 시간축에서 과거 24h speed와 원래 stale 표식을 재구성해야 한다.
- 모든 발견 사항은 모델별 포팅 과정에서 검증하며, 누락 특징을 조용히 삭제하지 않는다.

## 결과

아직 본 학습/제출 완료 없음. 이후 검증 증거와 채택 여부를 모델별 manifest/results에 추가한다.

## 실행 자원 변경 — 2026-11-03 사용자 지시
- 앞으로 이 재실험 캠페인의 모든 대기/미완료 학습은 현재 로컬 호스트 CPU에서 실행한다. golab2 신규 실행 및 GPU 신규 실행을 금지한다.
- 기존 golab2의 완료 결과는 그대로 회수·검증·제출하고, 미완료 모델은 원본 소스/입력 바이트와 완료 블록을 검증한 뒤 로컬로 이전한다. 경로와 파일 시간 정밀도 변경은 이전 기록에 명시한다.
- 로컬 메모리 여유를 확인하며 최대 4개 학습을 배정하고 물리 특징 재계산 자원을 함께 고려한다. 이전 CUDA 완료 결과는 실행 이력을 보존하며 재작성하지 않는다.

## golab2 폐기 — 사용자 운영 공지
- golab2의 SSH/rsync/명령 실행/상태 폴링을 전면 금지한다. controller 원격 폴링·수집 코드를 제거했고 launch.remote는 subprocess 호출 전에 예외를 발생시킨다.
- 추가 회수 파일이 필요하면 `/home/t-lab01/golab2-recovered-20261103/home-yehwan/sunrun-rerun-20261103/`의 로컬 파일만 읽는다. 미완료 실행은 현재 로컬 CPU에 한정한다.

## 완료 결과 표시 정리 — 사용자 최신 지시
- 전체 S/TIMSE/RMSE에서 다른 완료 모델보다 모두 불리한 결과와 동일 전체 평가 보고서의 중복 결과는 리더보드에서 숨긴다. 예측·모델·점수 저장은 유지한다.
- 기준 비교 모델 `e2000fs_calib_point4_144h`는 남기고 동일 결과의 dirmo pointweight4 및 rawdf 표기는 숨긴다. 지표별 장점이 있는 TCN·calib11g·shape 등은 유지한다.
- 이 선택은 표시 정리이며 다음 실험 특징/파라미터 선택이나 모델 채택 근거로 사용하지 않는다. 과거 전체 공개 요청보다 최신 숨김 요청을 우선한다. 자동 게시기는 최신 원격 visibility를 유지한다.

## dirrec 계열 제거 — 사용자 최신 지시
- 이름에 `dirrec`가 포함된 전체 8개(DirRec·DirRecMO·stratify·foundation 변형)를 사용자 제외 목록에 기록하고 실행 registry 및 리더보드 대기 항목에서 제거한다. 모두 미실행 상태였다.
- 제거된 실험의 원본 자료는 보존하며 다시 배정하지 않는다. launcher는 사용자 제외 목록을 확인하여 실행을 차단한다.

## 2026-11-03 23:00 KST golab2 예약 — 최신 사용자 지시
- 대기 큐 뒤쪽 27개를 `golab2-assignment-2300.json`에 고정하고 로컬 registry에서 제거하여 중복 실행을 방지한다.
- 이 27개에 한해 이전 golab2 폐기/원격 금지 및 로컬 전용 지시를 최신 사용자 요청으로 변경한다. 나머지 로컬 실험은 유지한다.
- 23:00 이전 학습 금지. GBM은 CUDA_VISIBLE_DEVICES="" CPU, Kumo는 GPU만 사용한다. 물리 특징 의존 실험은 검증된 특징 완료 후 실행한다.
- 예약 검증 완료: 2026-11-03 21:17 KST. golab2 작업 스케줄러의 23:00 트리거 및 WakeToRun/StartWhenAvailable 확인. 대기 27개, 학습 실행 0개, 원격 scheduler 정상.
- 입력 전송 완료: prepared 225,755,615 bytes / raw 213,699,522 bytes, 각 2,816,784행·446열 검증. 원격 소스와 로컬 소스 바이트 일치 및 9개 runner preflight 통과. Kumo GPU 모델 로드 통과. 증거: `golab2-schedule-verification.json`.
- 동시 최대 3개·각 CPU 4 threads·메모리 admission 적용. `e1604_hsspeak25`는 로컬 물리 특징 계산의 정상 종료 및 검증 후 `golab2_physics_bridge.py`가 특징을 전달하면 실행한다. 학습 종료 후 별도 평가·제출 검증이 필요하며 예약/학습만으로 완료 처리하지 않는다.

## 2026-11-04 전체 잔여 로컬 이전 — 최신 사용자 지시

- 요청: golab2 실패분을 회수하고 전체 남은 항목을 현재 로컬에서 실행한다. golab2의 해당 27개 예약 작업과 스케줄러를 중지하고, 완료 18개는 재학습하지 않고 산출물을 회수하여 공식 평가·게시한다. 실패 1개·중단 3개·미실행 5개는 로컬 큐에 편입한다.
- 원격 원본 실행 기록은 `recovered_golab2_20261104/`에 보존한다. 회수한 prepared/raw 입력과 현재 입력의 전체 바이트를 직접 비교하며 해시는 계산하지 않는다. 완료 산출물의 복사본에서 경로와 입력 시각 메타데이터만 현지 경로로 연결하고, 모델·예측·실행 소스 바이트는 유지한다. 원래 학습 호스트는 golab2로 기록한다.
- Kumo 실패 원인은 golab2 Python 개발 헤더 `Python.h` 부재에 따른 Triton 빌드 실패이다. 로컬 Python 헤더와 기존 Kumo 체크포인트를 확인했고 모델 구조·학습 설정은 변경하지 않는다.
- 로컬 이전 실행의 중단 시각에 커널 메모리 부족 로그가 확인됐다. 미완료 4개 원본 산출물을 별도 보관하고 최대 동시 2개, 전략 계열 12GiB/MIMO 20GiB 예약으로 재시도한다. 새 메모리 예약은 모델 설정 변경이 아닌 실행 자원 조정이다.
- 이전 CPU 전용 실행 지시를 현재 사용자 장치 규칙으로 정정한다. GBM은 CPU 및 `CUDA_VISIBLE_DEVICES=""`, 신경망은 GPU0 및 `CUDA_VISIBLE_DEVICES=0`만 사용한다. GPU가 바쁘면 대기하며 CPU로 우회하지 않는다.
- 기존 사용자 제외 14개는 유지한다. 기존 manifest에 있지만 큐에서 누락된 `realmlp_direct`, `tabiclv2_direct`도 지원되는 원본 foundation runner로 포함해 총 82개 비제외 모델을 대조한다. 기존 제출 28개와 회수 완료 18개를 제외한 36개가 학습 대상이다. 완료는 실제 종료·모델/예측 검증·공식 평가·제출 영수증 확인 후에만 기록한다.
- 이전 검증 완료: prepared 입력 225,755,615바이트 및 raw 입력 213,699,522바이트의 전체 내용이 로컬과 동일하다. 완료 18개 전부 `publish.validate_run`을 통과했다. 근거는 `local-adoption-20261104.json`, `adopted-validation-20261104.json`, `retry-source-comparison-20261104.json`에 기록했다.
- 10:15 KST 실행 확인: `sunrun-local-20261104` 컨트롤러와 독립 학습 세션 2개가 가동 중이다. `kumo_direct`는 GPU0에서 F1의 여러 head를 실제 완료하여 기존 Python.h 실패가 재현되지 않았고, `extreme_histprior`는 CUDA가 숨겨진 CPU에서 실행 중이다. 회수분 `agnostic_v2_context`의 첫 공식 제출 영수증이 생성돼 전체 제출 29개, 회수 후 평가/제출 대기 17개, 학습 중 2개, 나머지 학습 대기 34개이다. 이 수치는 전체 캠페인 완료 선언이 아닌 해당 시각의 상태다. `local-execution-health-20261104.json` 참조.

## 2026-11-04 대기 17개 golab2 재배정 — 최신 사용자 지시

- 요청: 학습 대기 34개 중 17개를 golab2의 별도 tmux에서 최대 3개씩 병렬 실행한다. 현재 로컬 학습 2개는 유지한다. 이는 이전 전체 로컬 지시의 해당 17개 배정만 변경한다.
- `golab2-assignment-20261104-17.json`에 미실행 CPU 모델 17개를 고정했다. 신경망 direct 2개는 로컬 GPU 큐에 유지한다. 원래 모델·특징·seed·rounds 및 평가 프로토콜은 변경하지 않는다.
- 새 원격 디렉터리는 `/home/yehwan/sunrun-golab2-20261104-17`, 감독 tmux는 `sunrun-golab2-17-20261104`이며 각 실험은 별도 `sr23-<experiment>` 세션, 로그, 종료 코드를 가진다. 이전 원격 큐와 예약은 계속 비활성 상태로 보존한다.
- 원격 여유 RAM 약 30GiB와 기존 작업 없음 확인 후 배정했다. CPU 실험은 `CUDA_VISIBLE_DEVICES=""`, 각 4 threads, 최대 동시 3개 및 작업당 8GiB admission을 적용한다. 검증된 prepared/raw 입력 및 물리 특징을 기존 원격 원본에서 같은 바이트로 연결하고 입력 크기·mtime과 원래 배포 기록 일치를 확인했다. runner compile 및 17개 예약 사전 검사를 통과했다.
- 중복 실행 방지: 해당 17개는 로컬 registry에서 제거하고, 로컬 run 경로에 원격 배정 marker를 둔다. 완료 후 수집기는 이 marker를 검증하고 원본 산출물을 보존한 뒤 완전한 실행 디렉터리를 원자적으로 설치해 공식 평가·제출한다. 원격 학습 완료만으로 제출 완료를 선언하지 않는다.
- 실행 검증: 10:25:52 KST에 golab2의 `tailcal_crps`, `transfer_center`, `transfer_recency` 3개가 시작됐다. 실제 Python PID 3개, 각각 CPU4threads 및 빈 CUDA_VISIBLE_DEVICES를 확인했고 transfer 두 모델은 F1 horizon19 학습 완료 로그를 확인했다. 나머지 원격 14개는 대기, 로컬 2개 실행·17개 학습 대기를 유지한다. `golab2-execution-health-20261104-17.json`, `split-queue-verification-20261104.json`에 증거를 기록했다.
- 자동 회수: `collect_remote17.py` 및 테스트 10개를 검증하고 실제 원격 상태/입력 preflight를 통과했다. `sunrun-remote17-collector-20261104`에서 60초 간격으로 원격 완료/실패 산출물을 회수하고, 검증된 성공분은 기존 게시 lock 아래 공식 평가·제출한다. 원본과 오류 로그는 `recovered_golab2_20261104_17/`에 보존한다. 게시 중에는 조회 간격이 길어질 수 있으나 원격 감독 스케줄러는 독립적으로 다음 실험을 실행한다.

## 2026-11-04 10:42 KST 로컬 1개 추가 실행

- 사용자 요청으로 대기 중인 `e2000fs_budget_spectral`을 로컬 CPU2threads 및 빈 CUDA_VISIBLE_DEVICES로 추가 실행했다. 메모리 여유 약20.7GiB를 확인했고 기존 로컬2개와 golab2 배정을 유지했다. 독립 tmux `srff-e2000fs_budget_spectral`, 원본 소스 snapshot 일치와 실제 Python 프로세스를 검증했다. 기존 메모리 admission 및 감독 컨트롤러가 이 실행의 완료·제출을 추적한다.

- 2026-11-04T01:44:31.396836+00:00 사용자 추가 요청: `e2000fs_feat30_gaussian_pilot`을 CPU2threads/빈 CUDA_VISIBLE_DEVICES로 별도 tmux에 추가 실행. 시작 전 메모리 여유 약19.9GiB 확인. 모델 구조·특징·seed·rounds는 그대로 유지.

- 2026-11-04T01:46:37.812281+00:00 사용자 추가 요청: `e2000fs_feat30_hyper_gain`을 CPU2threads/빈 CUDA_VISIBLE_DEVICES로 별도 tmux에 추가 실행. 여유 RAM 19.2GiB와 캠페인 예약 합 13.6GiB를 확인. 원래 모델·특징·seed·rounds 유지.

- 2026-11-04T01:49:06.387662+00:00 사용자 요청으로 `e2000fs_feat30_lognormal_crps`을 CPU2threads/빈 CUDA_VISIBLE_DEVICES로 추가 실행. 시작 전 실제 여유 RAM 및 캠페인 예약 메모리 검사 통과. 기존 모델·특징·seed·rounds 유지.

## 2026-11-04 사용자 요청: 무거운 대기항목10·11 제외

- `e2000fs_calib5_detpool_all`, `e1604_hsspeak25`는 원격에서 미실행 대기 상태임을 확인한 후 사용자 제외로 등록했다. 로컬 manifest와 배정 기록, 원격 스케줄러 및 리더보드 대기 항목에 반영했다. 기존 코드·물리 특징·과거 제출 및 오류 기록은 보존한다.
- 원격 감독만 안전하게 재시작했고 실제 학습3개는 유지했다. 검증 결과 학습중3/완료3/대기9/사용자제외2이며 두 제외 항목의 실행 디렉터리는 없다. 제외 상태를 스케줄러·수집기의 정상 종결 상태로 처리하는 회귀 테스트13개를 통과했다.
- 수집기는 현재 진행 중인 게시를 끝낸 후 `sunrun-collector-reload-heavy` 감독이 새 코드를 재적용하도록 구성했다. 현재 게시 프로세스를 중단하지 않고 제외 모델의 자동 재실행·제출을 방지한다.
