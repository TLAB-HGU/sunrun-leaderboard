# 직교 축 손실 ablation 2차 (라벨 품질 · lead 균형 · 최근성 · Balanced MSE) → `e2000fs_calib_offset_roll`

사전등록 문서는 `strategy.md`다(가설, arm 정의, 문헌 확인, 선별 규칙, 게이트). 학습은 모두 로컬에서 한다.

## 구성
- `core.py`: 보드 스크립트의 러너를 복사하고 objective만 주입했다. objective에 관측 라벨 마스크와 origin 시각을 넘긴다. `test_core.py`는 AST가 같은지 검사한다.
- `objectives.py`: arm A(보드 손실 그대로)와 4개 arm(obsmask, leadbal, recency, bmc). `test_objectives.py`는 A의 bit 동일성, 유한차분 gradient, 곡률을 검사한다.
- `prepare.py`: 테스트를 돌리고, 대조군을 다시 채점하고, 5개 smoke와 A parity를 확인한 뒤 `assignment.json`, `registry.json`, `manifest.json`, `scheduler-state.json`, `preflight.json`을 쓴다.
- `local_scheduler.py` + `scheduler_support.py`(로컬 tmux `srlossax2-local-scheduler-20261110`): 최대 5 job, 4 thread.
  - seed42 선별 후 arm별로 seed43/44를 release(`waiting`)하거나 `not_selected`로 둔다.
  - 모든 run이 끝나면 `analyze_results.py`를 실행한다. 결과는 `analysis/report.{md,json}`와 `campaign-complete.json`이다.
- 대조군은 `research/e2000fs_calib_offset_roll/results/s{42,43,44}`(보드 seed)다.

## 상태 확인
- `sunrun --once`
- `scheduler-state.json`, `selection.json`, `scheduler.log`
- 실패한 run은 `scheduler-state.json`에 `failed`와 오류가 남고, `runs/{이름}/train.log`가 보존된다.

## 게시
게이트를 통과한 arm만 게시한다(`campaign-complete.json`의 `passed`). 게시 작업은 자동화하지 않았다.
