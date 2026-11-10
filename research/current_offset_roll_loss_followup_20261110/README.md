# 2026 후속 손실 논문 8편 → `e2000fs_calib_offset_roll` 손실 ablation

사전등록 문서는 `strategy.md`다(가설, arm 정의, 문헌 확인, 선별 규칙, 게이트).

## 구성
- `core.py`: 보드 스크립트의 러너를 복사하고 objective만 주입했다. `test_core.py`는 AST가 같은지 검사한다.
- `objectives.py`: arm A(보드 손실 그대로)와 8개 arm. `test_objectives.py`는 A의 bit 동일성, 유한차분 gradient, 곡률 양수를 검사한다.
- `prepare.py`: 테스트를 돌리고, 대조군을 다시 채점하고, golab2 원격 root `/home/yehwan/sunrun-loss-followup-20261110`를 만들고, 9개 smoke와 A parity를 확인한다.
  - 결과 파일은 `preflight.json`, `combo-smoke.json`, `smoke_parity/`다.
- `host_scheduler.py` + `scheduler_support.py`(golab2에서 실행): 6 slot, 4 thread.
  - seed42 선별 후 arm별로 seed43/44를 release(`waiting`)하거나 `not_selected`로 둔다.
- `remote_scheduler_controller.py`(로컬 tmux `srlossfu-remote-scheduler-controller-20261110`): golab2 scheduler를 foreground ssh로 유지하고, 끊기면 다시 연결한다.
- `collector.py`(로컬 tmux `srlossfu-collector-20261110`): 검증된 run을 weights까지 `runs/`로 가져온다.
  - 모든 run이 끝나면 `analyze_results.py`를 실행한다. 결과는 `analysis/report.{md,json}`와 `campaign-complete.json`이다.
- 대조군은 `research/e2000fs_calib_offset_roll/results/s{42,43,44}`(보드 seed)다.

## 상태 확인
- `sunrun --once`(golab2 실행 중/대기열)
- `collector-state.json`, `remote-state.json`, `selection.json`
- `training-blocker.json`: 실패가 있을 때만 생긴다.

## 게시
게이트를 통과한 arm만 게시한다(`campaign-complete.json`의 `passed`). 통과 arm이 생기면 다음을 한다.
- seed 3행과 `e2000fs_calib_offset_roll_loss_{arm}_mean3`를 게시한다.
- seed 행은 push_policy로 숨긴다.
- 보드 폴더 `research/{name}_mean3/`를 만든다.

이 게시 작업은 아직 자동화하지 않았다. 통과 arm이 생기면 따로 실행한다.
