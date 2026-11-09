# Complete offset-roll ablation campaign

165 registered evidence conditions: the historical 162-condition inventory plus three Hessian-only controls. Original baseline A only; B1 is excluded. Thirteen validated historical conditions are imported. New work consists of 113 fits (57 local/golab, 56 golab2) and 39 replays. Final pruned confirmation adds three fits after the removal manifest freezes.

`assignment.global.json` is the immutable evidence inventory; `allocation.json` assigns each new experiment to exactly one host. Host `assignment.json` files contain dependency-closed execution subsets. Local `registry.json` and remote scheduler layout support `sunrun` discovery. Both hosts cap admission at five jobs, 20 threads and 6 GiB available memory reserve. GBM processes hide CUDA.

The central `controller.py` supervises host schedulers, collects remote validated results, runs global analysis, freezes one removal manifest, then runs final three-seed confirmation without feedback tuning. Host-specific smoke and full F1/F2/F3 saved-model inference/scoring gates precede new launches. `verify_imports.py` checks original training metadata and full binary equality of the original inputs before admission.

## Operations

Use the repository `.venv/bin/python`, always with `CUDA_VISIBLE_DEVICES=''`. `prepare_campaign.py` is a one-time isolated preparation command and refuses an already-prepared campaign. Existing campaign outputs must not be overwritten.

- Inspect `controller-state.json`, `scheduler-state.json`, `remote-state.json`, and the controller/scheduler logs.
- Run `controller.py --root <campaign> --once` only when the persistent controller is stopped; `.controller.lock` prevents duplicates.
- Run `analyze_results.py --root <campaign> --partial --bootstrap 0` for provisional diagnostics. It does not freeze removal decisions.
- The controller runs full bootstrap analysis automatically after every registered condition is validated, then `--final` after the fixed final configuration completes.
- `sunrun --once` shows running jobs and completed scores. Local Tlab01 is golab; the remote WSL host is golab2.

Normal completion requires `campaign-complete.json`, final analysis, and all exit/artifact validations. A live tmux session is not completion. Failed runs retain logs and are not automatically retried under changed settings. No external publication is performed.

## Verification

`CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q research/offset_roll_complete_ablation_20261106`

See `strategy.md` for hypotheses, fixed decision rules, source references, and uncertainty limits. Scientific conclusions remain pending until the full audit completes; final pruning can worsen performance under the requested evidence-only retention rule.
