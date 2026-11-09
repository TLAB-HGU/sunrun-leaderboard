# Current first-place model plus three historical-speed lags

User-requested joint addition: retain all28source features of
`e2000fs_calib_offset_roll_drop_t035_v650_s42`, add `v_lag24`, `v_lag48`,
`v_lag72`, and fit fresh seeds42/43/44. This is31source/215expanded inputs.
The pinned parent audit is `parent-lock.json`; `strategy.md` is the frozen
protocol. The source order and all scientific settings remain fixed.

All three runs train on golab2. A local tmux service keeps the actual remote
scheduler foregrounded on SSH/WSL; a separate collector brings back all fitted
models, predictions and validation evidence. Existing host workloads remain.
The standard `sunrun` monitor discovers the remote assignment and run folders.

The publication controller waits for `training-complete.json`, checks the full
three-seed report with paired2,000-draw uncertainty analysis, then publishes all
three individual results as explicitly requested, regardless of exploratory
performance-gate outcome. It verifies official stored scores, visible Space
rows and pinned architecture links. There is no ensemble or best-seed selection.

Current status is in `collector-state.json`, `runtime.json` and
`publication-controller-state.json`. Training completion, experiment registration
and publication completion are separate. Only `campaign-complete.json` with
`complete:true` and three verified visible result rows establishes completion.

Local services are `remote_scheduler_controller.py`, `collector.py`, and
`publication_controller.py`; their exact tmux identities/logs are recorded in
matching `.launch.json`/`.log` files. Preserve failed attempts before recovery.
`launch_services.py --resume` resumes the first two services without duplicating
live exact-owned sessions. The publication controller is separately protected
by its lock and can use the same `launch_services.start` helper for recovery.

Full results will be written to `analysis/report.md` and `publication/receipts.json`.
The repeated official-period, seed42-driven selection is explicitly exploratory;
fresh fitting does not create independent generalization evidence.
