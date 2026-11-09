# Four-axis successor

User-authorized axes: ACE minute support, minute speed–density coupling,
past-only recurrence reliability/phase, and fixed t045 CH area24/48h changes.
Twelve fresh CPU fits, seeds42/43/44, against frozen D; independent additive
unions, report only. Exact hypotheses and gates are in strategy.md.

The owned prepare_controller.py service first prepares label-free caches under
shared CPU admission, then waits for the preceding eight-axis campaign's ready
axes1–7 to finish validation and analysis. It does not wait for source-blocked
CME axis8. No current campaign is interrupted and no partial score chooses the
parent or additional features. Local registry is empty to avoid reporting remote
fits as local; the active preparation service is visible in sunrun. Following
staging, all12 queued/running jobs appear under golab2's scheduler layout.

Remote destination: /home/yehwan/sunrun-followup-feature-axes4-20261109.
Controller: sraxes4-prepare-controller-20261109; actual remote foreground
scheduler transport: sraxes4-remote-controller-20261109. Runtime and controller
JSONs distinguish waiting, preparing, training, blocked, and complete. Require
campaign-complete.json complete:true plus the full report before declaring
experimental completion. No automatic leaderboard publication.

Feature additions per24h head:3/2/6/2, giving215/214/218/214 expanded columns.
All original212inputs are preserved. Source-smoke.json records actual-data
feature availability and unchanged shared runner bytes; it is not fit evidence.
Each remote augmented input is reread and original columns compared exactly.

Recovery never overwrites partial remote staging or incomplete collected runs.
Read exact launch receipts and failed logs before safe matching resume. All
fitted models, predictions, scoring and replay receipts are collected locally.

## Execution host change (2026-11-09)

The golab2 staging step failed (ssh exit 1, no remote error captured), and the
user set a 16-hour deadline for all queued work. The 12 fits therefore run on
**local**. Recipes and features are unchanged; only `execution_host` and `data`
paths differ.

- Augmented inputs were built locally with `build_data.augment`. The
  original-column equality proofs are in `input/axis*.validation.json`.
- `assignment.json`, `preflight.json` (key `local`), `scheduler-state.json` and
  `registry.json` were written for local execution.
- tmux `sraxes4-host-scheduler-20261109` runs `host_scheduler.py`.
- tmux `sraxes4-analysis-controller-20261109` runs
  `local_analysis_controller.py`, which writes the report and
  `campaign-complete.json`.
- The partial golab2 staging directory
  `~/sunrun-followup-feature-axes4-20261109` is preserved untouched for diagnosis.
