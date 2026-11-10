# Current-model direct-S ablation

Approved replacement study: fresh A/B/C × seeds42/43/44, no publication.
The complete scientific preregistration is in strategy.md. Actual objective
identity is recorded separately from legacy pipeline configuration fields.

`prepare_campaign.py` runs the synthetic tests, freezes source bytes and input
identity, then registers three nonscoring smokes and nine full runs. It refuses
to overwrite an existing preflight. `campaign_controller.py` preserves other
workloads, admits through the shared CPU lock, requires all smokes before full
training, verifies fresh forecasts/model replay, and starts the postfit analyzer
only after all nine runs validate. Do not launch experiment scripts manually
around these gates. The baseline smoke also executes the frozen original runner
and requires exact prediction/offset parity.

The controller may wait while the host is full. That is not experiment
completion. Live state is in runtime.json/scheduler-state.json and the analysis
status.json; per-run train.log/train.exit show actual execution. Source changes,
failed fits or lost sessions fail closed, preserving evidence; no automatic retry
with changed settings. Resume the controller with the same frozen sources only
after diagnosing any failure. No leaderboard write is implemented.

Initial preparation and controller command, from repository root:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python research/current_offset_roll_direct_s_20261109/prepare_campaign.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -u research/current_offset_roll_direct_s_20261109/campaign_controller.py
```

Use the existing durable controller session rather than starting a second copy.
Source snapshot equality (including imported evaluation helpers) and a controller
ownership lock prevent silent source drift and duplicate orchestration.

After verified completion, report.md/json appear in
analysis/current_offset_roll_direct_s_20261109/. Statistical implementation uses
2,000 paired draws, recomputed quantiles and Holm adjustment for two primary
comparisons. Repeated official-period exposure limits generalization claims.
