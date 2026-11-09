# Completed tree-count ablation; research goal remains active

All3fresh fits exited0;54boosters and official forecasts reproduced independently.
MeanS97.813555→99.400507; pairedgain−1.622213%,95CI[−3.019933%,+1.537568%].
Notadopted,notpublished;50%target48.746221 remains unmet. See report.md for the
ordered outcome/diagnosis/uncertainty, results.json and independentaudit.
Original prelaunch/runtime description below is historical.

# Active training-only tree-count experiment

The research goal remains officialS<=48.746221040299844, against fixed best
single97.49244208059969. No achievement is claimed. The completed age-repair
experiment failed and was not adopted; see ../w50_age_retrain_20261108/report.md.

This preregistered next component keeps all28source features/212expanded
columns and their original values, loss, cutoffs, training rows and offset
policy. Every prior is freshly fitted for900rounds. A training-only stopping
subset selects the first minimum native RMSE; the selected prefix feeds the
unchanged offset fit, and a full model is freshly fitted for that count.
There is no patience parameter, fixed300choice or evaluation-target selection.
See strategy.md and protocol-review.md for hypotheses, limits and gates.

Actual scientific fits started2026-11-08 15:08:52UTC (2026-11-09 00:08:52KST).
Seeds42/43/44 run on golab2, fourCPUthreads each,CUDA invisible. The three other
lag-feature fits observed on the host were preserved; admission respects the
total6job/24thread and memory limits. Live status is in runtime.json and
live-process-verification.json; preflight.json is a historical prelaunch snapshot.

Local owned services:

- srw50-roundselect-remote-scheduler-controller-20261108: foreground actual
  remote scheduler through WindowsOpenSSH/WSL, with keepalives and CPU-only env.
- srw50-roundselect-collect-controller-20261108: artifact mirror and terminal
  vs success distinction. It does not choose scientific hypotheses or scores.

Remote experiment sessions: srw50-roundselect_s{42,43,44}-20261108. Remote root
is /home/yehwan/w50-roundselect-workspace-20261108/research/w50_roundselect_20261108,
also discoverable through ~/sunrun-w50-roundselect-20261108. Remote assignment/
scheduler-state drives sunrun; local registry is empty to avoid counting mirrored
outputs as local training. Check sunrun-running.txt. Completion requires actual
exit0, all predictions/models/metadata and official parity, not session creation.

frozen_controls/ contains all9 original parent prior-prediction arrays. The
newly fitted900endpoint must reproduce them<=1e-6 before any prefix selection.
Mismatch halts the experiment as a control/protocol failure. The stored prior
always has900trees; the full model's tree count is selected_rounds. Any replay
must explicitly apply the recorded prefix to the prior rather than using its
default900-tree prediction.

Preflight: local11tests, actual-host synthetic212input/24output spectral/native
RMSE test with exact endpoint invariance and saved-prefix replay. That small
test is compatibility evidence, not model skill. Source diff and feature lock
were recorded before the first fit. Controls/provenance/sources were staged
before admission. No packages, original data/scorer or other campaigns changed.

The existing performance-goal evaluator command follows the explicit
../w50_age_retrain_20261108/goal-handoff.json to this study's evaluate_goal.py;
the fixed objective is unchanged. Invoke this evaluator directly after completion:

```sh
.venv/bin/python research/w50_roundselect_20261108/evaluate_goal.py
```

It returns2for incomplete evidence,1for complete official scores failing50%,
and0only for all3single-model scores plus verified CI/integrity gates. No hidden
goal state is modified. Historical age-study evaluation can still use that
script's explicit --root to bypass the ongoing-campaign handoff. Paired analysis
waits for all three fits and uses the frozen168h/2000draw protocol.

No automatic publication. Promotion additionally requires every seed<=96.517518
(fixed best minus1%), all matched-seed/CI/auxiliary gates and documented source
integrity. Hash, historical publication/receipt, calibration and repeated-
leaderboard-selection limitations remain explicit. Pre-cutoff complexity
selection cannot create missing eruption information or guarantee50% improvement.
