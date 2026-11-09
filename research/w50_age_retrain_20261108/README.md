# Active fresh-training campaign

Goal: official S <=48.746221040299844 against frozen best single
97.49244208059969; independent SOTA comparison. This campaign is ongoing,
not a completed50% improvement claim. See strategy.md for the preregistration.

Actual new fits: seeds42/43/44; no stored booster/prediction reuse. Only
img_age_h values change. Parent code/features/objective/cutoffs stay fixed.
The raw_age.parquet file is a lossless projection of raw origin,h,img_age_h
where h=1; the causal reconstruction deduplicates origins as documented.

golab2 has independent tmux sessions srw50-age_retrain_s{seed}-20261108 and
srw50-host-scheduler-20261108 (initial detached attempt; current scheduler is
foregrounded by the local controller below). Three simultaneous four-thread
CPU fits are permitted by the resource-only preregistration amendment before
seed44 admission; the experiment/seed count remains three. Resource checks
precede each admission. Host assignment/scheduler-state drive
sunrun discovery; the local registry is intentionally empty because local
copies are remote artifacts, not local training jobs. remote-registry.json
retains the recipes. collect_controller.py mirrors logs/progress and, once
all jobs terminate, trained boosters. It never chooses another hypothesis.

Local collector: srw50-collect-controller-20261108. Check sunrun --once and
runtime.json. A submitted command does not establish completion: require
train.exit0, complete.json, full period models/predictions/metadata and official
score parity. Failed preflight: remote environment lacks pytest; no full fit
ran in that attempt. Local8tests passed, and remote input/config preflight
passed without installing packages. Real-data two-round smoke is separately
marked smoke and cannot be scored/adopted.

After all three fits terminate and artifacts synchronize:

```sh
.venv/bin/python research/w50_age_retrain_20261108/analyze_results.py
.venv/bin/python research/w50_age_retrain_20261108/evaluate_goal.py
```

Independent full-model replay, causality and identity audit must create a
valid verification.json before the goal evaluator can PASS. Exit2=PARTIAL,
exit1=completed-score failure, exit0=verified goal. Do not create artificial
audit passes. Hash identities remain an explicit earlier automatic-approval
gap; preserve source commit, store revisions and byte/time identities.

Codex active goal and .omx/goals/performance/solarwind-s50/ retain the ongoing
research objective. A null experiment triggers cause diagnosis before another
preregistered change. No automatic score-driven parameter search, promotion,
or external publication. Unseen future validation remains separately proposed.

Execution recovery: first detached remote attempts vanished without exit receipts.
Their partial outputs live in failed_attempts/remote_session_01 and are not
reused. Clean fits restarted under the real scheduler's foreground SSH connection,
owned by local tmux srw50-remote-scheduler-controller-20261108. Inspect its log
and exit receipt plus the experiment train.exit files; tmux creation is insufficient.
