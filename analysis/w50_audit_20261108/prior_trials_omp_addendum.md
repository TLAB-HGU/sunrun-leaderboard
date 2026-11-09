# Prior-trial OpenMP addendum

**The output-strategy OMP1 issue is not established in the age-retrain, round-selection or time-weighting trials. All nine preserved actual training scripts explicitly export OMP_NUM_THREADS=4, matching frozen baseline D.** The experiment and parent-core defaults use `setdefault(..., '1')`, which preserves an exported4. This addendum does not withdraw prior numerical results or reclassify the three trials as thread-confounded.

## Primary evidence

| Campaign | Seeds | Actual training-script export | Requested CLI | Evidence grade |
| --- | --- | --- | --- | --- |
| w50_age_retrain_20261108 | 42,43,44 | OMP4, OPENBLAS1, MKL1, CUDA empty | CPU, threads4 | B |
| w50_roundselect | 42,43,44 | OMP4, OPENBLAS1, MKL1, CUDA empty | CPU, threads4 | B |
| w50_timeweight | 42,43,44 | OMP4, OPENBLAS1, MKL1, CUDA empty | CPU, threads4 | B |
| Frozen D | 42,43,44 | OMP4, OPENBLAS1, MKL1, CUDA empty | CPU, threads4 | B |

Grade A would require a preserved, identity-bound training-process environment or in-process OpenMP ICV capture. Grade B is a preserved actual launch script, matching launch command, and consistent source snapshots. Grade C is requested-thread metadata or a different process's environment and cannot prove effective fitting threads. This audit has grade B training evidence; it does not promote requested nthread4 or a process's total thread count to grade A.

All treatment exports are at line4 and actual model commands at line5:

- `research/w50_age_retrain_20261108/runs/age_retrain_s42/train.sh:4`, `age_retrain_s43/train.sh:4`, `age_retrain_s44/train.sh:4`.
- `research/w50_roundselect/runs/roundselect_s42/train.sh:4`, `roundselect_s43/train.sh:4`, `roundselect_s44/train.sh:4`.
- `research/w50_timeweight/runs/timeweight180_s42/train.sh:4`, `timeweight180_s43/train.sh:4`, `timeweight180_s44/train.sh:4`.

Each command before shell redirection was compared with its preserved launch.json command; all nine match exactly. Each provenance.json records XGBoost2.1.4. The sidecar JSON preserves current full script contents, paths, sizes and mtime_ns. These are current-content evidence, not proof against all historical modification.

Baseline D s42 is `research/drop035_v650_confirm_20261108/runs/drop035_v650_s42/run.sh:4`. Original golab2 s43/s44 are `/home/yehwan/sunrun-drop035-v650-confirm-20261108/runs/drop035_v650_s43/run.sh:4` and the analogous s44 path. Their contents were captured by the preceding read-only baseline check in `research/w50_output_strategy/runtime-omp-correction-review.log` and are reproduced in this addendum's JSON; no redundant remote poll was run.

## Why other OMP1 strings are not evidence of OMP1 training

All three campaign experiment.py files at lines14–15 default OMP to1 only when absent. Their preserved executed/prelaunch snapshots do the same. Each inspected preserved parent core uses setdefault at lines11–12. The explicit actual-shell export4 therefore survives these defaults.

- Age scheduler `host_scheduler.py:158` and `admit_third_controller.py:58` emit OMP4. The OMP1 export in `analysis_controller.sh:3` belongs to post-fit analysis.
- Round-selection scheduler `host_scheduler.py:174` emits OMP4. Its remote scheduler wrapper at `remote_scheduler_controller.py:14` starts the controller under OMP1, but the per-run shell explicitly replaces that value with4. OMP1 in `analysis_controller.sh:4`, `diagnosis_controller.sh:4` and `reconcile_controller.py:44` belongs to other processes.
- Time-weight scheduler `host_scheduler.py:205`, including the executed-workspace snapshot, emits OMP4. OMP1 in controller/collection/analysis/diagnosis/replay services is not the fitting child's final launch environment. `verify_prior_head.py:7` explicitly sets1 for that verifier process; it is not the actual train.sh command.

No additional `omp_set*`, `threadpool_limits`, OMP_THREAD_LIMIT, KMP or GOMP override was found in the inspected fitting source/snapshot paths. This does not reconstruct unrecorded inherited environment, affinity or cgroup settings. The preserved time-weight `live-resource-verification.json` lists17 process threads per job but no process OpenMP environment or per-thread CPU deltas; it cannot independently establish effective parallelism.

Requested threads are passed to DMatrix/training: age preserved parent `core.py:120`–124; round-selection preserved `experiment.py:115`–117; time-weight preserved `experiment.py:187`–190. Serialized nthread is a request, not an effective runtime measurement. The already-recorded installed XGBoost2.1.4 binary proof in `runtime-parallelism-check.log` establishes its minimum(requested, OpenMP maximum, processor count, thread limit) behavior. It does not create a mismatch when both actual launchers explicitly export4.

## Interpretation and retained decisions

Existing numeric scores remain measurements of their saved prediction artifacts. No scoring arithmetic was rerun, and this addendum does not newly certify the prior validation. The new OMP finding alone does not invalidate those numbers. Existing results.json files for all three campaigns still record `status=complete`, `candidate_gate_passed=false`, and `adopted=false`; those decisions remain unchanged.

A demonstrated fitting-thread mismatch would weaken a claim that a difference was caused solely by the intended experimental component, because numerical reductions and resource conditions would be additional changes. That mismatch is not demonstrated here. Consequently, this audit does not weaken these three prior comparisons on an OMP1-versus4 basis and does not supersede their earlier causal conclusions on that basis. It does limit any stronger claim that requested nthread4 alone proves identical measured effective runtime behavior. No magnitude, sign, score effect or failure explanation attributable to thread count is inferred.

No sealed355/time-weight report,98/probe report or earlier audit was edited. No reports are superseded by a newly proven mismatch, because none was found. The current output-strategy OMP1 cohort remains a separate implementation issue and cannot be generalized to these prior actual launchers.

## Scope and validation

The audit checked nine script environments, nine exact launch-command matches, nine recorded XGBoost versions, three baseline launchers and21 relevant Python source/snapshot files. Relevant source files parse successfully; no experiment module was imported or executed. No new runtime polls, model fits, predictions, scoring, target reads, remote mutations or new trial IDs were performed. Writes are limited to this new Markdown addendum and its JSON/log sidecars.

No automatic retests are justified by this audit. New trials remain gated on diagnosis of the corrected current output-strategy run. Original reports remain historical records.
