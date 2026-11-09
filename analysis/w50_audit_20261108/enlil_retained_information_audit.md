# Retained ENLIL information: source audit after round-selection failure

Read-only findings from native Explorer `/root/state_failure_history`.
No target columns, current scientific scores, fitting or prediction corrections
were used for this lookup. This document records the code facts, not an
incremental skill claim or a selected next model.

## What the current model actually receives

`research/drop035_v650_confirm_20261108/baseline.py:15` retains only
`enlil_n_pending` and `enlil_kp` from ENLIL. `selected_names()` intersects the
larger name schema with that selected inventory. `load_grid():25` reads these
constant values at h=1; `core.py:80` repeats the same origin-level scalars in
every24-lead input block. Current round-selection inputs are independently
verified identical to that parent. No numerical arrival time or lead-specific
arrival feature enters the current28-feature matrix.

## Fetch, version and availability semantics

`data-v2/fetch.py:207,234` requests monthly WSAEnlilSimulations responses. A new
month is stored as returned; an existing month appends unseen simulationIDs.
Changed fields under a stored simulationID are not updated. This is not a
per-CME latest-as-of version selection and is not a historical public receipt.

Notebook `data-v2/notebooks/raw_to_input_df.ipynb`, source cell `5c8a9436`
(around1127), admits simulations with top-level estimatedShockArrivalTime.
It reduces them to completion, arrival, max(kp90,kp135,kp180), glancing flag,
then deduplicates this four-value tuple. CME/event/simulation IDs and versions
are discarded. Multiple runs for one CME can be counted separately; distinct
simulation IDs with an identical tuple collapse.

Availability is completion time, binned floor(completion,1h)+1h, including
exact-hour completions. At origin O, admitted completion bins are in
[O−7days,O], inclusive. This does not establish publication by origin.

There is no explicit `impactList.location==Earth` filter in this path; top-level
arrival is assumed relevant. An inspected November2024 raw response has an
arrival-bearing record with other spacecraft in impactList. The semantic
Earth guarantee of the top-level field requires official documentation; the
example alone neither establishes nor disproves it.

## Exact formulas

Let dt=estimatedShockArrivalTime−origin, in hours, over the admitted tuples.

|Generated quantity|Implemented rule|
|---|---|
|enlil_n_pending|Count dt>=−6h; no upper arrival-horizon limit|
|enlil_kp|Maximum Kp among pending tuples;0 if none; missing/falsy individual Kp becomes0|
|enlil_next_arr_h|Minimum pending dt;NaN if none|
|enlil_since_arr_h|−max(dt) for−96<=dt<−6;NaN if none|
|enlil_glancing|Whether all pending tuples are glancing;NaN if none|
|enlil_dt_target|h−enlil_next_arr_h, assembled around notebook1257|

Pending counts simulation tuples, not unique CME events, and includes arrivals
predicted up to six hours before origin. Arrival affects admission to pending,
but its numeric location within the72-hour forecast is lost by the retained
count/max reduction. Two different forecast arrival times can therefore map to
the same retained summaries. This is an information distinction, not evidence
that another feature improves official S.

## Filling and the broader available snapshot

`research/rerun_df_ffill_20261103/prepare.py:51` forward-fills each feature by
origin separately within each horizon, validating duplicate keys. The saved
ffill_report.json records zero fills for pending and Kp. By contrast,
enlil_next_arr_h and enlil_dt_target each have1,201,680 filled cells in the
broader snapshot. Reintroducing these prepared columns would inherit stale
scalar/timing semantics; their names do not establish correct re-projection at
later origins. They are not in the current model. The training-y fill stage
changes only y, not these features.

Other excluded information includes since-arrival time, glancing, duration,
individual Kp scenarios, CME speed/direction/width and event identity. The raw
schema containing a field is distinct from a reliable as-issued input archive.

## Decision implication and next checks

After the failed optimization control, a target-qualified issued-arrival
information audit is justified. The external evidence report
`cme_feature_design_evidence.md/json` independently finds training-era messages
whose explicit Earth forecast arrival changes while Kp stays fixed. Before a
fresh cumulative A+B experiment, establish exact parser/issue/revision rules,
causal origin admission, missing/ambiguous cases and any existing controlled
arrival-feature failure. Preserve all current28features; do not silently replace
pending/Kp or use retrospective IPS/prime flags. No propagation epoch may be
invented from a CME ID or a speed without a stated reference height.

Neither adding filled legacy arrival columns nor constructing a new issued
feature is adopted by this source audit. Historical archive immutability and
public receipts remain gaps, and no50% attainability is inferred.

## Official target semantics follow-up

The external researcher compared official WSA-ENLIL/33848/1 UI and its API
record. The UI explicitly labels Earth Shock Arrival Time2024-10-11T04:00Z;
the API uses that same top-level estimatedShockArrivalTime while impactList
contains only Mars and STEREO A. This supports an Earth-specific top-level
convention; OpenAPI itself supplies no explicit universal field description.
Do not infer non-Earth contamination from a missing Earth impactList entry,
and do not add an Earth-in-impactList requirement. Exact source/API comparison
is preserved in cme_feature_design_evidence.md/json. The earlier repository
example was an audit question, not evidence of an implemented target error.

## Duplicate-work and prerequisite check

The Explorer found the following existing exact-parent registered additions
under `research/feature_addition_all_20261108`: add_393(next arrival),
add_394(since arrival),add_395(glancing),add_406(target-relative arrival),
and group_enlil(all four). All three seeds are registered waiting in the
inspected local/saved-remote state; absence of local run directories is not
proof of a remote job's live status. No equivalent new job is launched here.
See that campaign's strategy.md:7,registry.json:8063/9313,
assignment.global.json and core.py:113. These additions use prepared filled
columns, so they do not automatically test coherent event-time reconstruction.

Completed prior drop_enlil_kp/drop_enlil_n_pending/dropgroup_enlil controls
under offset_roll_complete_ablation_20261106 remove existing summaries on a
different30-feature parent. They do not establish a failure of arrival timing.
No completed feature-preserving arrival-addition result on today's parent was
established by this bounded lookup. Waiting/unexecuted conditions are not
negative results.

`research/followup_feature_axes8_20261109/strategy.md:37` separately registers
issued speed,width,direction,issue-age (not the issued arrival-time question).
Its2026 archive is present, but its three issued-CME runs remain blocked_source
until historical body-version evidence is established. Do not duplicate that
kinematics intervention or silently treat archive presence as passing its gate.
The physical information distinction therefore motivates a source/revision
audit next, not another immediate arrival/kinematics training allocation.
