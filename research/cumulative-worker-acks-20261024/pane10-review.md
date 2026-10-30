# pane10 read-only review — source admission, champion aggregation, builder→parquet linkage

> **UPDATE 2026-10-24T11:54Z (20:54 KST):** The actual lane12 materialized envelope
> now exists. Its independent audit is in
> `research/lane10-audit-20261024/REPORT.md` and `audit.json` (audit code:
> `leaderboard/experiments/cumulative_lane10_20261024/audit_materialized.py`).
> The code-level “no campaign envelope” statements below are superseded. The
> actual audit found builder→parquet value agreement but a semantic poison-probe
> mismatch and source-receipt mapping failures. A subsequent v2 CH-availability
> addendum (`audit_v2.json`) also passed non-CH invariance and full CH rebuild;
> its poison receipt is structurally bound but still reuses the v1 flawed ACE
> copy, and it has no separate source admission. v2 is not a candidate. Treat
> this file as the earlier code-level review plus the champion section; use
> REPORT.md for the materialized verdict.

Date: 2026-10-24 (UTC). Reviewer: pane10 (independent review lane).

**Status: REVIEW ONLY. This is not an approval, promotion, or pass receipt. No candidate is certified. No code, package, graph, prediction, or other lane output was modified. No training, official scoring, lb-eval, upload, or new ensemble/adapter chain was started. Existing completed/running outputs are preserved.**

## Scope and method

Read-only inspection of the Codex-owned canonical package
`leaderboard/experiments/cumulative_20261024/` at these hashes:

- `build_inputs.py` `a51fe9a23b137b756a13d975a57dfa74d8f0f0e38918a21f6ffb04fa3fb74dfd`
- `gate.py` `2ccfd4c58d1608173b737f96e721f1d6c6c6bae67a2d94dc4978c63859c3e916`
- `runner.py` `c4e821a6174b76792b6cb4ebcace5c7926c14d815420683f3c7c8be418185c37`
- `poison.py` `a4def07e0288b81f7613a4b1ef9ee1952ce19b71a2c886fe0e4d7174fb21bf51`
- `launch.py` `3b2e885e9ad93baf54df8015417fa1c8ed3ae71479d7d027a19962a6f0c7433e`
- `contracts.py` `de3c8dbb1b36852f6bc7c45a19de6946e89fc680d1c0fd2667d6d6c2c3960407`
- `README.md` `7e44583363b1497272f2cc096cc98df4e5eddc11ef03e4c6da6604cab5e821ea`

Also audited read-only: all 525 `lbeval` records in
`research/expos/graph.jsonl`; current `lbeval` summaries for E1604, E1769, E1790,
E0472; historical source receipts under `research/` and `leaderboard/experiments/`;
and the historical base111 builder `leaderboard/experiments/solar_direct111_20261024/run.py`.

The reported canonical test pass (93 tests + 2 subtests; 95 collected test items,
protected unchanged, independent review APPROVE) is accepted as a process fact. It does not close the
semantic gaps below; those tests are synthetic and do not prove actual source or
materialization provenance.

## Verdict summary

| Area | Machine-enforced? | Verdict |
|---|---|---|
| Mean-per-seed aggregation convention | Yes | Correct; `mse_seedmean` is mean of per-seed MSEs |
| Champion arithmetic from `per_seed` | No | Summary is copied, not recomputed at pin/compare |
| Champion grid/evaluator/mask comparability | No | `contract_sha256` is candidate-derived; manual eligibility only |
| Source-specific admission rules | No | Three self-asserted booleans; manual review mandatory |
| Block → source provenance | No | Flat source list; no per-block mapping |
| Builder → materialized parquet equality | No | No output hash/row comparison; probe can pass a one-row builder |
| Actual builder code coverage | No | Caller-selected code paths; trivial wrapper sufficient |
| Candidate-specific envelope exists | — | None found; this review is code-level only |

## 1. Source admission

### What the code enforces

`build_inputs.pin` (line 42) requires each `spec["sources"]` item to have
`id`, `path`, `availability_path`. It reads the availability JSON and requires:

- `source_id == id`
- `passed is True`
- `byte_bound is True`
- `causal is True`

It then records `sha256` of the source file and `sha256` of the availability JSON.
`gate.manifest` requires a non-empty source list, unique source IDs, and valid
hashes. `launch.check_current` requires the frozen source list to match the
manifest, and `poison.write_receipt` binds `source_availability_sha256`.

This correctly prevents silent source-list drift and duplicate source IDs, and
rejects an old receipt that lacks the three booleans.

### Findings

**S1 — Admission is self-asserted; the package does not apply source rules.**
`build_inputs.pin` checks only the three booleans and the `source_id` string. It
does not check observation+2h, Issue time, SRS `:Issued:+1h`, S3 `LastModified`,
byte ranges, product/version, or any other source-specific rule. Its own docstring
states that it checks integrity and shape, not the scientific truth of provenance,
and that independent review remains mandatory. A JSON containing
`{"source_id":"SW-xxx","passed":true,"byte_bound":true,"causal":true}` passes
without supporting evidence. This is a documented limitation, not an automated
admission gate.

**S2 — No block → source mapping.**
`sources` is a flat list; `blocks` maps names to columns only. Nothing binds a
feature block to one or more source IDs, source paths, or availability receipts.
A builder can read an undeclared file, or a declared source can be unused, and the
envelope still validates. `poison.write_receipt` binds the source list globally,
not per block group. Source completeness therefore cannot be established from the
envelope; it requires code-level review of each block's builder.

**S3 — Whole-file hash is not row/byte-level historical availability.**
`runner.file_hash(item["path"])` hashes the entire file. For parquet/DB sources
this pins the file, not the exact rows/bytes used at the origin's issue time. The
contract requires exact source/availability receipts; a whole-file hash alone is
insufficient. Historical receipts found in the workspace (e.g.
`solar_enablephys_gonghourlywsa_20261016/input_snapshot/source_receipt.json`)
contain only `source_id` plus hashes, not the new three booleans, so they cannot be
reused as a new admission receipt.

### Current evidence

At the time of this code-level review, no cumulative-campaign `inputs.json`,
`inputs.unchecked.json`, `manifest.json`, `provenance-files.json`, or
`poison-ref.json` existed under `research/` or `leaderboard/experiments/`. That
is now superseded: the lane12 materialized envelope and its audit are at
`research/lane10-audit-20261024/REPORT.md`. The materialized audit confirms that
source paths/hashes are present but source receipts do not semantically cover
all sources, so **no source is admitted by this review**.

## 2. Champion metric aggregation

### What the code enforces

`build_inputs.pin_champion` (line 24) selects the last `lbeval` graph record for
the named experiment, requires its `per_seed` key set to equal the candidate's
seed set, and stores:

- `mse = event["summary"]["mse_seedmean"]`
- `aggregation = "mean_seed_mse"`
- `contract_sha256 = gate.evaluation_identity(candidate_contract)`
- `source_sha256 = gate.digest(event)`

`gate.compare` (line 309) requires `champion["contract_sha256"] ==
evaluation_identity(candidate_contract)`, requires the champion aggregation to
equal the candidate aggregation, recomputes `parent_mse` from the supplied
full-period parent SSE/count rows, and checks it against `champion["mse"]`. It
also finds the champion's trusted `lbeval` event by `source_sha256` and checks the
event summary against `champion["mse"]`.

### Verified facts

- `expos/lbeval.py` defines
  `mse_seedmean = np.mean(per_seed[*]["mse"])` (mean per-seed MSE), not the MSE of
  seed-averaged predictions. The two aggregations are correctly kept separate.
- All 525 current `lbeval` records are internally consistent:
  `summary.mse_seedmean == mean(per_seed[*].mse)` (0 inconsistencies in the audit).
- E1604: mean-seed `3233.0900956568826` from per-seed
  `3196.265751284845`, `3254.880537195207`, `3248.123998490595`; the public
  seed-averaged `3222.883` is a different aggregation and must not be substituted.
- E1769: mean-seed `3319.1179068614324`; public seed-averaged `3307.938884266501`
  (different aggregation). E1790: mean-seed `3179.9632169195834` (banned ensemble
  diagnostic). E0472: mean-seed `3281.30713749811`.

### Findings

**C1 — The champion summary is not recomputed from `per_seed`.**
`pin_champion` copies `event["summary"]["mse_seedmean"]`. `gate.compare` compares
the champion event summary to `champion["mse"]`, which is the same copied value,
so an inconsistent `lbeval` record (summary not equal to mean of `per_seed`) would
pass both the pin and the compare if the supplied full-period parent SSE matched
the summary. The candidate path is recomputed from SSE; the champion arithmetic
path is not. Current graph records are consistent, but the guard does not enforce
this invariant. The synthetic champion tests use a consistent summary; no
regression covers an inconsistent champion summary.

**C2 — `champion.contract_sha256` is a candidate-side fingerprint, not a
champion-side binding.**
`pin_champion` sets it to `evaluation_identity(candidate_contract)`, and
`gate.compare` only checks that it still equals the current candidate identity.
`lbeval` events contain no grid, evaluator, mask, or prediction-file hash. The
`evaluation_identity` function deliberately excludes `mask_sha256` and `sources`.
Consequently the field cannot prove that the champion was scored on the same
grid/evaluator/mask as the candidate. A champion from a different evaluation
contract with the same seed set could pass. The README/docstring correctly says
the leader must establish comparator eligibility from current records; that
manual requirement is not machine-enforced.

**C3 — "Last `lbeval` record" is selected without uniqueness or eligibility
checks.**
`pin_champion` uses `records[-1]` and does not require exactly one record, nor
does it check the record's `goal`, `gate`, `baseline`, or blend/ensemble status.
A later re-score or an ineligible comparator would silently replace the earlier
record. `source_sha256` binds the chosen event, but the choice itself is implicit.

**C4 — Full-period parent SSE rows are caller-supplied.**
`evidence.py` only exports preofficial paired rows. `gate.compare` expects a
full-period evidence object with parent/candidate SSE rows and a receipt, but no
canonical exporter binds those parent rows to the champion's actual prediction
files. The gate checks the arithmetic against the champion summary, not the
provenance of the rows. The README calls for "real whole-support SSE/count
receipts"; this remains a manual evidence requirement, not a package-enforced
link.

### Verdict

The mean-per-seed convention is correct and the current graph is consistent.
Comparator **equivalence** (same grid/evaluator/mask, eligible direct model, exact
per-seed arithmetic) is **not machine-proven**; it remains a leader-reviewed
condition.

## 3. Featurebuilder ↔ materialized parquet

### What the code enforces

- `build_inputs.pin` hashes the train parquet and all four grids, checks targetless
  grids, checks training targets are before the cutoff, and calls
  `runner.validate_training` and `runner.select_features`.
- `runner.validate_training` checks required columns, UTC keys, unique
  `(origin,h)`, `target_t == origin+h`, finite `y`, and finite features.
- `runner.select_features` checks enabled block columns exist in train and grids,
  are finite, nonconstant in train, non-overlapping, and not duplicate/no-op
  copies of already-selected columns.
- `poison.probe` calls the supplied builder on raw frames, checks output columns
  exactly, checks future invariance and past sensitivity, and records raw-frame
  hashes.
- `poison.write_receipt` binds the artifact digest (train/grid/source/features/
  blocks), source list, builder code hashes, and per-group probe cases.
- `poison.validate_poison` checks the receipt hash, artifact/source binding,
  builder code hashes, group presence, columns, and probe-case invariants.
- `runner.run_stage` re-validates the frozen inputs and reads the materialized
  train/grid parquets; it never re-runs the builder.

### Findings

**F1 — There is no builder-output hash or row-level comparison to the
materialized parquet.**
The poison receipt proves that *a builder with the recorded code hashes* was run
on raw frames and showed future invariance/past sensitivity. It does not prove
that the train/grid parquet values were produced by that builder. A different
materializer that writes the same column names, plus a passing probe receipt,
would pass `validate_poison`. The artifact digest binds the parquet bytes but not
their causal provenance. This is the largest remaining gap and matches the repair
document's own caveat that "semantic linkage between materialized features and the
tested builder still require[s] independent scientific review."

**F2 — The probe does not require the full prediction grid or index alignment.**
`poison.probe` checks raw index/column alignment and builder output column names,
but it does not require the builder output index to equal the expected origin/h
grid. The canonical test fixture returns a one-row DataFrame and passes. A one-row
builder therefore satisfies the probe. The probe is a temporal-behavior test, not
a materialization test; it cannot certify the parquet's rows or values.

**F3 — Builder code paths are caller-selected; the actual feature modules are not
required.**
`poison.write_receipt` accepts any `builder_code_paths`. `validate_poison` only
requires a non-empty set of absolute `.py` files whose hashes match; `launch`
registers whatever is in the receipt. The canonical test passes
`poison.__file__` with a local lambda builder. The actual base111 builder modules
(e.g. `lean.py`, `lean_enlil.py`, `run_official_a2b.py` used by the historical
`solar_direct111_20261024` runner) are not required to appear. A receipt can bind
only a trivial probe wrapper while the real materializer remains unregistered.
The leader must review the builder code set and ensure it includes the real
feature builder and the materializer.

**F4 — No actual campaign envelope exists yet.**
No new cumulative-campaign `inputs.json`, `manifest.json`, or `poison-ref.json`
was found. The historical E1807 `poison.json` is an old three-boolean test, not a
canonical receipt, and cannot be reused. This review is therefore code-level: it
cannot certify any source or builder connection for a future candidate.

### Verdict

**Not proven.** The package can reject a missing/inconsistent receipt, but it
cannot prove that the materialized parquet columns/values came from the tested
builder and admitted sources. A promotion or final confirmation based only on the
canonical test pass would be a false pass for this specific claim.

## 4. Science status (unchanged)

No verified full-period improvement over a comparable champion and no 5% result
exists. Using the mean-seed aggregation that the gate actually compares:

- E1604 mean-seed `3233.0901`; 5% threshold `3071.4356`. E1604 is the comparable
  historical reference, not a new candidate.
- E1769 mean-seed `3319.1179` (public seed-averaged `3307.938884`), above the
  mean-seed threshold.
- E1790 mean-seed `3179.9632` is a banned ensemble diagnostic, above the threshold
  and not eligible.
- E0472 mean-seed `3281.3071`, above the threshold.

These are historical/exposed diagnostics, not fresh holdout evidence. This review
does not create a scientific improvement claim.

## 5. Required before any promotion/confirmation (recommendations; not implemented)

1. **Champion arithmetic:** recompute `mean(per_seed[*].mse)` and require it to
   equal `summary.mse_seedmean` at pin and compare time; require exactly one
   selected `lbeval` record and record comparator eligibility (direct, non-blend,
   non-oracle, same grid/evaluator/mask).
2. **Source admission:** require per-block source IDs/paths/receipt hashes and
   source-specific evidence fields (not just three booleans); record a named manual
   review under the applicable source rule; reject a flat source list with no
   block mapping.
3. **Builder→parquet:** require a registered materializer and a builder-output
   digest (or full row-level comparison) for every block, including base111; run
   the probe on the full expected grid and compare the produced columns/values to
   the materialized parquet; include the actual feature builder modules in the
   registered code set.
4. **Full-period evidence:** bind the parent SSE/count rows to the champion's
   prediction artifacts and evaluator, not only to the champion summary.
5. Do not issue a pass/promotion/confirmation receipt until 1–4 are satisfied.
   A missing check must remain a recorded limitation, never a zero-filled success.

## 6. Immutability and preserved state

- No file outside `research/cumulative-worker-acks-20261024/pane10.md` and this
  review document was written by pane10.
- `leaderboard/experiments/cumulative_20261024/`, root `AGENTS.md`,
  `research/worker-prompt.md`, `expos/`, `scorer/`, and other lane code were read
  only and not modified.
- The last read-only `expos verify` showed `protected_changed: []`.
- Existing E1736/E1755/E1756–E1764/E1765/E1769 outputs and any still-running
  legacy driver were preserved; no new official evaluation, lb-eval, upload, or
  ensemble/adapter chain was started by this lane.

## 7. Latest blocking addendum (2026-10-24T12:22Z): ACE source rule

The actual v1/v2 materialized data reveal a new, more basic failure than the
receipt gaps above:

- **ACE is joined at the forecast origin, not at `origin-2h`.** The contract is
  `ACE observation time + 2h <= origin`; the historical
  `data-v2/derived/vendor/alfvenicity.py` implements `observation_times + 2h`.
  `info_ablation.speed_features` uses `s.shift(0)`, `headroom.ace_features`
  uses rows up to the origin, and the base-grid/training builders join those
  features at the origin. At `2025-06-02T00:00Z`, `v_lag0` is 848.2 vs the
  required 902.2, `ace_temperature_k_m24` 499,595 vs 517,000; 11 inspected
  columns fail. This affects every ACE model, v1 and v2, and the poison probe.
- **v2 direct-vs-production parity passes, but the old poison FAIL persists.**
  v2 ACE columns match actual `HR.ace_features` at all 11 audited origins
  (max abs 6.58e-3, NaN mask 0), but the poison receipt's ACE case is the v1
  `raw_probe_window` copy; it differs at 27 columns, max abs 7,775.83. This is
  the "canonical validate passes while the probe is the wrong path" failure.
- **Source mapping is partial, not admitted.** ACE and CH/SUVI old caches are
  value-identical prefixes of current caches (36,034 ACE hours; 36,015 CH/SUVI
  rows/all columns), so they can be mapped without new collection. But
  `sources_v2.json` uses unknown `SW-CH` (CH maps to `SW-005`), SUVI has only
  size/slot evidence rather than exact S3-object bytes, and DONKI old/current
  are not identical (152/155 local and 5/12 live files differ); current
  `store/v2/donki` explicitly replaces the old paths. `input-envelope-v2`
  remains unchecked/pending.

Minimal next-training gate: fix ACE at `origin-2h` (all `v_*`, `ace_*`,
`speed_d24`, `epam_p1_d6`), rerun the actual builder→parquet parity and poison
with the real path, then map `SW-CH`→`SW-005`, use current `SW-006`/`SW-010`
paths or provide historical byte-bound receipts, and freeze a new envelope.
No training/registration/promotion/official evaluation until then. Exact
functions, origin, columns and expected/actual values are in
`pane10-ace-latency-fail.md`.

**v3 update (13:10Z):** v3 clears the ACE formula blocker. Independent
`audit_v3.json` (`9f09bca2...`) shows `v3_ace_obs_plus_2h` PASS, actual v3
builder→parquet PASS at 11 origins (NaN mismatch 0, max abs 5.21e-3), CH
rebuild PASS, and poison values PASS (full raw at `origin-2h`, 0 ACE
mismatches, unperturbed max 2.85e-5, future delta 0/past 100). Two checks fail:
`v3_probe_cutoff_2h` (builder gives 0 at 23:00 and 100 at 22:00, but the poison
masks still use forecast `ORIGIN`, not `ORIGIN-2h`, so the declared probe does
not isolate the source boundary) and `v3_source_admission` (`SW-CH` not
`SW-005`, old ACE/SUVI/CH/DONKI paths/receipts, old DONKI bytes differ from
current SW-010). At run level the 954 old DONKI runs are a subset of current
962; the 8 current-only runs are after 2026-10-08, so the five audited selected
runs are identical and no ENLIL value delta through 2026-09-26 is observed. v3
is not a candidate and must not be frozen/trained/promoted until both are
resolved. The prefix/subset evidence is recorded separately in
`source-v3-mapping.json`; it is not a source-admission receipt.

**This document is a validation report, not a pass receipt.**
