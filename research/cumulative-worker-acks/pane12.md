# pane12 ACK — 2026-10-24

## Read
- `AGENTS.md` (compact control surface, 2026-10-24)
- `research/CUMULATIVE-WORKER-CONTRACT.md`
- `leaderboard/experiments/cumulative_20261024/README.md`

## Understanding
- One frozen direct parent; published champion is a separate comparator. No new parent is
  promoted by this preparation. No scientific improvement exists yet.
- One explicit recipe leaf per ablation; common control, fixed rows/seeds/cutoffs/masks.
  Full-period total MSE is the final criterion; preofficial evidence is diagnostic only.
- No new sources, no official-score tuning, no upload/submission, no protected-file changes.
  Registered/shared runners are immutable. Guard failures must not be bypassed with old/ad-hoc runs.
- The package `leaderboard/experiments/cumulative_20261024/` is Codex-owned; this pane only reads it.
- Existing completed/running outputs are preserved. `E1822` completed and remains an unevaluated
  historical artifact, not a promoted result.

## Ownership
- Lane12 only. Own files under `leaderboard/experiments/cumulative_lane12_20261024/` and
  `research/cumulative-worker-acks-20261024/pane12.md`; do not modify the package, root prompt,
  other lanes, `scorer/`, `expos/`, or protected data.
- Read-only use of the existing base111 builders in `experiments/headroom_audit`,
  `experiments/d1_longitude_strips`, `experiments/d143_axisA_analog/a2b`,
  `experiments/d147_e0421_lean`, `experiments/d148_enlil_live`.

## Preparation performed (reversible, completed)
1. Reconstruct the existing base111 ordered feature frame with the read-only builders.
2. Materialize `train` (target < 2026-05-29), folds F1/F2/F3 hourly origin x72 targetless grids,
   and the final 2833 x72 targetless grid. No official truth.
3. Record the exact 111-column order, keys, labels, target cutoff, seed/3-head/depth3/trees900
   recipe and every difference from the historical E1807 implementation.
4. Run `poison.probe` on the actual base feature builder (future invariance + past positive control)
   and write the bound receipt.
5. Deliver code/data SHA256, ordered-feature hash and source receipts to the leader.
6. Keep the scientific status at "no improvement yet"; wait for the leader's next contract.

## Delivered (preparation only)
- Envelope: `leaderboard/experiments/cumulative_lane12_20261024/materialized/`
  - `train.parquet` 613,993 rows / 10,688 origins, target_t < 2026-05-29, SHA256
    `78b9198cd58f40cfb55e68cb97657c86609d6c38f4c7173fb0c43d531ffcbb6c`
  - `f1.parquet` 210,816; `f2.parquet` 212,544; `f3.parquet` 202,176; `official.parquet`
    203,976 rows / 2,833 origins. All full hourly origin x72, targetless, no `y`/`target_t`.
  - 111 ordered features equal to E1807 (`features_equal=true`); feature-list SHA256
    `eec9dbcc22ade11e316203f17e3715c132a4d9fc2c2cf7c8679d8c617cd6425e`.
- Proof: `poison-receipt.json` SHA256
  `f92356eebed4baa9c1c4ed585a7d46bb309f46cdcdb343eaf94d4b3baa9870d2`;
  `poison-ref.json` SHA256
  `2ccf99ee467b30b6c5b5c7a7de9593499c6270676e9e02eade6fc8549ec84d53`;
  `poison.validate_poison` passes. Probe origin 2025-06-02T00:00Z: future delta 0.0
  (552 rows mutated), past delta 100.0 (1153 rows mutated).
- Recipe/data differences: `diff-vs-E1807.json`. Fold grids are hourly (E1807 period grids
  were 3h); no training or EID was performed.
- Source paths/SHA and receipt references: `sources.json`; CH and ENLIL have no standalone
  receipt and remain review-required. Not claims of admission.
- Full handoff: `leaderboard/experiments/cumulative_lane12_20261024/materialized/HANDOFF.md`.
- Code: `prepare_base111.py` SHA256
  `ca3203342e9556fa82e7ace3e216c69ec92e62f704c5722d866df92ec96c398f`.
- Scientific status: **no improvement yet**. No official truth or official score was used.

## v2 CH-availability rebase (preparation only)
- Read lane9's actual evidence: `suspect_415_audit.json` (SHA
  `aec45e8c…bb92f`), `suspect_415_audit.parquet` (`805bdde…6ae3`),
  `availability_sw008.json` (`48e61719…8c65`), `ch_delta_block.parquet`
  (`abeaba2d…d8c`), plus README/HANDOFF.
- Independent reconciliation: 2022-10..11 v1 base CH has 467 non-null origins;
  52 were already causal and 415 are suspect. Of the 415, 403 have no causal
  replacement and 12 have a causal value (360 rows, 0 equal, max diff 0.01122).
  v2 has 64 non-null (52 + 12) and 403 lost, matching the lane9 audit exactly.
- v2 envelope: `leaderboard/experiments/cumulative_lane12_20261024/materialized_v2/`.
  Same keys/label/111 order; only the 30 `ch_*` columns replaced.
  `train_v2` SHA `198c3bdd…ac11`; `f1_v2` `080f90b9…fe91`; `f2_v2`
  `ea4ca9f8…8783`; `f3_v2` `13877ece…405a`; `official_v2` `d783df83…081c`.
  Feature-list hash remains `eec9dbcc…425e`; non-CH columns unchanged (0).
  Global identity: all 5 frames have identical `(origin,h)` key sets
  (613,993 / 210,816 / 212,544 / 202,176 / 203,976) and all 81 non-CH
  columns plus `target_t`/`y` are bit-identical; only the 30 `ch_*` changed.
  `ch_filtered_v2.parquet` `ed675569…5c8`; `ch_availability_v2.json`
  `cadad835…dd63`; `sources_v2.json` `5a0c56b4…47b9`; `diff-v1-v2.json`
  `d62b30d8…4672`; `HANDOFF.md` `75a5d399…6d9e`.
- Builder↔parquet: `verify_v1_v2.json` (`4b7e6e76…89a9`) rebuilds 7 sample
  origins; v1/v2 CH max abs diff ≤ 1.1e-8 / 7.3e-9, zero mismatches at 1e-6;
  non-CH v1/v2 changed = 0. 2022-10-15 v2 CH is NaN by the causal rule.
- Probes: `poison-receipt-v2.json` (`aa4efc77…7ed`), ref `3ce518ad…9f47`.
  ACE future/past 0/100; CH values 0/0.123; CH availability 0/0.013753
  (one past unpublished slot made available). `validate_poison` passes.
- `ch_availability_v2.json` (`cadad835…dd63`) status is
  `numeric_selection_checked_source_review_pending`; **no boolean pass**.
  SW-008/source review remains with the leader. No training, EID, official
  evaluation or score-based selection. Scientific status: **no improvement yet**.
- Lane9 confirmation request not needed: `suspect_415_audit.parquet` gives
  415 suspect origins, 12 with a causal available value (360 rows) and 403 with
  no replacement. Our v2 independently gives 467 base non-null -> 64 v2 non-null
  (52 already causal + 12 recovered), 403 lost, 0 new. This is the same audit,
  not a coverage-only inference. Exact 56-vs-64 difference in the delta block
  is the two-leg delta requirement, not a disagreement about the 403.


## v3 production full-raw ACE +2h + causal CH (preparation only)
- Fixed the two lane10 numeric failures and the latest ACE latency check:
  v2's poison copy used a 2025-04-15..06-25 window + `ffill().bfill()` and
  differed from actual `HR.ace_features` in 27 ACE/speed columns (max 7,775.83).
  v3 uses actual `HR.ace_features(origin-2h)` on the full raw snapshot via the
  production `headroom.hourly` path; no raw date filter and no raw ffill/bfill.
  ACE observation +2h <= origin is now enforced for all `ace_*`/`speed_d24`/
  `epam_p1_d6`/`v_*` features. CH keeps the v2 SW-008 causal rule.
- Regression first: `--default-parity` (`lag_hours=0`) matches v2 on exact
  `(origin,h)` keys / ordered 111 columns / dtypes / NaN masks; 0 value
  mismatches, max_abs 0.0 (`regression-v2-v3-precheck.json`
  `0822b340…02101`). Materialized `lag_hours=2` changes only the 53 expected
  ACE/speed columns; 0 unexpected NaN/value/dtype/keys/columns/CH/rec/ENLIL/
  strip/label changes (`regression-v2-v3.json` `93fe72b1…b8dfc`).
- v3 frames (new values; same keys/111 columns/labels):
  train `8bf08ab1…212076`, f1 `90d4e66f…71924b`, f2 `217c9461…8704bc`,
  f3 `cc0ebff8…4d411d`, official `102a83df…982d2d`.
- Boundary + official representative builder checks: 8/8 pass, 0 NaN
  mismatches (`verify_v3.json` `b3c9fa66…fb1a9f`).
- Probes: ACE future/past 0/100 (11,715/24,333 rows); CH values 0/0.123;
  CH availability 0/0.013753 (1 slot). Unperturbed 111-column parity 0 NaN,
  max abs 2.848e-05; ACE-only parity 38 cols 0 mismatch, 2.441e-05.
  `v2_ace_obs_plus_2h` passes on all 11 checked columns.
- Receipt `28736fae…55e0039`; ref `a6d67dd7…54c9d4e`; prep
  `6b561239…d835914`; probe `9d8db6f1…a36e69`; recipe `8865387d…da23c0`.
  Canonical `poison.validate_poison` passes; current code hashes bind.
- Local audit `audit-v3-local.json` `604dcb39…81bff`;
  `raw_ace_actual_vs_probe` true, `prep_and_poison_binding` true,
  `regression_v2_v3` true, `boundary_builder` true, `v2_ace_obs_plus_2h` true.
  `source_manifest_review_pending` remains false/limitation; no boolean source
  pass.
- Code: `prepare_base111_v3.py` `ec96ab1e…510bee`; `verify_v3.py`
  `73f3568e…1cdb8d`; `HANDOFF-v3.md` `87505f1b…adf7f1`; compatibility exact
  match `compatibility-v3.json` `4c4eab9f…38a173`.
  Handoff: `leaderboard/experiments/cumulative_lane12_20261024/materialized_v3/HANDOFF-v3.md`.
- Verify command:
  `cd ~/sunrun/leaderboard && env PYTHONPATH=.:experiments/cumulative_20261024:experiments/cumulative_20261024_lane9:experiments/d147_e0421_lean:experiments/d148_enlil_live:experiments/d143_axisA_analog/a2b:experiments/headroom_audit:experiments/d1_longitude_strips .venv-fusion/bin/python -m experiments.cumulative_lane12_20261024.verify_v3`
- v1/v2 SHA256SUMS still verify; no canonical, registered runner, lane9/lane10
  or other-lane file was modified. No training, EID, official evaluation or
  candidate selection. Scientific status: **no improvement yet**.
