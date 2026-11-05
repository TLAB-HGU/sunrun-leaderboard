# Lane10 independent review request — CH causal availability

Status: **open; blocking baseline/candidate training and registration**
Requester: pane7 leader
Date: 2026-10-24 20:52 KST

## What to review

1. **Lane9 415-origin claim.** Verify row-level evidence for the base111 CH
   look-ahead in 2022-10-01..2022-11-30:
   - 467 base non-null origins;
   - 52 with a complete causal 24h delta (both legs available);
   - 415 without a complete causal delta.
   Reproduce from `store/scratch/worker-d147/featcache/0c1b...parquet`,
   `store/ch-v1/ch-hourly.parquet`, and
   `~/.local/state/sundb/stage/suvi_hourly/year=*/*.parquet`.

2. **Pane7 independent audit.** Check:
   - `research/cumulative-pane7-20261024/audit_ch_415.py`
     SHA256 `44033cacc6c4f8638b72e5b22c007e3fdb57694fba78c5ba1dafd7020fb1735d`
   - `ch_415_evidence.parquet` SHA256
     `15b1fa9856f1a38f1891aabb04d8c49c8a8bf426b3871a439d1c0e2bd51e8e5b`
   - `ch_415_summary.json` SHA256
     `def5fb3517b8be0fa21ec9baf3bcec27af5447823c2caa860927a7bad2144fd0`
   Confirm or reject pane7's 467/64/403 boundary and reconcile with lane9's 415.

3. **Lane9 causal replacement block.**
   - `leaderboard/experiments/cumulative_20261024_lane9/ch_delta_block.parquet`
     SHA256 `abeaba2d7c1bd224348f3b1ac7f02fe54bc564d01c0ea59826210ae9f19e2d8c`
   - `availability_sw008.json` SHA256
     `48e617194054b8d298a1d337757703f153ce1af7b0f996ca0c701144c91a8c65`
   - `poison-receipt.json` SHA256
     `f38b2305ac20b574d991c3707b3d3f570450a339b89e38a2385784d17d3a2541`
   Verify exact `(origin,h)` keys, 30 columns, no `y`, SW-008 filter, and that
   the receipt was produced by the actual builder (not a metadata-only claim).

4. **Lane12 causal base materialization.** Verify the new base111 from lane12:
   same 10,688 origins, same labels/target cutoff, same 111 ordered feature
   columns, f1/f2/f3/official keys, builder→parquet equality, and that every
   affected base `ch_*` column is replaced by the causal block or explicitly
   excluded. Existing materialized train is
   `leaderboard/experiments/cumulative_lane12_20261024/materialized/train.parquet`
   SHA256 `78b9198cd58f40cfb55e68cb97657c86609d6c38f4c7173fb0c43d531ffcbb6c`;
   feature-list SHA256
   `eec9dbcc22ade11e316203f17e3715c132a4d9fc2c2cf7c8679d8c617cd6425e`.

5. **Source receipts.** Lane12 `sources.json` marks CH (SW-008) and ENLIL
   (SW-010) as `review_required`. Do not convert an unknown/partial receipt into
   a pass. Record each unresolved source as a limitation.

## Required output

Write `research/cumulative-worker-acks-20261024/pane10-causal-ch-review.md`
with:
- pass / fail / limitation for each item 1–5;
- exact reproduced counts and hashes;
- whether the causal replacement block may replace the base CH columns, must be
  added as a separate block, or must be excluded;
- whether the lane12 base is safe to use for baseline reproduction;
- any remaining provenance gap.

Do not edit lane9/lane12/package/shared code. Do not start training,
registration, preflight, promotion, official evaluation or upload. This is a
review only; a pass here is an integrity review, not a 5% or scientific-improvement
claim.

## Update 2026-10-24 21:08 KST

Lane9 delivered the row-level evidence:
- `suspect_415_audit.parquet` SHA
  `805bdde00df0e72df8058d52ebe8a60bf382000dceaa7733e88643e70f2b6ae3`
  (12,450 rows = 415 origins x 30 base CH columns; 12,450/12,450 `leak=true`).
- 12,090 rows have no eligible causal slot; 360 rows have a causal value
  (12 origins with some values; 403 origins all-NaN).
- **Discrepancy to reconcile:** lane9 `positive_control.json` reports
  7 some-available / 408 all-NaN for the 415 subset, while `DELIVERY.md`
  reports 12/403. Explain the differing subset definition.
- Replacement block SHA
  `a5e41c876a03987d7f91b205f525ed8e1687d99840a2226d1ba0ee1a14edda83`;
  lane12 mapping SHA
  `b3534b239f38aaa70746111348ea59ee07e2ff2c20af26edfca6aa5664dfa1de`.

Latest lane10 audits (read-only evidence):
- legacy `audit.json` now fails only `raw_ace_actual_vs_probe` and
  `source_manifest`; `prep_and_poison_binding` passes.
- `audit_v2.json` passes all v2 CH/feature/schema checks and fails only
  `v2_canonical_envelope` (no new canonical envelope receipt yet).

Please reconcile the 415 subset and record the final pass/fail/limitation. Do
not start training, registration or official evaluation.

## Update 2026-10-24 21:12 KST

Latest `audit_v2.json` SHA
`ed67316532a759226c8d7c6e61da7d1ea52d4af3dd423fe200250b9d0c2af41b`
fails:
- `v2_poison_semantics`: 26 ACE mismatches; the v2 probe reuses the v1
  raw-probe copy and has no unperturbed builder→parquet parity.
- `v2_canonical_envelope`: no canonical v2 envelope receipt yet.

All v2 CH/feature/non-CH/schema checks pass. Lane12's v2 poison receipt is
`aa4efc77ffe23722d2f257552075ff67829bca03abbb7410d43620ba559637ed`.
Keep the causal gate closed until these two are cleared.
