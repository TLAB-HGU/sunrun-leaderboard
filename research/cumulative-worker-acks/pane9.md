# pane9 ACK — cumulative worker contract (2026-10-24)

## Read (actual files, hashes verified)

- `/home/t-lab01/sunrun/AGENTS.md` — SHA256 `03ffed49ee46aebe0979cc0502079bcaddeda83d30bf328eae55a4cc8a584ec7`
- `/home/t-lab01/sunrun/research/CUMULATIVE-WORKER-CONTRACT.md` — SHA256 `9397d582949fea811100eb83983108d54fce7fe7de64a2265114218ca3ac9fec`
- `/home/t-lab01/sunrun/leaderboard/experiments/cumulative_20261024/README.md` — SHA256 `7e44583363b1497272f2cc096cc98df4e5eddc11ef03e4c6da6604cab5e821ea`
- Canonical package status as stated by the user: 93 tests + 2 subtests pass, protected unchanged, independent review APPROVE.

## Understanding

One frozen **direct** working parent; the published champion is a separate comparator. Full-period
leaderboard total MSE (>=5% below a comparable champion) is the only success criterion;
preofficial/roll3 results are selection/diagnosis only. Pin code/data/features/source availability,
seed map, cutoffs, grids, masks and aggregation. Public seed-averaged MSE != mean per-seed MSE.
One explicit recipe axis per arm; A/B/A+B share base, rows, seeds, cutoffs and masks. Promotion is
paired, preofficial, locked and rejects stale parents. No official-score tuning, no correction/blend/
stacking/expert mixture, CPU-only, admitted sources only, exact byte-bound availability receipts.
Every new feature path needs a real future/past poison probe and positive control. A failed guard is
never bypassed with an old/ad-hoc runner. The historical June–September window is exposed; a frozen
confirmation documents a score, not a fresh holdout.

## Ownership / forbidden boundaries

- pane9 owns only `leaderboard/experiments/cumulative_20261024_lane9/` and this ACK file.
- `leaderboard/experiments/cumulative_20261024/` (Codex-only), root `AGENTS.md`,
  `research/worker-prompt.md`, package code/tests, protected evaluators and other lanes are
  read-only to me and will not be modified.
- Existing completed/running outputs are preserved; no broad kill/cleanup; no new candidate,
  no new official evaluation, no official-score-based selection/tuning.
- No upload/push/submission; no protected truth/evaluator change.

## Current progress

- CH lane input-block preparation is in progress in the lane9 path.
- Existing CH diagnostics (dev/selection/audit) remain **partial-period diagnostics**, not a
  full-period result and not a 5% claim. They are not used to choose a candidate.
- The canonical package's `poison.probe`/`write_receipt` interface is the required evidence path.
- No scientific improvement over a comparable champion is yet established.

## Next reversible preparation

1. Materialize a CH-delta block on exactly the base111 origin/h grid; keep the base label mask
   and all horizons unchanged.
2. Record the original CH source path, source ID, availability/issue-time rule and hashes.
3. Run `poison.probe` with actual future/past raw perturbations through the CH-derived builder;
   write the bound receipt and `poison-ref.json`.
4. Hand the block, column list and provenance to the leader. Do not select on partial-period
   gains and do not run a new official evaluation.

## Progress update (reversible prep; no candidate, no official evaluation)

- Built `leaderboard/experiments/cumulative_20261024_lane9/ch_delta_block.parquet`:
  614,615 base111 `(origin,h)` rows, 30 CH-delta columns, no `y`, exact key match.
  Block SHA256 `abeaba2d7c1bd224348f3b1ac7f02fe54bc564d01c0ea59826210ae9f19e2d8c`.
- Availability receipt `availability_sw008.json`: SW-008 proxy
  (`obs_end+1h<=origin`, `s3_modified<=origin`), `passed=true`, `causal=true`,
  `byte_bound=true`, `causal_violations=0`, `byte_check_failures=0`.
- `poison.probe` receipt `poison-receipt.json` SHA256
  `f38b2305ac20b574d991c3707b3d3f570450a339b89e38a2385784d17d3a2541`:
  base111 111 columns/1 case, CH 30 columns/3 cases, future_delta=0,
  past_delta>0 on every case.
- Critical provenance finding: the unfiltered `ch-hourly` cache has
  `s3_modified > slot` for 55.9% of `ok` slots (5.1% >24h, mostly 2022-08..11).
  The lane9 block filters this; 2022 coverage is ~40%, 2023-2026 ~99-100%.
  The base111 `ch_*` columns were built without this filter and must be audited
  by the leader before a causal CH claim.
  Evidence: base111 `ch_ch_t035_meridian30_equatorial_area` equals the
  unfiltered cache at 100% of 10,528 non-null origins; in 2022-10..11, 467 base
  origins are non-null but only 52 have a filtered CH delta (415 leak).
- Handoff: `leaderboard/experiments/cumulative_20261024_lane9/{README.md,HANDOFF.md}`.
- Still no full-period CH result and no 5% claim; partial-period CH diagnostics
  used the unfiltered cache and are not selection evidence.

## Delivery to leader + lane12 (actual paths/SHA)

- `leaderboard/experiments/cumulative_20261024_lane9/DELIVERY.md`
- `ch_replacement_block.parquet` SHA256
  `a5e41c876a03987d7f91b205f525ed8e1687d99840a2226d1ba0ee1a14edda83`
  (30 causal CH levels + 30 masks + age/image flag; no `y`).
- `suspect_415_audit.parquet` SHA256
  `805bdde00df0e72df8058d52ebe8a60bf382000dceaa7733e88643e70f2b6ae3`
  (415 origins × 30 features; exact SW-008 file SHA/obs_end/S3).
- `suspect_415_audit.json` SHA256
  `aec45e8cfd795d8d4d13e9c481145c0892cc13254e43a742f1e80cbab5cbb92f`
- `replacement_vs_base.json` SHA256
  `b9ec4798224f442311a759781dbd57b0a73ea94d96a5c0014436c74d20f52668`
- `positive_control.json` SHA256
  `70e8409f07d45f78e79ed9be9c7053f2c64bc435eead04f8a3c33a8d8bba3d4c`
- `replacement-poison-receipt.json` SHA256
  `e256fbff5aa10815ba46cc12ec16e8dd17d2fc94520b3881eb0ab863b8938589`
- 415/415 origins and 12,450/12,450 rows are leaks; 12,090 rows have no
  eligible causal slot. lane12's 613,993 training keys are 100% covered.
- No new official evaluation, no candidate, no score-based selection; no
  full-period improvement or 5% claim.

### Lane12 integration mapping

- `leaderboard/experiments/cumulative_20261024_lane9/lane12-integration.json`
  SHA256 `b3534b239f38aaa70746111348ea59ee07e2ff2c20af26edfca6aa5664dfa1de`.
- lane12 `train.parquet` keys 613,993; covered 613,993; lane9-only 622;
  lane12-only 0. Join on `(origin,h)`, replace same-named `ch_ch_*`, carry the
  `*_avail` masks and `ch_image_available`, drop the 622 extras.

## Round 2 — source binding + lane12 CH comparison

- Source binding: pass 9 / fail 2 / unknown 6. ETag/checksum absent from DB,
  collector code, WAL, sidecars, XML cache and NetCDF header. Exact-byte
  equality is **unknown**, not pass. Old `availability_sw008.json`
  `byte_bound=true` is superseded for exact-byte claims by
  `availability_sw008_binding.json` (`byte_bound="unknown"`).
- Files: `source_binding_audit.parquet` (`1d449246…5461cb`),
  `source_binding_audit.json` (`d577f93e…a4d612`),
  `source_rule_matrix.json` (`0c78aec6…526644`),
  `SOURCE-BINDING-AUDIT.md` (`d400595b…1dd898`).
- lane12 v2 CH vs lane9 replacement: values match (max abs 7.448e-9, 0 diffs
  >1e-6) but masks differ on **5,881 keys / 101 origins**, all with
  `ch_source_age_h > 6h` (v2 applies a 6h cap; lane9 does not). v3 absent.
- Specific rows: `lane12_ch_compare_v2.parquet` (`13deda64…73bc6`),
  `lane12_ch_mask_mismatch_origins.parquet` (`cc0396b2…8bdefb`),
  `lane12_ch_compare_summary.json` (`aa3b4082…173b59`).
- Full handoff: `leaderboard/experiments/cumulative_20261024_lane9/DELIVERY-ROUND2.md`
  SHA `e89af6ae…9738940`; earlier `DELIVERY.md` SHA `6d39b237…116d41`.
- No official evaluation/tuning; no performance improvement claim.

## Round 2 critical — full base111 CH leak scan

- The 415 audit was only the 2022 Oct–Nov target-nonnull extreme subset.
- Full scan (`all_ch_leak_scan.parquet`, SHA `4fa21724…b2a2ae6`):
  10,525 / 10,528 base-nonnull origins leak; 315,750 / 315,840 base CH
  feature rows leak under `base_available_at = max(slot, obs_end+1h,
  s3_modified) > origin`.
- Severity: 10,059 same-slot 0–1 h; 26 1–6 h; 26 6–24 h; 414 >24 h;
  3 no leak; 160 no source.
- Outside the 415: 10,110 leak any; **63 >1 h; 37 >6 h; 11 >24 h**.
  The 11 severe outside are 2025-03-11/12 (4) and 2026-02-11 (7).
- The 10,059 same-slot cases are ~1 h publication latency; still non-causal
  under the strict rule. The 415 are data outages.
- Action: replace **all** base CH columns, not only the 415. Lane12 v2 is
  causal (`obs_end+1h <= origin and s3_modified <= origin; slot <= origin;
  origin-slot <= 6h`) and matches lane9 values; its mask differs only for
  101 origins with source age >6 h.
- Files: `all_ch_leak_severity.json` (`054d7d38…e8246636`),
  `all_ch_leak_outside_415.parquet` (`9cc6be9c…da3d4cde`),
  `scan_all_ch_leaks.py` (`8f93976e…7a09bab`).
- Full handoff: `leaderboard/experiments/cumulative_20261024_lane9/DELIVERY-ROUND2.md`.
- No official evaluation/tuning; no performance improvement claim.

## Round 2 — replacement block full invariants

- `verify_replacement_invariants.py` checked all 614,615 rows / 10,688 origins:
  30 values + 30 masks, unique keys, no `y`, 0 mask/value mismatches, age 1–82 h,
  no negative/zero, constant across h, `all_pass=true`.
- 611 rows / 10 origins have `ch_image_available=1` but all 30 CH features NaN;
  use `*_avail` masks for feature availability, not the image flag.
- Files: `replacement_invariants.json` (`38690c02…be25a05`),
  `verify_replacement_invariants.py` (`9ade00e2…157876a7`).
- No official evaluation/tuning; no performance improvement claim.

## Round 2 — base CH provenance

- All 30 base111 CH columns match the `ch-hourly` cache exactly at the 6 h
  as-of slot: 315,840 / 315,840 values, max abs diff 0.0. The full leak scan's
  source attribution is exact; the leak is in the cache selection, not a
  base111 recomputation.
- Files: `base_ch_provenance.json` (`53ced8ac…de7544d`),
  `verify_base_ch_provenance.py` (`e7dad239…efa8c29`).
- No official evaluation/tuning; no performance improvement claim.
