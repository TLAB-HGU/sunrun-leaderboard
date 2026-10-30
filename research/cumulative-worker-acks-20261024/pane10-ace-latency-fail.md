# pane10 blocking handoff — ACE `obs+2h` violation and v2 source mapping

Time: started 2026-10-24T12:22Z (21:22 KST); v3 revalidation 13:10Z (22:10 KST). Reviewer: pane10.
Status: **BLOCKING — no training, registration, preflight, promotion, official
evaluation or upload.** This is a review handoff, not a pass receipt.

Latest audits:

- `research/lane10-audit-20261024/audit.json`
  SHA256 `3c55e9c307df33aacda184cd30025d063057b4d4364e06e54288e874535843c2`
- `research/lane10-audit-20261024/audit_v2.json`
  SHA256 `f3b19ed2777ee6721c8169583e6f73ef127501e30ee6853e96720d101a0365e1`
- Audit code:
  `leaderboard/experiments/cumulative_lane10_20261024/audit_materialized.py`
  SHA256 `aa2314e19c76ae408c5fedf85d345ba5e9e2875cbb68d8643ad9731f708ea78f`
- Audit code v2:
  `leaderboard/experiments/cumulative_lane10_20261024/audit_materialized_v2.py`
  SHA256 `3f7a293bdde5c344464208adabba7738d87c6052cb71c260a619ee5f8f967a72`

## 1. New blocking finding: ACE is used at the origin, not `origin-2h`

The contract rule is `ACE observation time + 2h <= forecast origin`
(`research/CUMULATIVE-WORKER-CONTRACT.md`; `AGENTS.md`). The historical
`data-v2/derived/vendor/alfvenicity.py` applies exactly this pattern:
`origins = observation_times + 2h` and `cutoff = origin - 2h`.

The base111 builders do **not** apply it. At
`origin = 2025-06-02T00:00:00Z`, the parquet value equals the ACE observation at
the origin, not at `origin-2h = 2025-06-01T22:00:00Z`:

| column | actual at origin | expected at origin-2h | diff |
|---|---:|---:|---:|
| `v_lag0` | 848.20001 | 902.2 | **-53.99999** |
| `v_lag1` | 877.20001 | 977.9 | **-100.69999** |
| `v_m6` | 924.90002 | 933.9 | -8.99998 |
| `v_sd6` | 56.534378 | 44.304988 | +12.22939 |
| `v_m24` | 911.46252 | 897.70833 | +13.75419 |
| `v_sd24` | 120.24221 | 134.20605 | -13.96384 |
| `ace_speed_kms_last` | 848.20001 | 902.2 | **-53.99999** |
| `ace_speed_kms_m6` | 914.29999 | 925.1 | -10.80001 |
| `ace_speed_kms_m24` | 911.27502 | 894.77 | +16.50502 |
| `speed_d24` | 241.36 | 280.02 | -38.65999 |
| `ace_temperature_k_m24` | 499595 | 517000 | **-17405** |

This is a source-rule violation, not a float-rounding issue. The v2 CH rebase
does not touch ACE, so v2 inherits it.

Exact code paths to fix:

1. `leaderboard/experiments/d143_axisA_analog/a2b/run_official_a2b.py:133`
   `base_grid_for`:
   ```python
   feats = (ia.speed_features(o, ace)
            .join(HR.ace_features(o))
            .join(HR.ch_features(o)))
   ```
   ACE speed/plasma terms must be computed at `o_ace = o - pd.Timedelta(hours=2)`
   and reindexed/aligned to `origin=o`. CH is still computed at `o`; `rec26/27/28`
   are historical `target_t - days` values and are unchanged. Strips are still
   selected at `o`.
2. `leaderboard/experiments/headroom_audit/info_ablation.py:27`
   `speed_features` uses `v_lag0 = s.shift(0)` and rolling windows ending at the
   supplied origin. Call it with `origins - 2h`, then relabel the feature index to
   the real forecast origins.
3. `leaderboard/experiments/headroom_audit/headroom.py:69` `ace_features` uses
   the raw/plasma rows up to the supplied origin. Call it with `origins - 2h`,
   then relabel the feature index to the real forecast origins.
4. `leaderboard/experiments/d147_e0421_lean/lean.py:73` `build_train` and
   `leaderboard/experiments/cumulative_lane12_20261024/prepare_base111.py`
   `base_frame` use `ia.build` with unshifted origins. Recompute the
   `v_*`, `ace_*`, `speed_d24` and `epam_p1_d6` columns at `origins-2h` and bind
   them to the real `origin` keys, or make an owned builder copy with the same
   shift. Use a new cache key; do not reuse the old cache.
5. `leaderboard/experiments/d148_enlil_live/lean_enlil.py:54` wraps
   `A.base_grid_for`; it inherits the same unshifted ACE path. Fix the inner
   builder or the wrapper, not only ENLIL.
6. `leaderboard/experiments/cumulative_lane12_20261024/prepare_base111_v2.py`
   changes only CH; it does not fix ACE. Apply the fix in v2/v3 and regenerate
   the envelope.

Poison must also move its cutoff to `origin-2h`: future perturbations strictly
after `origin-2h` must be invariant, past perturbations at/before `origin-2h`
must be positive, and the unperturbed probe must match the parquet for each fold.

## 2. v2 direct comparison with production raw ACE

Latest v2:

- `materialized_v2/sources_v2.json`
  SHA256 `5a0c56b491775bdc9c8194a3caf49c267ef90f8b3c418471a4a88e18e77e47b9`
- `materialized_v2/input-envelope-v2.json`
  SHA256 `3d5d9bc04bb3623f214bccf1074b1730c3ab1b1c7ff30af61cfe7a648c01ff69`
- `materialized_v2/poison-receipt-v2.json`
  SHA256 `aa4efc77ffe23722d2f257552075ff67829bca03abbb7410d43620ba559637ed`
- `materialized_v2/probe-summary-v2.json`
  SHA256 `918efb1330e23777085cdf5e4c0a8c67c24f8bb582d5bb784f15de546598931c`
- `materialized_v2/verify_v1_v2.json`
  SHA256 `4b7e6e7679e3e9232be686a36ea9a8748f19f948d388aff3219f9567b9c489a9`

v2 ACE columns match the actual production `HR.ace_features` production path
within float32 tolerance at all 11 audited origins; NaN mask mismatch is 0. The
largest absolute differences are `6.58e-3` (f2 2026-01-31), `2.60e-3` (official
2026-09-26), `2.98e-3` (f2 2025-10-01), and `2.44e-5` (f1 2025-06-02).

The **previous poison-copy FAIL is reproduced**: at `2025-06-02` the v2 receipt's
ACE case still calls `P1.raw_probe_window()` + `P1.ace_features_from_raw()`, not
the actual `HR.ace_features` path. The copy differs in 27 ACE-related columns,
with maximum absolute difference **7,775.83**
(`ace_temperature_k_m24`: actual 499,595 vs probe 507,370.83); `speed_d24`
differs by 14.52 and `ace_speed_kms_m6` by 10.60. The receipt's ACE case is
byte-identical to v1's probe case. Canonical `validate_poison` still passes
structurally; that is not a semantic pass.

The v2 CH probe is better: at the probe origin the selected CH row equals the
actual v2 path (max diff 0.0), and the full v2 CH rebuild is exact. But
`probe-summary-v2.json` still records no unperturbed builder→parquet parity and
uses an ffill/bfill CH window; the full-row v2 audit is what closes the CH value
gap.

## 3. Actual SW-ID/source mapping

Current catalog rules and paths (`data-v2/reports/catalog.json`):
`SW-001` ACE historical; `SW-005` CH hourly (`store/v2/ch`); `SW-006` SUVI Fe195
strip (`store/v2/suvi-fusion`); `SW-008` GOES-18 SUVI L1b raw (historical);
`SW-010` DONKI WSA-Enlil live (`store/v2/donki`). `SW-CH` is **not** a catalog
SW-ID; the CH cache maps to `SW-005`.

What maps without new collection:

- ACE: old `store/ch-v1/ace.parquet` and current `store/v2/ch/ace.parquet` have
  36,034 common hourly timestamps with byte-equal `filled_speed_kms` and
  `was_missing`; current adds 295 later rows. The old cache is a value-identical
  prefix for the evaluation period. Do not claim this fixes the `obs+2h` bug.
- CH: old `store/ch-v1/ch-hourly.parquet` and current `store/v2/ch/ch-hourly.parquet`
  have 36,015 common rows; all 180 common columns are equal; current adds 320
  later rows. The prefix is mappable, but `sources_v2.json` uses the unknown ID
  `SW-CH`; it must be `SW-005` and point at the current manifest.
- SUVI: old `store/suvi-fusion/suvi-hourly-v2.parquet` and current
  `store/v2/suvi-fusion/suvi-hourly-v2.parquet` have 36,015 common rows; all 148
  common columns are equal; current adds 320 later rows. The prefix is mappable.
  The AGENTS operational proxy still requires exact local bytes equal the S3
  object, `S3 LastModified <= origin`, and `observation end <= origin-1h`.
  `ch_availability_v2.json` verifies local file size/slot/obs_end/s3_modified,
  but size is not an exact S3-object byte hash; add ETag/object SHA256 or an
  equivalent reviewed receipt. This is an existing contract rule, not a new one.
- DONKI/ENLIL: old `store/headroom-audit/donki` + `store/donki-live` do **not**
  byte-identically map to current `store/v2/donki`: of 155 local files, only 3
  have the same SHA256 as the current manifest (152 differ); of 12 live files, 0
  have the same SHA256 (5 differ, 7 absent), while the current manifest has 520
  files. The **run-level** content is better: old has 954 unique WSA-Enlil runs,
  current has 962; all 954 old runs are in current, and the only 8 current-only
  runs are after `2026-10-08`, outside the evaluation period. At the five audited
  origins the selected latest `(modelCompletionTime, arrival, kp, isEarthGB)`
  tuple is identical. So the old archive is a **historical value-subset** for the
  evaluation period, not a current byte-identical source. The current
  `store/v2/donki/MANIFEST.json` still says it replaces the old paths for new
  experiments, gives a `vintage_rule` selecting by
  `submissionTime/modelCompletionTime <= origin`, and warns that `fetched_at` is
  not historical availability. Provide a historical receipt for the old subset,
  or rebase ENLIL to current SW-010; do not present the old files as current
  SW-010. No observed ENLIL value delta through 2026-09-26 follows from this
  run-subset equality.

Current v2 `sources_v2.json` still has `status` values
`file_present_review_required`/`numeric_selection_checked_source_review_pending`,
no boolean admission. `input-envelope-v2.json` status is
`preparation_unchecked_not_frozen_ch_source_review_pending`. The ACE/raw-ACE
entries still point to `origin-features-v2.manifest.json`, whose output is
`origin-features-v2.parquet`, not the ACE inputs; that receipt is not a cover.

### 3.1 Actual SW-by-SW disposition

| SW / source | current catalog path or rule | v2/v3 path | required evidence | disposition |
|---|---|---|---|---|
| `SW-004` ACE gap-filled speed | `store/ch-v1/ace.parquet` (filled speed input) | labeled `SW-001` | correct ID + standalone receipt; v3 builder now applies `obs+2h` | **ID/receipt FAIL; value prefix equal to current `store/v2/ch/ace.parquet`** |
| `SW-003` ACE frozen snapshot | raw-ace-snapshot (5 parts) | labeled `SW-001` | correct ID + source-tree/availability receipt; do not merge with filled-speed receipt | **ID/receipt FAIL** |
| `SW-005` CH hourly | `store/v2/ch`, current manifest | `SW-CH` + old `store/ch-v1/ch-hourly.parquet` | rename to `SW-005`, bind current manifest; old prefix is value-identical on 36,015 rows | **ID/receipt FAIL; no new download needed** |
| `SW-006` SUVI Fe195 strip | `store/v2/suvi-fusion` | labeled `SW-008` + old `store/suvi-fusion` | correct ID; local bytes equal S3 object, `S3 LastModified<=origin`, `obs_end<=origin-1h`, slot age; old prefix is value-identical on 36,015 rows | **proxy values PASS; exact S3 bytes FAIL** |
| `SW-007` SUVI hourly metadata | stage `suvi_hourly` metadata | labeled `SW-008` | correct metadata ID; `obs_end`/`S3 LastModified` selection evidence | **ID FAIL** |
| `SW-008` SUVI L1b raw | historical raw source | stage path currently shares the strip sidecar | raw L1b object bytes/publication or a separate historical receipt; do not use the strip sidecar as the raw receipt | **receipt FAIL / unresolved** |
| `SW-009` DONKI archive | old `store/headroom-audit/donki` / `store/donki-live` | labeled `SW-010` | historical receipt for the 954-run subset; old/current bytes differ (152/155 local, 5/12 live) but 954 old runs are a subset of current 962; 8 current-only after 2026-10-08 | **ID/historical receipt FAIL; no observed value delta through 2026-09-26** |
| `SW-010` DONKI live | `store/v2/donki`; `vintage_rule = submissionTime/modelCompletionTime <= origin`, never `isMostAccurate`; replaces old paths | not used by v3 | rebase to current path/vintage rule for new experiments; current has 962 runs | **current path not used** |

What is **allowed proxy evidence** (do not demand a new download): SUVI
`slot<=origin`, `obs_end+1h<=origin`, `S3 LastModified<=origin`, and the CH
selection derived from that metadata. What is **historical evidence**: the old
ACE/CH/SUVI caches are value-identical prefixes and can be documented as such.
What is **not established**: exact local-bytes-equal-S3-object for the selected
SUVI images, a standalone ACE/raw and CH upstream receipt, and the current
SW-010 vintage for the old DONKI files. Those remain pending; do not fill them
with booleans.

## 4. Minimal conditions for the next training run

Do not train/register/preflight/promote until all of these are true:

1. ACE speed and plasma features at forecast `origin` are computed from
   observations at `origin-2h`; the 11 values above pass with the documented
   float32 tolerance.
2. The actual ACE builder, not `raw_probe_window`, is the poison path; the ACE
   masks use **`> origin-2h` for future and `<= origin-2h` for past** (not the
   forecast origin), the future perturbation is invariant, the past perturbation
   is positive, and the unperturbed builder output matches each materialized
   grid. v3 passes the builder half but still fails this mask declaration.
3. Source IDs/paths are current and explicit: `SW-001` ACE with the 2h rule;
   `SW-005` CH (not `SW-CH`); `SW-006`/`SW-008` SUVI with the exact-byte proxy
   evidence; `SW-010` DONKI current vintage or a separate historical receipt.
4. v2/v3 has a frozen `sources_v2`/input envelope and an ACE-fixed poison
   receipt; no boolean is set without evidence.
5. Only then may the leader start baseline/candidate training. The v2 CH
   rebase itself is a valid preparation: non-CH exact, CH rebuild exact, and
   causal CH selection all passed. It is not a candidate and has no score.

## 5. v3 revalidation (2026-10-24T13:10Z / 22:10 KST)

v3 was delivered after the v2 handoff. I re-ran the same audit axes against the
v3 paths and v3 builder code, not against v2 or the v1 probe copy.

- Audit: `research/lane10-audit-20261024/audit_v3.json`
  SHA256 `9f09bca20838a5d19d4676860abdb2a7ba56d6d550e6bb3f036377b1b93288ed`
- Audit code: `leaderboard/experiments/cumulative_lane10_20261024/audit_materialized_v3.py`
  SHA256 `9ff87a5587a4f810a2a936ad08cd2ceb9dbc60d06de8db3f5e0218407571733e`
- Overall: **FAILED, `v3_probe_cutoff_2h` and `v3_source_admission`.** No candidate/pass receipt.

Passed on v3:

| Check | Result |
|---|---|
| `SHA256SUMS` | **PASS** |
| 111 features / recipe / `ace_recompute` rule | **PASS** |
| schema, keys, rows, no duplicates | **PASS** |
| train↔f1/f2/f3 overlap (all 111; 60,083/59,974/53,462 rows) | **PASS (0 NaN mismatch, max abs 0.0)** |
| CH rebuild from `ch-hourly` + v2 SW-008 rule | **PASS (mask 0, max abs <1e-8, causal/stale 0)** |
| **ACE `obs+2h` at `2025-06-02`** | **PASS** — `v_lag0` 902.2, `v_lag1` 977.9, `v_temperature_m24` 517,000; all 11 inspected values within float32 tolerance of `origin-2h` |
| actual v3 builder → parquet, 11 origins (f1/f2/f3/official 2 each, train 3) | **PASS** — NaN mismatch 0, max abs 5.21e-3 |
| v3 poison structural binding + canonical `validate_poison` | **PASS** |
| v3 poison values | **PASS** — full raw snapshot at `origin-2h`; 0 ACE mismatches; unperturbed max abs 2.85e-5; future delta 0, past delta 100 |
| **v3 poison cutoff declaration** | **FAIL** — builder correctly gives delta 0 at 23:00 and 100 at 22:00, but `prepare_base111_v3.py:588-592` still uses `ORIGIN`, not `ORIGIN-2h`, for the ACE future/past masks. The aggregate probe can pass without isolating the 2h boundary. |
| v3 availability evidence | **PASS** — 24 byte checks, 0 failures; 0 causal/staleness violations |

Exact v3 actual/expected at `origin-2h` (all pass, float32):

| column | v3 actual | expected | diff |
|---|---:|---:|---:|
| `v_lag0` | 902.20001 | 902.2 | +1.22e-5 |
| `v_lag1` | 977.90002 | 977.9 | +2.44e-5 |
| `v_m6` | 933.90002 | 933.9 | +2.44e-5 |
| `v_sd6` | 44.304989 | 44.304988 | +4.29e-7 |
| `v_m24` | 897.70831 | 897.70833 | -2.03e-5 |
| `v_sd24` | 134.20605 | 134.20605 | +6.37e-6 |
| `ace_speed_kms_last` | 902.20001 | 902.2 | +1.22e-5 |
| `ace_speed_kms_m6` | 925.09998 | 925.1 | -2.44e-5 |
| `ace_speed_kms_m24` | 894.77002 | 894.77 | +1.95e-5 |
| `speed_d24` | 280.01999 | 280.02 | -1.10e-5 |
| `ace_temperature_k_m24` | 517000 | 517000 | 0 |

Exact cutoff reproduction at `origin = 2025-06-02T00:00Z`, `lag = 2025-06-01T22:00Z`:

- perturb only `2025-06-01T23:00Z`: ACE plasma max delta **0.0**, speed max delta **0.0** (correctly ignored);
- perturb only `2025-06-01T22:00Z`: ACE plasma max delta **100.0**, speed max delta **100.0** (correctly included).

So the v3 **builder** cutoff is correct. The **receipt's declared probe** is not: it perturbs every row `> ORIGIN` as future and `<= ORIGIN` as past, so rows in `(ORIGIN-2h, ORIGIN]` are mislabeled. Fix `prepare_base111_v3.py` ACE masks to `> ORIGIN - 2h` / `<= ORIGIN - 2h` **and pass `origin=ORIGIN - pd.Timedelta(hours=2)` to the ACE `probe_nan` call** (its `allowed` assertion must use the source cutoff, not the forecast origin). Keep the CH/availability cases at `ORIGIN`. Regenerate `poison-receipt-v3.json` and `probe-summary-v3.json`, then rerun `audit_materialized_v3.py`. This is a small probe fix, not a feature rewrite.

Current v3 hashes:

- `train_v3.parquet` `8bf08ab1cafb9cfdbac2a10a17a541363783039ae49ea4bf5c6397d640212076`
- `f1_v3.parquet` `90d4e66f8cc4b05304244379f1e6f5ff3737a37f3cb84369f55315ad8471924b`
- `f2_v3.parquet` `217c9461186a799fd289448b39f13de0706a433f5545ae522cb5b8e3ce8704bc`
- `f3_v3.parquet` `cc0ebff8c9ac35b980e4ecbd5458daed3dee8f491fed7e1047be8c37164d411d`
- `official_v3.parquet` `102a83df3c6fc65a66d01fb2d2e9d793e065825ace85bfe910eca2c4de982d2d`
- `poison-receipt-v3.json` `28736fae42806209f2dea4ca35ec520f0ffd70a89b325d1563f3f133755e0039`
- `probe-summary-v3.json` `9d8db6f1bbce12f6e8b8c9ca97dc0cc44c708c4f41bb11c52065b3fcbaa36e69`
- `verify_v3.json` `b3c9fa668f30cdb8cf36c4050450f49aa88a04fd83a5bdaecb64555ab0fb1a9f`
- `sources.json` `4810504936ffd5138c7177287c53ad61179550f339c5a96296ddf7c334e1295a`
- `ch_availability_v3.json` `8fb70185a791ea89f1665850ad91e0b39edc67163bbd154926e6200ce00c3359`
- source-prefix/subset mapping (not an admission receipt): `research/lane10-audit-20261024/source-v3-mapping.json`
  SHA256 `2091a9aed3987756e57fb3f61afed09762cad0814530665ddc8b04a54d4d750b`

**v3 source admission still fails**, for the same concrete reasons as v2:

- `sources.json` still uses `SW-CH` (not `SW-005`) and the old
  `store/ch-v1/ch-hourly.parquet` path; statuses remain
  `file_present_review_required`/`directory_present_review_required`.
- ACE/raw-ACE receipts still point to `origin-features-v2.manifest.json`, whose
  output is `origin-features-v2.parquet`, not the ACE inputs.
- SUVI still has only cache sidecar/size/slot evidence; exact local-bytes-equal-S3
  object evidence remains missing (the AGENTS proxy rule). Its `SW-008` label is
  also wrong for the strip cache: catalog `SW-006` is the Fe195 strip and
  `SW-008` is raw L1b; the strip/stage entries should use the matching ID.
- `SW-010` still uses old `store/headroom-audit/donki` + `store/donki-live`;
  current `store/v2/donki` says it replaces them, and 152/155 local + 5/12 live
  files differ. At run level the 954 old runs are a subset of current 962; the 8
  current-only runs are after `2026-10-08`, so the five audited selected runs are
  identical. ENLIL needs a historical receipt (or a current rebase for new
  experiments), not necessarily a value retrain for the evaluation period.

So v3 clears the ACE formula/value path, but **the poison cutoff declaration and
source admission/source-vintage binding are not cleared**. Do not
freeze/train/promote v3 until both `v3_probe_cutoff_2h` and
`v3_source_admission` are resolved. If the leader changes the ACE poison masks
to `origin-2h`, rebases ENLIL to current SW-010, and fixes the
SW-CH/SW-005 + ACE/CH/SUVI receipts, I can reuse `audit_materialized_v3.py`
immediately; no new audit design is needed.

## 6. Boundaries

No lane12/canonical/package/shared code was edited by pane10. No training,
registration, preflight, promotion, official evaluation, upload, ensemble or
new adapter chain was started. Existing outputs are preserved.
