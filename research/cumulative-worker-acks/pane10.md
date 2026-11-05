# pane10 ACK — cumulative worker contract (2026-10-24)

## Read
- `/home/t-lab01/sunrun/AGENTS.md` — SHA256 `03ffed49ee46aebe0979cc0502079bcaddeda83d30bf328eae55a4cc8a584ec7`
- `/home/t-lab01/sunrun/research/CUMULATIVE-WORKER-CONTRACT.md` — SHA256 `9397d582949fea811100eb83983108d54fce7fe7de64a2265114218ca3ac9fec`
- `/home/t-lab01/sunrun/leaderboard/experiments/cumulative_20261024/README.md` — SHA256 `7e44583363b1497272f2cc096cc98df4e5eddc11ef03e4c6da6604cab5e821ea`

## Understanding
One frozen direct working parent; the published champion is a separate comparator. Full-period total MSE (>=5% below a comparable champion) is the only success criterion; preofficial/roll3 evidence is selection and diagnosis only. One explicit recipe axis per arm; A/B/A+B share base, rows, seeds, cutoffs and masks; promotions are paired, preofficial and locked. Source admission requires exact availability receipts; a file on disk is not historical availability. Every new feature path needs a real future/past poison probe and positive control. Public seed-averaged MSE and mean per-seed MSE are different aggregations. New lineage runs use `launch.py` only; a failed guard is never bypassed with an old/ad-hoc runner. Shared/registered runners and protected evaluators are immutable. CPU-only, admitted sources only, no official-score tuning, no upload/submission. Existing completed/running outputs are preserved; no broad kill or cleanup. No verified full-period improvement or 5% achievement exists yet.

## Ownership
- pane10 owns only `research/cumulative-worker-acks-20261024/pane10.md` and its lane10 review handoff `research/cumulative-worker-acks-20261024/pane10-review.md`.
- `leaderboard/experiments/cumulative_20261024/` is Codex-owned and **read-only** to me; I will not modify package code, tests, roots, `AGENTS.md`, `research/worker-prompt.md`, or any other lane's files.
- Existing outputs and running jobs (E1755, E1756–E1764/E1765, E1769, E1736 modern/legacy members) are preserved, not modified.

## Next work (read-only review lane)
1. Review the leader's **source admission** logic and its receipts.
2. Review **champion metric aggregation** (mean-seed vs public seed-averaged) without recomputing or fabricating scores.
3. Review the **actual feature-builder ↔ materialized-parquet column/grid linkage**, including base111 column order and block mapping.
4. Write only validation findings/limitations to `pane10-review.md` for the leader. Do not start new ensembles, official adapter chains, training, official evaluation, or false-pass receipts.

## Review delivered
- `research/cumulative-worker-acks-20261024/pane10-review.md` — read-only code/graph review of source admission, champion aggregation, and builder→parquet linkage. It is a findings/limitations report, **not** a pass receipt. No candidate is certified.
- `research/lane10-audit-20261024/REPORT.md` + `audit.json` — actual materialized-envelope audit (2026-10-24T11:54Z). Builder→parquet values pass; semantic poison probe and source-receipt mapping fail. **Not a pass receipt.**
- `research/lane10-audit-20261024/audit_v2.json` — v2 CH-availability addendum (latest 2026-10-24T12:22Z). Non-CH exact, CH rebuild exact for all rows, 24 byte checks/0 failures. v2 poison receipt is structurally bound (artifact/source/ref/builder/canonical validate pass) but its ACE case reuses the v1 flawed copy and has 26/27 ACE-column mismatches. Latest v2 now has `sources_v2.json` and `input-envelope-v2.json`, but both remain unchecked/pending; `SW-CH` is not a catalog ID and v2 still points at old source vintages. v2 is a preparation, not a candidate. **Not a pass receipt.**
- `research/lane10-audit-20261024/source-v3-mapping.json` (`50610625...`) — value-prefix/subset evidence for v3 ACE/CH/SUVI/DONKI. **Not a source-admission receipt.**
- `research/cumulative-worker-acks-20261024/pane10-ace-latency-fail.md` — new blocking handoff (2026-10-24T12:22Z). **ACE is used at the forecast origin instead of `origin-2h`**: at `2025-06-02T00:00Z`, `v_lag0` is 848.2 vs the required 902.2, `ace_temperature_k_m24` 499,595 vs 517,000, and 11 inspected columns fail. This affects v1 and v2; it is a source-rule violation, not float tolerance. The handoff gives exact functions, origin, columns, expected/actual values, and minimal fixes. **No training/promotion until fixed.**

## Latest source mapping (2026-10-24T12:22Z)
- ACE old/current caches are value-identical on 36,034 common hours; CH and SUVI caches are value-identical on 36,015 common rows/all common columns. These prefixes can be mapped without new collection, but the ACE `obs+2h` rule is still missing and SUVI exact-S3-byte evidence is only size/slot-based.
- DONKI old/current are **not** value-identical: 152/155 local files differ and 5/12 live files differ; current `store/v2/donki` is the admitted SW-010 path and explicitly replaces the old paths. ENLIL must be rebased or given a separate historical receipt.
- `sources_v2.json` uses unknown `SW-CH`; CH maps to `SW-005`. `input-envelope-v2.status` remains `preparation_unchecked_not_frozen_ch_source_review_pending`.

## v3 revalidation (2026-10-24T13:10Z)
- `research/lane10-audit-20261024/audit_v3.json` (`9f09bca2...`) and `audit_materialized_v3.py` (`9ff87a55...`) re-ran the same axes on v3 code/paths.
- **ACE `obs+2h` PASS**, actual v3 builder→parquet PASS (11 origins; NaN mismatch 0; max abs 5.21e-3), CH rebuild PASS, SHA/schema/overlap PASS. Builder cutoff is correct: +100 at 23:00 gives 0, at 22:00 gives 100.
- **Two FAILs:** (1) `v3_probe_cutoff_2h`: probe masks still use `ORIGIN`, not `ORIGIN-2h`, so the declared future/past probe does not isolate the 2h source boundary; (2) `v3_source_admission`: IDs are mislabelled (ACE filled `SW-004`/raw `SW-003`, CH `SW-005`, SUVI strip `SW-006`/meta `SW-007`, DONKI archive `SW-009`, current live `SW-010`); old paths/receipts; SUVI exact S3 bytes missing; old DONKI bytes differ from current SW-010 (run-level 954 old runs are a subset of current 962; 8 current-only after 2026-10-08, so no observed ENLIL value delta through 2026-09-26). v3 is still a preparation, not a candidate. No training/promotion.

## Science status
No verified scientific improvement over a comparable champion and no 5% full-period result exists. Prior E1769/E1790/E1604 numbers remain historical/exposed diagnostics, not a new unbiased result.
