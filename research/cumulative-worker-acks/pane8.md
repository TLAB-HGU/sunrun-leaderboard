# Pane 8 ACK — cumulative experiment contract

## Read
- `/home/t-lab01/sunrun/AGENTS.md` (compact, SHA256 `03ffed49ee46aebe0979cc0502079bcaddeda83d30bf328eae55a4cc8a584ec7`)
- `/home/t-lab01/sunrun/research/CUMULATIVE-WORKER-CONTRACT.md` (SHA256 `9397d582949fea811100eb83983108d54fce7fe7de64a2265114218ca3ac9fec`)
- `/home/t-lab01/sunrun/leaderboard/experiments/cumulative_20261024/README.md` (SHA256 `7e44583363b1497272f2cc096cc98df4e5eddc11ef03e4c6da6604cab5e821ea`)

## Understanding
One frozen direct working parent, one explicit recipe axis per ablation, full-period total MSE
only for the 5 % claim (preofficial = selection/diagnosis), paired protected roll3 evidence,
locked parent promotion, no official-window tuning, no bypassing `launch.py`/guard failures
with old or ad-hoc official runners, no correction/blend/stacking/expert mixture, CPU-only,
admitted sources only, exact source/availability receipts, real future/past poison on every
new feature path, checkpointed/resumable runs, and separate reporting of mean-seed MSE vs
MSE of seed-averaged predictions. No scientific improvement is established by this repair.

## Ownership
- This pane owns only `leaderboard/experiments/cumulative_20261024_lane8/` and this ACK file.
- `leaderboard/experiments/cumulative_20261024/`, root `AGENTS.md`,
  `research/worker-prompt.md`, package code and other lanes are Codex/other-pane owned and
  will not be modified.

## Next work (reversible preparation only)
1. Establish the actual provenance and issue-time availability receipts of the existing
   HUXt source/cache (E1466 TD-WSA HUXt state and the derived Earth trajectory/ring cache).
2. Record the original feature-builder code paths and SHA256 hashes.
3. Materialize a HUXt block on the same `base111` origin/h grid in the lane8 directory.
4. Run `poison.probe` with real future/past raw perturbations on the HUXt block (positive
   control included) and write the bound receipt.
5. Hash code, data and origin keys and leave a leader handoff.
No official evaluation, no independent-parent promotion, no new candidate registration.

## Baseline fact
E1813 (direct HUXt+ENLIL, 3 seeds) full-period seed-mean MSE `3277.1166`; 5 % target
`3061.739` and public champion `3222.883`. Not achieved. Diagnostics (dev/selection/audit)
remain diagnostics only.

## Completion (reversible preparation)

- Lane8 directory: `leaderboard/experiments/cumulative_20261024_lane8/`.
- Materialized `huxt_state` 14-column block on the canonical base111 `(origin,h)` grids:
  train 614,615 rows; f1 210,816; f2 212,544; f3 202,176; official 203,976.
  Key and file SHA256 per split are in `outputs/handoff.json`.
- SW-012 availability receipt: `outputs/availability.json` (`passed/byte_bound/causal = true`),
  bound to the E1466 snapshot/source/profile/state manifests and the production-path
  `source_poison.json` (future state delta 0, past positive control > 0).
- HUXt block poison through the real production path: probe 2025-06-01T03:00Z,
  future rows changed 1,892 / delta 0.0; past row changed 1 / delta 6.0 km/s.
  Fragment: `outputs/poison_receipt_fragment.json`; provisional lane-local receipt:
  `outputs/poison_receipt_provisional.json`.
- Handoff: `outputs/handoff.json` (code/data/origin-key hashes, integration steps,
  limitations). The campaign must merge the `huxt_state` poison group with the
  `base111` group and rebind the final receipt to the full frozen input envelope.
- No official evaluation, no candidate registration, no parent promotion.
- Scientific improvement still not established: E1813 full-period mean-seed 3277.1166,
  public champion 3222.883, 5 % target 3061.739.

## Availability audit (follow-up, 2026-10-24 UTC)

Read-only audit of E1466 boundary/cache vs the base111 CH `available_at` issue.
Lane8 outputs only: `leaderboard/experiments/cumulative_20261024_lane8/outputs/audit/`.
No cache/package/shared/other-lane file changed; no download, official eval, or promotion.

Result: **the literal base111 CH slot-join bug is not present in HUXt**.
`tdwsahuxt_state` selects by `available_utc` and validates
`available_utc == max(obs+2h, Last-Modified)`. All 361,862 native substeps and
34,768 origins satisfy `selected available <= substep`; 5,956/5,956 source rows
match the rule with zero recorded/expected mismatch.

The fallback evidence gap did propagate:
- 2,186 maps have no Last-Modified; 2,157 updates use `obs+2h`.
- 9,616/34,768 origins (2022-10-09..2023-11-15) remain fallback-unknown.
- 2022-10..11: 1,264 state origins, all unknown; 488 train origins split into
  421 unknown (23,312 rows) and 67 pre-coverage missing (3,915 NaN rows).
- Recovering LM from existing local manifests found 8 selected maps used before true availability (9 source rows; one unselected)
  availability: **24 origin violations, 1,656 official block rows, 253
  substeps**. Official = 2,810 verified origins + 23 violations (1,656 rows);
  f1/f2/f3 are fully verified (713,536 rows); train = 434,106 verified /
  176,594 unknown / 3,915 missing.
- The 5 official violation maps: 2026-06-16, 2026-08-19, 2026-08-21,
  2026-09-26 06:04, 2026-09-26 11:54. The 2026-09-26 maps were not available
  until 2026-09-29 (leak 62.9-66.8 h), after the official end.
- Two 2026-09-27 leaks are after coverage (0 block rows); the 2023-09-01
  state-only leak has 0 train rows but 16 history substeps.

Numerical/join checks passed: cache vs `lead_fields.h5` exact (34,768 origins,
0 mismatches); stored-floor production `_forecast_one` exact at 12
representative origins; full raw-profile rebuild <=0.5 km/s (float16); all
train/f1/f2/f3/official keys exact. Lane12 v2 train is a subset: 613,993 common,
622 block-only rows (2026-05-26..28), 0 v2-only; f1/f2/f3/official exact.

Handoff:
- `outputs/audit/HANDOFF.md` SHA `32d11e36549d6f5785ee87917fa0b3f67b6eeee22c03e1ca80458037c1546016`
- `outputs/audit/summary.json` SHA `61a373e52a393a887b0dc322f5c13da5db262c06ec095d7d88ad2eab2b218ed5`
- `outputs/audit/violations.json` SHA `c7d6f11cd73d0fde723a615ef0e916ade3e90209252bd722e8b65407b0c70499`
- `outputs/audit/audit_manifest.json` SHA `5f4eaa09382e3710a428766cfdb7460388c319040933db6659327998d0538355`
- `outputs/handoff.json` SHA `97dc7286bf1b1ea78981198ceee26ab05526c1b39ccf55e3d136939f5ed90320`
- `outputs/availability_causal.json` SHA `7e3300c6d4f4ef5fc74f96a357179bdc2bc8886ff1ab55284c251ee164c699dd` (causal pass=false)
- Causal verdict: `pass=false`; `availability.json passed=true` is a source
  snapshot receipt, not causal certification.

Leader/lane10: exclude or repair the 23 official violation origins (1,656
rows) and the 2023-09 state-only history violation; treat the 9,616 unknown
origins (176,594 train rows) as non-certified; f1/f2/f3 are verified; official
is not. Downstream history contamination after the leaks is not proven and
needs a counterfactual re-run. No scientific improvement is established.
