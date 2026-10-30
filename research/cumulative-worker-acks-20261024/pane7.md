# pane7 ACK — cumulative worker contract (2026-10-24)

## Read
- `AGENTS.md` compact control surface (latest user instruction wins).
- `research/CUMULATIVE-WORKER-CONTRACT.md`.
- `leaderboard/experiments/cumulative_20261024/README.md` (canonical package; 93 tests + 2 subtests passed, protected unchanged, independent review APPROVE).

## Understanding
- One frozen direct working parent; published champion is a separate comparator.
- Pin code/data/features/source availability, seed map, cutoffs, grids, masks and aggregation. Public seed-averaged MSE != mean per-seed MSE.
- One explicit recipe axis per arm. A/B/A+B comparisons keep base, rows, seeds, masks, cutoffs identical. Promotion is paired preofficial and locked; stale-parent overwrite refused.
- Full-period total MSE is final; preofficial evidence is selection/diagnosis only. No official-score tuning, no output blending/stacking, no past official-score parent choice.
- New lineage runs use the canonical `launch.py` only; registered/shared runners and protected evaluators are immutable.
- CPU-only, admitted sources only, no upload/submission without separate authority.
- State, registration, promotion and final-confirmation reservation are leader-only.

## Ownership
- pane7 owns this ACK and pane7 campaign/prep files only.
- `leaderboard/experiments/cumulative_20261024/` is Codex-only and read-only to me.
- Root prompt, `AGENTS.md`, `research/worker-prompt.md`, other lane code and the canonical package are not modified.
- Existing completed/running outputs are preserved; no broad kill/cleanup.

## Next work (reversible prep)
1. Read other lanes' prep artifacts and integrate only their materialized inputs/receipts, without editing their files.
2. Independently review actual source/data provenance and availability receipts.
3. Fix the comparable-champion aggregation (local mean-seed) and the direct baseline recipe.
4. Freeze one campaign input envelope: base111 + CH-delta + HUXt state.
5. Run the paired preofficial sequence base reproduction -> A (target delta) -> B (CH time change) -> A/B/A+B if required -> HUXt state block, with one-axis manifests and paired evidence.
6. Do not fix the mechanism choice in advance; assign by evidence.

## Progress update — CH causal availability (2026-10-24 20:55 KST)
- Read lane9/lane12 prep and independently audited the base111 CH join.
- Confirmed the mechanism: base111 `ch_*` is joined at image slot, not SW-008
  `available_at`; 2022-10..11 has 467 base non-null origins, only 52 with a
  complete causal 24h delta (lane9), i.e. 415 without a causal replacement.
- Pane7 audit: 10,688 origins; 10,528 non-null; 10,547 leak overall; 488 origins
  in 2022-10..11, 467 non-null, 467 leak, 64 causal current-only, 403 no causal
  current. Exact 415 is the full-delta subset; lane10 must reconcile.
- Wrote `research/cumulative-pane7-20261024/CH-CAUSAL-INTEGRATION.md` and
  `research/cumulative-worker-acks-20261024/pane10-causal-ch-request.md`.
- No baseline/candidate training, preflight, registration, promotion or official
  evaluation started. Lane9/lane12 keep their own paths; lane10 review requested.

## Progress update — remediation ledger + comparator cross-check (2026-10-24 21:06 KST)
- Created/updated `research/cumulative-pane7-20261024/REMEDIATION-LEDGER.md`:
  L10-01 ACE builder↔parquet, L10-02 source receipts, L10-03 poison binding,
  L09-01 CH 415 evidence, L12-01 causal base v2, L10-04 re-audit, CMP-01.
- Cross-checked comparator aggregations:
  E1604 local mean-seed `3233.0901`, public mean-prediction `3222.8830`;
  E0472 mean-seed `3281.3071`; E1813 mean-seed `3277.1166`; E1789 `3180.2038` (banned).
- Critical: E1604 is **not** current-contract causal (origin-hour ACE, labels
  through first forecast origin, correction method). E0472 and E1813 use the
  old origin-hour ACE path; causal eligibility is unverified. No causal-eligible
  full-period champion is currently verified.
- Prepared `COMPARATOR-DESCRIPTOR.json` (`gate_ready=false`) and
  `COMPARATOR-CROSSCHECK.md`; public and mean-seed aggregations are separate.
- No baseline/candidate training, preflight, registration, promotion or official
  evaluation started.

## Science status
No verified full-period improvement over a comparable champion and no 5% achievement exists yet. All prior full-period numbers remain historical/exposed evidence, not a new unbiased result.

## Progress update — round 5 (2026-10-24 21:32 KST)
- Read latest lane12 `materialized_v2/HANDOFF.md`, `sources_v2.json`,
  `verify_v1_v2.json`, `input-envelope-v2.json`, and lane9 `DELIVERY.md`.
- Ran lane10's canonical v2 audit directly on the latest v2
  (`research/cumulative-pane7-20261024/lane10-audit-latest/audit_v2.json`,
  SHA `e614ccab...`): fails `v2_poison_semantics` and
  `v2_canonical_envelope`.
- Confirmed the v2 ACE path locally: `ace_speed_kms_last` matches origin-hour
  (9.8e-6) not origin-2h (18.7); lane10 independently finds 11/11 sample
  origins mismatch (`ace_v2_vs_production.json`).
- Wrote `ROUND5-ASSIGNMENTS.md` with exact lane12 v3, lane10 re-audit and lane9
  byte-binding/CH-comparison tasks, verification commands and resume conditions.
- Source rules fixed to the original contracts: ACE obs+2h
  (`AGENTS.md:26`), SUVI operational proxy (`AGENTS.md:35`), ENLIL completion
  time, HUXt `max(obs+2h,LastModified)` (`AGENTS.md:35`). Unknown/ETag gaps stay
  unknown; no new evidence condition invented.
- No baseline/candidate training, preflight, registration, promotion or official
  evaluation started. Lane12/lane10/lane9 remain the owners of their fixes.
