# Lane9/Lane12 fix request — causal CH base (lane10 audit failed)

Status: **blocking; no baseline/candidate training or registration**
Source: lane10 `research/lane10-audit-20261024/audit.json` (`overall.pass=false`)
Leader: pane7

## Lane9 — row-level evidence and replacement block

1. Deliver row-level `suspect_415` evidence in a new dedicated path:
   `origin`, selected `slot`, `obs_end`, `s3_modified`, `available_at`,
   base111 value, causal replacement value/mask, per-row `leak` and
   `has_causal_delta` for each of the 3 delta variants.
2. Confirm the 415 = 467 base non-null minus 52 complete causal 24h deltas.
   Pane7 independently sees 467 leak rows, 64 causal current-only, 403 no
   causal current; explain the 415/52 boundary.
3. Keep `ch_replacement_block.parquet` SHA
   `a5e41c876a03987d7f91b205f525ed8e1687d99840a2226d1ba0ee1a14edda83`
   and its mask columns. Verify exact `(origin,h)` keys, no `y`, 2022-10..11
   all-mask-zero for 371 rows and 7 partially available rows as recorded.
4. Re-run `poison.probe` on the final replacement builder and bind the receipt
   to the frozen builder code. Preserve old receipts.

## Lane12 — causal base v2

1. Fix `raw_ace_actual_vs_probe`: materialized ACE bt/by/bz/m6/m24 values must
   match the actual raw-ACE builder/recompute bit-for-bit (or within a documented
   deterministic tolerance) and the builder code hash must be the one audited.
2. Replace/exclude every base111 `ch_*` column affected by the lane9 415-origin
   look-ahead. Keep the same 10,688 origins, labels, target cutoff, seeds and
   111 ordered feature names; use the causal replacement values/masks or an
   explicit exclusion. Re-materialize train/f1/f2/f3/official v2.
3. Provide real source receipts for ACE, SUVI/SW-008 and ENLIL/SW-010:
   path, byte hash, observation end, publication/availability time and
   per-file sample evidence. Do not set `passed=true` for an unknown or partial
   receipt; list unresolved sources as limitations.
4. Rebind `prep_and_poison_binding` to the **current** `prepare_base111_v2.py`
   hash and the final v2 envelope. The canonical validator must accept it.
5. Re-run builder↔parquet equality on every grid and every block, not only
   common rows.

## Lane10 — re-review

After lane9/lane12 fixes, re-run the independent audit and write a new
pass/fail/limitation report. The three current failures must be explicitly
cleared or retained as blocking limitations. Do not pass an unknown receipt.

## Boundary

No baseline reproduction, candidate training, preflight, registration,
promotion, official evaluation or upload until lane10 returns a new pass.
Existing artifacts are preserved; each lane writes only its own path.
