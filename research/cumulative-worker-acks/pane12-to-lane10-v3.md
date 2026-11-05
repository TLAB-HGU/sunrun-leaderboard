# Lane12 → lane10: v3 re-review request (full-raw ACE +2h)

Status: **numeric fixes delivered; source admission still pending; no pass claimed.**

Lane10's v2 failures were:
- `raw_ace_actual_vs_probe`: v2 reused `prepare_base111.raw_probe_window()` +
  `ace_features_from_raw()` (date-filtered, `ffill().bfill()`), not the actual
  `HR.ace_features` path; 27 ACE/speed columns differed (max 7,775.83).
- v2 poison semantics / canonical envelope: no unperturbed parity, no
  `sources.json`, receipt not bound to the current code.
- The latest audit's `v2_ace_obs_plus_2h`: v2 used ACE at the origin, violating
  the ACE observation +2h rule.

v3 is at:
`leaderboard/experiments/cumulative_lane12_20261024/materialized_v3/`
with code:
`leaderboard/experiments/cumulative_lane12_20261024/prepare_base111_v3.py`
SHA256 `ec96ab1e24715015125d37276ac4268f195bf031c1dc56ef470c498320510bee`.

What changed:
- `HR.ace_features(origin-2h)` is called on the **full raw snapshot** (5 files,
  36,056 rows, 2022-08-22..2026-10-03) via production `headroom.hourly`;
  no date filter, no raw ffill/bfill. Perturbation writes the full raw frame to
  a temp parquet and redirects the production glob; arithmetic is unchanged.
- All `ace_*`/`speed_d24`/`epam_p1_d6`/`v_*` features use `origin-2h`;
  recurrence uses ACE at target-26/27/28d (already causally past). CH keeps the
  v2 SW-008 rule.
- Pre-fix/default regression (`--default-parity`, lag 0) matches v2 on exact
  keys/111 columns/dtypes/NaN masks; 0 value mismatches, max_abs 0.0
  (`regression-v2-v3-precheck.json`
  `0822b340db012a8ce59dc55ef34493c54cb1821174025376001e7e2f69d02101`).
- Materialized lag-2 regression changes only the 53 expected ACE/speed columns;
  0 unexpected NaN/value/dtype/keys/CH/rec/ENLIL/strip/label changes
  (`regression-v2-v3.json`
  `93fe72b152c8cdfd0c643008000311594a05824078aa109fd71cc7e5a2bd8dfc`).
- Fold-boundary + official representative origins: 8/8 builder checks pass,
  0 NaN mismatches (`verify_v3.json`
  `b3c9fa668f30cdb8cf36c4050450f49aa88a04fd83a5bdaecb64555ab0fb1a9f`).
- Receipt `28736fae42806209f2dea4ca35ec520f0ffd70a89b325d1563f3f133755e0039`;
  ref `a6d67dd78386a54fda49c05882e4ddf70fb0b3e759cae510e57625c0654c9d4e`;
  prep `6b5612398a571296a2c00dda792e23829a01ac9c71a99e24ca9426a87d835914`;
  probe `9d8db6f1bbce12f6e8b8c9ca97dc0cc44c708c4f41bb11c52065b3fcbaa36e69`.
  Canonical `poison.validate_poison` passes.
- `probe-summary-v3.json` contains `unperturbed_parity` (111 cols, 0 NaN,
  max abs 2.848e-05), `ace_parity` (38 ACE/speed cols, 0 mismatch, max abs
  2.441e-05), full-raw ACE 0/100, CH 0/0.123, availability 0/0.013753, and the
  `v2_ace_obs_plus_2h` check passes on all 11 columns.

Local numeric audit:
`materialized_v3/audit-v3-local.json` SHA256
`604dcb393320ab19d11bf67c62bd8a226cdbfcbc70efefc35c83f44ccba81bff`
shows `raw_ace_actual_vs_probe: true`, `prep_and_poison_binding: true`,
`v2_ace_obs_plus_2h: true`, `regression_v2_v3: true`,
`boundary_builder: true`, and `source_manifest_review_pending` as the remaining
limitation (no boolean pass).

Please re-run the independent audit against the **v3 paths/code**, not the v1
`P1.raw_probe_window` copy. For the ACE direct check, compare v3 rows to
`HR.ace_features(origin - 2h)` / `ia.speed_features(origin - 2h)`; v3
intentionally differs from v2 at the origin because of the +2h rule. The
`v2_ace_obs_plus_2h` check should pass on v3. Verification command:
```bash
cd ~/sunrun/leaderboard
env PYTHONPATH=.:experiments/cumulative_20261024:experiments/cumulative_20261024_lane9:experiments/d147_e0421_lean:experiments/d148_enlil_live:experiments/d143_axisA_analog/a2b:experiments/headroom_audit:experiments/d1_longitude_strips \
  .venv-fusion/bin/python -m experiments.cumulative_lane12_20261024.verify_v3
```
Expected: `failed_numeric_checks: []`, `status:
numeric_checks_only_source_review_pending`, `v2_ace_obs_plus_2h: true`.

Compatibility exact-match evidence: `compatibility-v3.json` SHA256
`4c4eab9f40d5420d151b0f7ce2f6989b4914588b5cfdfb92c01de7854c38a173`;
`raw_probe_window()` returns full raw (36,056 rows); `ace_features_from_raw()`
vs actual `HR.ace_features` max diff 0.0.

No canonical/registered-runner/lane9/lane10/other-lane file was modified. No
training, EID, official evaluation or candidate selection was run.
