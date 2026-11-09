# Official metric and baseline audit — 2026-11-08

**Subsequent user clarification:** W explicitly means the implemented official S.
The half-S target below is therefore the confirmed research goal. Earlier
conditional wording records the initial metric audit; it no longer blocks scoring.

## Findings supported by fresh replay

The official metric is **Total S**, version `lead-timse-6h-v1`. Neither the implementation nor the current leaderboard README defines W or declares W an alias for S. A W-reduction claim is therefore not evaluable as written; do not silently rename S. If the requested 50% target is interpreted as the actual official ranking metric S, the present best single forecast sets **S ≤ 48.7462210403**. This is only a conditional translation, not an achieved result.

There are three different references:

| Reference | Total S | Interpretation |
|---|---:|---|
| Actual current best single, `drop_t035_v650_s42` | 97.4924420806 | Current board rank 1, selected seed; not evidence of a reproducibly superior configuration |
| Same new configuration, seeds 42/43/44 | 97.4924420806 / 97.8737083320 / 98.0745134372 | Mean 97.8135546166; sample variance .0874156181; SD .2956613234; worst 98.0745134372 |
| Adopted cumulative incumbent A+B1, seeds 42/43/44 | 98.5914666420 / 99.6857298384 / 100.0014656239 | Mean 99.4262207015; sample variance .5475330263; SD .7399547461; worst 100.0014656239 |
| Historical frozen SOTA `drop_t035_equatorial_s44` | 97.7876205817 | Prior snapshot, now board rank 3; not current SOTA |

The current board was read at immutable dataset revision `f756e1623af7f0cbd9fd9c0fc181902f542580bf`: all **207** result records were inspected. Its best model is also the best single model, so current SOTA and current best-single numerical denominators coincide. The adopted cumulative reference remains a separate configuration. Full ranking is in `metric_board_snapshot.json`. `metric_evidence.json` contains fresh official Total/F1/F2/F3/by-lead replays, MAE (both pooled-pair and equal-lead versions), input identities, environment and source revision. All six single-model score replays match their stored S values within 1e-10 and require zero prediction replacements. Historical SOTA replay also matches exactly. This is prediction-score reproduction, not an independent training rerun.

The new two-feature-removal campaign's existing paired 168-hour CI for ΔS versus original A is **[-3.33195, +0.30722]**, crossing zero. Its own candidate gate is false (`research/drop035_v650_confirm_20261108/analysis/report.md:11`). Its better selected-seed board rank does not justify promoting it over A+B1 or claiming a validated structural improvement. Selection over 207 entries and repeated official-period exposure must remain explicit.

## Metric, grid and missingness

`leaderboard/scorer/competition.py:13` fixes the version and leads 1–72; `:16` fixes half-open periods. `:75` constructs origins at 00/06/12/18 UTC, ensuring all 72 target hours fit in one period. F1 is [2026-01-01,2026-04-01), F2 [2026-04-01,2026-07-01), F3 [2026-07-01,2026-09-30). September 30 is excluded by policy, not a missing file. First origins are period start 00:00; last are March 28, June 27, September 26 at 18:00 UTC. Counts are 348/352/352, total 1,052 origins and 75,744 possible pairs.

For each lead, MSE is evaluated only on finite truth. Exactly 100 observation-quantile intervals produce a velocity-width weighted integral of conditional upper-tail MSE, normalized by observed velocity range. S_h=(MSE_h×TIMSE_h)^(1/4). Total S is the unweighted mean over 72 S_h; pooled Total is recomputed using all three periods per lead, not averaged from F-specific scores (`competition.py:101`, `:151`, `:170`). RMSE and TIMSE displayed totals likewise average lead metrics. MAE is not official ranking output, hence both aggregation variants are labelled in the evidence file.

| Period | Valid / possible pairs | Missing truth |
|---|---:|---:|
| F1 | 17,674 / 25,056 | 29.462% |
| F2 | 17,473 / 25,344 | 31.057% |
| F3 | 17,472 / 25,344 | 31.061% |

Missing truth is excluded without interpolation. Missing/nonpositive/nonfinite predictions are replaced with the latest valid observation at or before origin (`competition.py:88`, `:149`). Duplicate keys, nonhourly origins, out-of-envelope origins, invalid leads and duplicate observation timestamps raise errors. Extra hourly predictions inside the accepted envelope are ignored; the forecast population does not depend on each submission. Current replays have no replacement, so their scores are not buoyed by this fallback.

**Observed concern:** in the frozen December–September observation table, hour modulo 6 = 3 has only 302 valid observations out of 1,216; other residues have 929–975 valid. This large periodic missingness means lead groups sample different populations. The source/quality cause is not proved here; it is a high-priority raw-data audit, not evidence that filling truth would be legitimate. Equal-lead scoring does not remove this population difference.

## Truth provenance and temporal contract

`leaderboard/scorer/competition_data.py:15` names NASA's monthly ACE/SWEPAM hourly archive. `:18` parses 10-field rows, requires hourly HHMM and selects the named calendar month (some files include neighboring months). `:37` accepts status==0 plus positive finite speed. `:47` caches December 2025–September 2026 files and requires exact full hourly timestamp coverage. The canonical parquet discards original status values after masking; therefore audit of bad flags or raw sentinels requires source text, not just this parquet. `competition.load_observations` trusts an already canonical speed_kms column; quality validation must occur upstream.

The frozen campaign observations are exactly equal as dataframes to the official store observations at the historical pinned revision. Store provenance identifies all ten source URLs and existing source identities; these were read, not freshly verified against raw downloads in this slice. No publication-time log, calibration/revision history or processing-latency guarantee is encoded by the scorer. An origin timestamp contract alone cannot prove that every input was available at issue time. Physical observation-time and release-time audits remain necessary.

`leaderboard/README.md:33` sets training cutoff at period start minus 72h: 2025-12-29, 2026-03-29, 2026-06-29 00:00 UTC. Submission metadata enforces declared timestamps only; this does not technically verify model fit lineage. A+B1's existing audit additionally refits its short-blend weights using training slices and verifies parent/child forecasts (`research/post_ablation_five_axes_20261108/baseline_policy.py:119`), but this score-only replay did not rerun that training audit.

## Reproduction and limitations

Run:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python analysis/w50_audit_20261108/metric_reproduce.py
.venv/bin/python analysis/w50_audit_20261108/metric_board_snapshot.py
```

Both completed with exit 0. The first uses the actual official scorer path via explicit import, not a campaign copy. It checks all score differences, full positive finite aligned forecasts and equality of campaign/store observations. The second captures current revision first and reads all result records at that immutable revision. These are diagnostic read-only score audits, not training experiments; no publication or existing campaign mutation occurred.

The repository's referenced `research/CUMULATIVE-WORKER-CONTRACT.md` is absent. Current user instructions and extant root guidance were used; no protected scorer was edited.

**Automatic review rejection:** the attempted command computing new file hashes was blocked by the PreToolUse hook, which stated: “this workspace does not use sha256 or any file hash (user instruction 2026-11-02, AGENTS.md 'No Hashes'). Do not compute, record or verify hashes; record row counts, byte sizes, time ranges and columns instead.” This conflicts with the current user's newer request for hashes. The rejection was not bypassed. Byte size, mtime, immutable dataset revision and git commit were retained; fresh hash verification remains incomplete. Existing store provenance is preserved as evidence of declared provenance, not independently verified integrity.
