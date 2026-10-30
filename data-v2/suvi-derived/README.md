# SUVI v2 derivation

`build.py` reads fixed existing feature transforms and writes only the separate
`leaderboard/store/v2/{euvwave,suvi-dim,chmorph,chmask,fe171}` outputs. It performs
no fitting and reads no model predictions or evaluator truth. Frozen transforms
retain their original thresholds, missing values and training cutoff.

Run with `leaderboard/.venv-fusion/bin/python`, `nice -n 10`,
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. `--end` accepts a UTC timestamp or
`now` (the current UTC hour). `registry.json` lists the ordered refresh commands.

- `catalog`: Fe195 NetCDF SHA256, single-part S3 ETag/MD5 and size binding.
  Unchanged historical public object identities are reused only after fresh
  local hashing. New files receive a S3 HEAD request.
- `euvwave`: fixed hourly-cadence wave features. Existing historical rows can
  seed the new cache only if code/config and historical source membership,
  byte identity and publication timestamps match. Both future publication
  exclusion and an admitted-pixel positive control are checked.
- `suvi-dim`: preserves the original feature cache's training/official origin
  grid and appends hourly origins after its end. It uses source-bound cached
  maps and products, with unchanged transform keys; the wrapper hash is recorded
  separately. No OOF predictions are read to determine origins.
- `chmorph` and `chmask`: require the completed v2 champion image caches.
  Metadata is snapshotted per build. Morphology is the existing coarse 12x3
  CH-grid proxy, not a newly inferred pixel segmentation.
- `fe171`: snapshots the collector database and WAL, exports using the collector's
  existing DuckDB environment, and builds the 148-column raw hourly cache.
  Historical feature reuse requires the original complete metadata identity to
  match. **Raw hourly features are not origin-time-admissible by themselves.**
- `fe171-catalog`: full public S3 listing and local byte binding for Fe171;
  subsequent refreshes list only days containing newly seen local files.
- `fe171-verify`: writes `metadata-bound.parquet`, causal horizon strip features
  and selections, runs future/positive-control poison tests, then marks the
  Fe171 manifest admitted. Consumers must use the bound metadata/selection.

All origin-time selectors require observation end + 1 hour <= origin and
NetCDF S3 LastModified <= origin, with latest-earlier fallback. A build time is
not an observation timestamp. Manifests distinguish origin ranges, actual
source observation edges and coverage gaps.

`verify.py` re-hashes feature outputs and compares every historical row against
v1, including the 2,833 official origins, without reading any target labels.
`finish.py` waits for separately owned source/champion jobs and runs their
remaining dependent transforms; missing prerequisites after six hours are a
failure, never a completion record.
