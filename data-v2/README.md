## Consolidated physical data locations (2026-10-26)

Raw and model-input files now physically reside below this directory. Original
paths are compatibility symlinks; historical manifests and pinned recipes retain
their original paths and bytes. This relocation does not change source admission,
availability rules or the pending scientific validation gate.

| Content | Location relative to `data-v2/` |
| --- | --- |
| Central RawDB and pinned historical snapshots | `raw-store/` (`solar_raw.duckdb`) |
| Model-input store (dataset snapshots written by builders/ and derived/) | `model-inputs/leaderboard-store/v2/` |
| E2000 lag-0 training grids read by `research/*/experiment.py` | `model-inputs/leaderboard-store/e2000fs-lag0/` |
| Frozen ACE+2h grids and labels of E1987/E1988 (historical) | `model-inputs/leaderboard-store/e2000-ace2h/` |
| Prepared E1604 inputs (historical) | `model-inputs/leaderboard-store/e1604/` |
| Repository backup bundles (not a data store) | `backups/` |
| SUVI images | `suvi/data/` |
| SUVI metadata DB and existing WAL | `suvi/solar_g18.duckdb` |
| Near-real-time source files | `collectors/nrt/` |
| Spaceweather collector files | `collectors/spaceweather/` |
| Exported ACE/SUVI Parquet stage | `stage/` |
| Relocated research source overlays and caches | `research-inputs/` |

Exact old/new mappings and inode checks: `reports/consolidation-20261026.json`.
The RawDB service was stopped for relocation, restarted and checked through its
status API. Code, fitted models, experiment outputs and protected evaluator
contents were not rewritten. Resolved-path cache identities may require a fresh
cache; original artifact bytes are preserved. Existing PostgreSQL storage is
outside this file relocation.

> 2026-10-25: raw 원본의 경로·해시·버전·취득 기록은 중앙 `solar_raw.duckdb`가 기준입니다.
> 원본 파일과 기존 manifest는 보존하고, 피처·실험 입력은 기존 파일과 검증 절차를 유지합니다.
> 자동 수집은 재개하지 않았습니다. 로컬 stage 갱신: `sun-img/rawdb_stage.py` (명시적 실행만).

# Solar data v2

Raw refreshes and feature builders write under `model-inputs/leaderboard-store/v2/`
(`paths.STORE`); the RawDB lives in `raw-store/` (`paths.RAW_STORE`). The former
`store/` symlinks in the workspace root and in `leaderboard/` were removed on
2026-10-30, so nothing is addressed through `store/` any more. Historical
protected inputs and registered recipe files remain unchanged. A fetch timestamp
does not establish historical availability. Individual manifests record actual
coverage, upstream gaps and file hashes.

## Experiment inputs

`reports/defaults.json` reports the migration gate. Until it says `ready`, use the
previous pinned inputs for new experiments. After validation, use
`experiments.datav2.paths.pin_snapshot` and `experiments.datav2.recipe.Features`.
The three-seed reference replay is `python -m experiments.datav2.replay` through
ExpOS. Every seed uses the same copied input snapshot, including image eligibility.
Official experiments reject smoke mode and retain the training cutoff. New
experiments use inputs with timestamps at or before the origin; the former ACE
two-hour delay is removed (user instruction 2026-10-29). The `experiments.datav2`
replay keeps it only to reproduce its registered recipe. SOTA/champion selection
still comes from the research sheet.

`CHAMPION_READY.json` is published only after one complete champion build. It
binds all source and feature Parquets. Missing or stale receipts reject a new
snapshot; existing run snapshots remain independent of later refresh failures.

## Refresh commands

Use `leaderboard/.venv-fusion/bin/python data-v2/refresh.py raw|champion|derived`
from any directory. Refreshes hold `model-inputs/leaderboard-store/v2/.refresh.lock`; snapshot publication
holds its shared lock. Per-job logs and nonzero outcomes are retained. Source
downloads resume and serialize their own dataset locks.

The installed user-systemd timers schedule raw refreshes at 00/06/12/18 KST,
champion features at 03:00 KST and remaining features on Sunday at 04:00 KST.
They are enabled only after initial verification. Contention returns temporary
failure and retries after 15 minutes. GPU inference uses the shared GPU lease
pool; CPU maintenance runs at low priority.

Inspect `reports/`, `logs/`, dataset `MANIFEST.json` and `FILES.json` for completion
evidence. Successful collection is independent of scientific adoption. A source
that ends upstream, requires inaccessible weights or retains missing data is
reported with that limitation, rather than marked current.
