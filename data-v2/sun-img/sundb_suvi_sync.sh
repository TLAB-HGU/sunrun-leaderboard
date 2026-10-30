#!/bin/sh
# Export suvi_hourly -> ~/.local/state/sundb/stage/suvi_hourly (+ sun-db upload) from a copy of the live DB.
# suvi-live.service holds solar_g18.duckdb, and since 2026-10-02 it no longer runs --drive-sync, so the stage
# stopped at slot 2026-10-01 15Z. The copy includes the .wal so it carries every slot written since the last checkpoint.
set -eu
cd "$(dirname "$0")"
tmp=$(mktemp -d /tmp/sundb-suvi-XXXXXX)
trap 'rm -rf "$tmp"' EXIT
cp solar_g18.duckdb "$tmp/solar_g18.duckdb"
[ -f solar_g18.duckdb.wal ] && cp solar_g18.duckdb.wal "$tmp/solar_g18.duckdb.wal"
exec .venv/bin/python sundb_sync.py suvi --db "$tmp/solar_g18.duckdb"
