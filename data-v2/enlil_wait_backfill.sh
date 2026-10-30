#!/bin/bash
# 2026-10-20: CCMC/iSWA was unreachable (timeouts from here and from outside). Probe one archived file every
# 30 min for up to 48 h; on HTTP 200 run the two ENLIL backfills; on 403 stop (no workaround) and log it.
set -u
cd /home/t-lab01/sunrun/data-v2
PY=/home/t-lab01/sunrun/leaderboard/.venv-fusion/bin/python
URL=https://iswa.gsfc.nasa.gov/iswa_data_tree/model/heliosphere/enlil/IMF_1hour/2025/07/20250701_003038_IMF.dat
LOG=logs/enlil-wait-backfill.log
for i in $(seq 1 96); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 90 -A 'sunrun-data-v2/1.0 (research archive refresh)' "$URL")
  echo "$(date -u +%FT%TZ) probe $i http=$code" >> "$LOG"
  if [ "$code" = 200 ]; then
    $PY -u builders/raw_sources.py enlil_ambient >> "$LOG" 2>&1 && \
    $PY -u builders/raw_sources.py enlil_cme >> "$LOG" 2>&1
    rc=$?; echo "$(date -u +%FT%TZ) backfill rc=$rc" >> "$LOG"; exit $rc
  fi
  if [ "$code" = 403 ]; then echo "$(date -u +%FT%TZ) 403: stop, needs CCMC access" >> "$LOG"; exit 3; fi
  sleep 1800
done
echo "$(date -u +%FT%TZ) gave up after 48 h" >> "$LOG"; exit 4
