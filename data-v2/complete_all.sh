#!/bin/sh
# One-time completion of the initial v2 build, resumable across reboots (run by data-v2-complete.service).
# Each step leaves a marker in reports/complete/; a restarted service skips finished steps.
#   raw    : GONG backfill, NOAA ENLIL ambient, PUNCH (network; runs alongside champion)
#   champ  : refresh.py champion  -> store/v2/CHAMPION_READY.json
#   derived: refresh.py derived (after raw + champion)
set -u
ROOT=/home/t-lab01/sunrun/data-v2
PY=/home/t-lab01/sunrun/leaderboard/.venv-fusion/bin/python
M=$ROOT/reports/complete
mkdir -p "$M" "$ROOT/logs"
log() { echo "[$(date -u +%FT%TZ)] $*"; }

raw() {
  [ -f "$M/raw.done" ] && return 0
  rc=0
  for step in gong ambient punch; do
    [ -f "$M/raw-$step.done" ] && continue
    log "raw $step start"
    if [ "$step" = gong ]; then
      $PY "$ROOT/builders/gong_resume.py" >>"$ROOT/logs/complete-raw-gong.log" 2>&1
    else
      $PY "$ROOT/builders/remaining_raw.py" "$step" >>"$ROOT/logs/complete-raw-$step.log" 2>&1
    fi
    r=$?
    log "raw $step rc=$r"
    if [ $r -eq 0 ]; then touch "$M/raw-$step.done"; else rc=1; fi
  done
  [ $rc -eq 0 ] && touch "$M/raw.done"
  return $rc
}

raw &
RAW_PID=$!

if [ ! -f /home/t-lab01/sunrun/data-v2/model-inputs/leaderboard-store/v2/CHAMPION_READY.json ]; then
  log "champion start (workers ${DATAV2_WORKERS:-4})"
  $PY "$ROOT/refresh.py" champion
  r=$?
  log "champion rc=$r"
  if [ $r -ne 0 ]; then
    wait $RAW_PID
    exit 1
  fi
fi
wait $RAW_PID
RAW_RC=$?
if [ $RAW_RC -ne 0 ]; then
  log "raw incomplete; service will retry"
  exit 1
fi

if [ ! -f "$M/derived.done" ]; then
  log "derived start"
  $PY "$ROOT/refresh.py" derived
  r=$?
  log "derived rc=$r (per-job outcomes in reports/refresh-derived.json)"
  touch "$M/derived.done"
fi
log "initial v2 build finished"
touch "$M/all.done"
