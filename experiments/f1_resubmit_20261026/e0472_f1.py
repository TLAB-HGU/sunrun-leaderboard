"""E0472 (D148 lean+ENLIL, arm s3_std) re-run for the leaderboard F1 period with the F1 training cutoff.

F1 rule (leaderboard, 2026-10-26): F1 predictions come from a model trained and tuned only on data before
2026-02-26 00:00 UTC (F1 start - 72h). The E0472 recipe and code are unchanged; every date is shifted by
exactly three months, so each fit keeps the same distance to the evaluated origins as in the F2 run:

                       F2 (original E0472)                     F1 (this run)
  full base fit        targets < 2026-05-29                    targets < 2026-02-26
  W2 (path fit, blend) train < 03-01, OOF 03-04..05-28         train < 2025-12-01, OOF 12-04..2026-02-22 23:00
  W1                   train < 02-01, OOF 02-04..02-28         train < 2025-11-01, OOF 11-04..11-30 23:00
  W0                   train < 01-01, OOF 01-04..01-31         train < 2025-10-01, OOF 10-04..10-31 23:00
  scenario scale pool  candidates before 2026-01-01            candidates before 2025-10-01
  residual pool        candidate H + 72h < cutoff              candidate H + 72h < cutoff
  origins              2026-05-31 23:00 .. 09-26 23:00 (2833)  2026-02-28 23:00 .. 05-28 23:00 (2137)

The last W2 OOF origin is 2026-02-22 23:00, so every OOF label (blend weight) is before the cutoff; the original
E0472 W2 window used labels up to 2026-05-31, which F1 cannot. Nothing reads a stored prediction or the leaderboard store.

usage: EXPOS_OUT=<dir> .venv-fusion/bin/python experiments/f1_resubmit_20261026/e0472_f1.py [--seeds 0,1,2]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

LB = Path(__file__).resolve().parents[2]
# headroom.py pins LB to /home/t-lab01/sunrun/leaderboard, whose store was relocated on 2026-10-26; point it at this
# tree's store before info_ablation copies LB/OUT, or the pre-June DONKI runs silently vanish from every grid.
sys.path.insert(0, str(LB / "experiments/headroom_audit"))
import headroom as HR  # noqa: E402
HR.LB, HR.RUN, HR.OUT = LB, LB / "store/mse20-four-regimes/run-20261003", LB / "store/headroom-audit"
# SUVI slots count only when their .nc file exists; the stage metadata keeps absolute paths under sunrun/sun-img, which the
# same relocation moved to data-v2/sun-img. Without this every slot is rejected and all 17 strip features are NaN.
sys.path.insert(0, str(LB / "experiments/d1_longitude_strips"))
import strips as ST  # noqa: E402
SUN_IMG_OLD, SUN_IMG = "/home/t-lab01/sunrun/sun-img/", Path("/home/t-lab01/sunrun/data-v2/sun-img")
_resolve_nc = ST.resolve_nc
ST.SUNIMG = SUN_IMG
ST.resolve_nc = lambda path: _resolve_nc(str(SUN_IMG) + "/" + path[len(SUN_IMG_OLD):] if path.startswith(SUN_IMG_OLD) else path)
sys.path.insert(0, str(LB / "experiments/d148_enlil_live"))
import lean_enlil as LE  # noqa: E402 (read-only; patched in-process exactly like its own __main__)
if LE.ia.OUT != HR.OUT or LE.RUNS["done"].min() > pd.Timestamp("2023-01-01", tz="UTC"):
    raise SystemExit("DONKI history not loaded from the pinned store")

L = LE.L
SP = L.SP

F1_CUTOFF = "2026-02-26T00:00:00Z"
F1_ORIGIN_START, F1_ORIGIN_END, F1_N = "2026-02-28 23:00", "2026-05-28 23:00", 2137
F1_WINDOWS = (("W0", "2025-10-01T00:00:00Z", "2025-10-04T00:00:00Z", "2025-10-31T23:00:00Z"),
              ("W1", "2025-11-01T00:00:00Z", "2025-11-04T00:00:00Z", "2025-11-30T23:00:00Z"),
              ("W2", "2025-12-01T00:00:00Z", "2025-12-04T00:00:00Z", "2026-02-22T23:00:00Z"))
F1_SCALE_END = "2025-10-01T00:00:00Z"


def configure(seeds):
    cutoff = pd.Timestamp(F1_CUTOFF)
    last_label = pd.Timestamp(F1_WINDOWS[-1][3]) + pd.Timedelta("72h")
    if last_label >= cutoff:
        raise SystemExit("an OOF label reaches the F1 cutoff")
    SP.CUTOFF_ISO = F1_CUTOFF
    SP.ORIGIN_START, SP.ORIGIN_END, SP.N_OFFICIAL_ORIGINS = F1_ORIGIN_START, F1_ORIGIN_END, F1_N
    SP.OOF_TRAIN_END_ISO, SP.OOF_START_ISO, SP.OOF_END_ISO = F1_WINDOWS[-1][1:]
    SP.SEEDS = tuple(seeds)
    L.CUTOFF = cutoff
    L.WINDOWS = F1_WINDOWS
    L.SCALE_END = pd.Timestamp(F1_SCALE_END)
    L.ia.enlil_runs, L.A.base_grid_for, L.feature_columns = LE.enlil_runs, LE.base_grid_for, LE.feature_columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--f2-check", action="store_true",
                    help="keep the original F2 dates: replays E0472 to check this tree against the submitted F2 file")
    a = ap.parse_args()
    if a.f2_check:
        SP.SEEDS = tuple(int(s) for s in a.seeds.split(","))
        L.ia.enlil_runs, L.A.base_grid_for, L.feature_columns = LE.enlil_runs, LE.base_grid_for, LE.feature_columns
    else:
        configure([int(s) for s in a.seeds.split(",")])
    print(f"E0472-F1 cutoff={L.CUTOFF} windows={L.WINDOWS} scale_end={L.SCALE_END} seeds={SP.SEEDS} "
          f"enlil_runs={len(LE.RUNS)}", flush=True)
    sys.argv = [sys.argv[0], "--arm", "s3_std"]
    L.main()


if __name__ == "__main__":
    main()
