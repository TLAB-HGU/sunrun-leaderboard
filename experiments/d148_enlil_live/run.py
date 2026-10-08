"""D148 E0380 re-run with live DONKI WSA-Enlil runs through the official window.

E0380 (D141 arm=enlil) read only store/headroom-audit/donki, a local copy fetched 2026-10-04 that ends 2026-06-30,
so official origins in July-September saw "no pending CME". The DONKI API itself serves runs through 2026-10-07
(receipt: store/donki-live/{fetched_at.txt,SHA256SUMS,hdr_*.txt}); the live June file is identical to the local one.

Only change vs E0380: ia.enlil_runs reads the local files up to 2026-05 plus store/donki-live (2026-06..2026-10).
Training rows (targets < 2026-05-29) are unchanged; causality is the same rule (modelCompletionTime <= origin).
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/d141_e0147_ablation"))
import ablate as A  # noqa: E402

ia = A.ia
LIVE = LB / "store/donki-live"


def enlil_runs():
    files = [f for f in sorted(glob.glob(str(ia.OUT / "donki/WSAEnlilSimulations_*.json"))) if f[-12:-5] < "2026-06"]
    files += sorted(glob.glob(str(LIVE / "WSAEnlilSimulations_*.json")))
    rows = []
    for f in files:
        for s in json.load(open(f)):
            if s.get("estimatedShockArrivalTime"):
                rows.append((pd.Timestamp(s["modelCompletionTime"]), pd.Timestamp(s["estimatedShockArrivalTime"]),
                             max(s.get("kp_90") or 0, s.get("kp_135") or 0, s.get("kp_180") or 0),
                             bool(s.get("isEarthGB"))))
    return pd.DataFrame(rows, columns=["done", "arrival", "kp", "glancing"]).drop_duplicates().sort_values("done")


if __name__ == "__main__":
    ia.enlil_runs = enlil_runs
    ev = enlil_runs()
    print(f"d147 enlil runs={len(ev)} last_done={ev['done'].max()}", flush=True)
    sys.argv = [sys.argv[0], "--arm", "enlil"]
    A.main()
