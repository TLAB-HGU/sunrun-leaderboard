"""roll3 evaluator on synthetic predictions (isolated ExpOS graph via EXPOS_RESEARCH)."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

LB = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent('''
    import numpy as np, pandas as pd
    from expos import core, evaluate, roll3
    from experiments.roll3_common import folds, write
    filled, observed = evaluate._ace()
    core.init()
    core.add_direction("D-roll3-test", "t", "")
    ids = {}
    for name, noise in (("ctl", 60.0), ("good", 40.0), ("same", 60.0)):
        r = core.register("D-roll3-test", "test", name, "true", kind="full")
        ids[name] = (r["id"], noise)
    for name, (eid, noise) in ids.items():
        for k, cutoff, origins in folds():
            t = np.add.outer(origins.values, (np.arange(1, 73) * 3600e9).astype("timedelta64[ns]"))
            y = filled.reindex(pd.DatetimeIndex(t.ravel()).tz_localize("UTC")).to_numpy().reshape(len(origins), 72)
            y = np.nan_to_num(y, nan=450.0)
            for s in (0, 1, 2):
                rng = np.random.default_rng(1000 * s + k + (7 if name == "same" else 0))
                write(core.RUNS / eid, k, s, origins, y + rng.normal(0, noise, y.shape))
    b = roll3.evaluate_roll3(ids["ctl"][0], None, baseline=True)
    assert b["gate"]["noise"]["mde80"] > 0, b["gate"]
    g = roll3.evaluate_roll3(ids["good"][0], ids["ctl"][0])
    assert g["gate"]["pass"] and g["gate"]["pooled_rel"] < -0.3 and g["gate"]["ci95"][1] < 0 and g["gate"]["n_folds_improved"] == 3, g["gate"]
    n = roll3.evaluate_roll3(ids["same"][0], ids["ctl"][0])
    assert not n["gate"]["pass"] and abs(n["gate"]["pooled_rel"]) < 0.02, n["gate"]
    print(roll3.line(b)); print(roll3.line(g)); print(roll3.line(n))
    from expos import arrival
    ab = arrival.evaluate_arrival(ids["ctl"][0], None, baseline=True)
    aa = arrival.evaluate_arrival(ids["good"][0], ids["ctl"][0])
    assert ab["candidate"]["events"] >= 50 and aa["gate"]["events"] >= 50, (ab["candidate"], aa["gate"])
    print(arrival.line(ab)); print(arrival.line(aa))
''')


def test_roll3_synthetic(tmp_path):
    env = {**os.environ, "EXPOS_RESEARCH": str(tmp_path)}
    r = subprocess.run([sys.executable, "-c", SCRIPT], cwd=LB, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "R3|" in r.stdout and "ARR|" in r.stdout
    print(r.stdout)
