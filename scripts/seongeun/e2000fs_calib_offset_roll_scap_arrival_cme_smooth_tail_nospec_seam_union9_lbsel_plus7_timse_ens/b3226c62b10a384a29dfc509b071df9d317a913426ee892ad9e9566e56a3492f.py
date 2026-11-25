"""Train and score plus7_timse_ens from its preassembled model table (plus7 and TIMSE components, 50:50)."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "research/e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel_plus7_timse_ens/experiment.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "observations", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--ridge-weights", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ENTRY)]
    inputs = ["--data", str(args.data.resolve()), "--observations", str(args.observations.resolve())]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    for component in ("plus7", "timse"):
        for seed in (42, 43, 44):
            run = out / component / f"s{seed}"
            subprocess.run(command + ["--component", component, "--seeds", str(seed), "--out", str(run),
                                      "--threads", str(args.threads)] + inputs, env=env, check=True)
    subprocess.run(command + ["--finalize", "--plus7-runs", str(out / "plus7"), "--timse-runs", str(out / "timse"),
                   "--weights", str(args.ridge_weights.resolve()), "--out", str(out / "final")] + inputs,
                   env=env, check=True)
    shutil.copy2(out / "final/results/mean3_predictions.parquet", out / "predictions.parquet")


if __name__ == "__main__":
    main()
