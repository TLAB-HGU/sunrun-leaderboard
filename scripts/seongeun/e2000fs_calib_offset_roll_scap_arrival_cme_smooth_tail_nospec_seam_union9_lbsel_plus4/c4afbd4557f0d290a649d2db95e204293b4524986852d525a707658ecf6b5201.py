"""Train and score plus4 from its preassembled model table."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "research/e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel_plus4/experiment.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "observations", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)

    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ENTRY)]
    inputs = ["--data", str(args.data.resolve()), "--observations", str(args.observations.resolve())]
    cpu = ["--device", "cpu", "--threads", str(args.threads)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run(command + ["--out", str(out / "final")] + inputs + cpu, env=env, check=True)

    shutil.copy2(out / "final/results/mean3_predictions.parquet", out / "predictions.parquet")


if __name__ == "__main__":
    main()
