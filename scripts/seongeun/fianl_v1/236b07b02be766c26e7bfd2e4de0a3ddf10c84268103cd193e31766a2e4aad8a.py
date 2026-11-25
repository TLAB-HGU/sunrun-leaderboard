"""Reproduce the three-seed fianl_v1 forecast from the archived plus7 data table."""
import os
from pathlib import Path
import sys

entry = Path(__file__).resolve().parents[3] / "research/fianl_v1/experiment.py"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.execv(sys.executable, [sys.executable, str(entry), *sys.argv[1:]])
