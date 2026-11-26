"""Reproduce the fianl_v1 January-exclusion forecast (variant A) from data-v2/input/df.parquet."""
import os
from pathlib import Path
import sys

entry = Path(__file__).resolve().parents[3] / "research/fianl_v1_janexcl/run.py"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.execv(sys.executable, [sys.executable, str(entry), "--variant", "A", *sys.argv[1:]])
