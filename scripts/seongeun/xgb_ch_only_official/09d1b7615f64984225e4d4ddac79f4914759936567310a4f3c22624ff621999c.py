"""seongeun alias for the frozen CH-only candidate."""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[3]
        / "scripts/jsy301/xgb_ch_only_official/09d1b7615f64984225e4d4ddac79f4914759936567310a4f3c22624ff621999c.py"),
    run_name="__main__",
)
