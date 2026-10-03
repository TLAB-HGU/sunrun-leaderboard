"""seongeun alias for the frozen ACE+CH candidate."""
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[3]
        / "scripts/jsy301/xgb_ace_ch_official/5247ee2ce72d69d5d8953697f8df47ccea96a5cf3176df7df1247613b6076838.py"),
    run_name="__main__",
)
