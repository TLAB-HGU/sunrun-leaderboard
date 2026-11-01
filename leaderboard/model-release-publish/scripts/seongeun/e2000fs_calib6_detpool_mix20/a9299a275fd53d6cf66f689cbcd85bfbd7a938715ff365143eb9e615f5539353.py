"""e2000fs_calib6_detpool_mix20 -- reproducible script for config a9299a275fd53d6cf66f689cbcd85bfbd7a938715ff365143eb9e615f5539353.

    pred = 0.80 * e2000fs_calib4_detpool_gpd + 0.20 * e2000fs_calib2_mixture2comp

The 20% tail share is pre-registered in research/e2000fs_calib6-20261031/preregistration.md;
no fitting, no walk-forward selector and no truth is used at inference. Reproduce with:

    cd /home/t-lab01/sunrun/research/e2000fs_calib6-20261031
    python mix_lane.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path('/home/t-lab01/sunrun/research/e2000fs_calib6-20261031')))
import mix_lane  # noqa: E402

mix_lane.main()
