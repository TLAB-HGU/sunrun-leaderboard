"""Frozen single-experiment entrypoint."""
from core import main

EXPERIMENT = 'drop_ch_ch_t035_equatorial_area_s43'
CONFIG = {'seed': 43, 'exclude_features': ['ch_ch_t035_equatorial_area']}

if __name__ == "__main__":
    main(locked_experiment=EXPERIMENT, locked_config=CONFIG)
