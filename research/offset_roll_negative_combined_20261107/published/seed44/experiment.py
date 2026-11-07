"""Frozen single-experiment entrypoint."""
from core import main

EXPERIMENT = 'combined_negative_s44'
CONFIG = {'seed': 44, 'exclude_features': ['ch_ch_t035_equatorial_area', 'strip_v450_logmed', 'rec27', 'strip_v650_dark', 'ch_ch_t055_meridian15_area', 'ace_density_cm3_last', 'strip_v750_logmed', 'ace_density_cm3_m6'], 'alpha': 1.0, 'spectral_hessian_only': True}

if __name__ == "__main__":
    main(locked_experiment=EXPERIMENT, locked_config=CONFIG)
