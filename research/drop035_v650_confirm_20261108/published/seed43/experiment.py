"""Frozen single-experiment entrypoint."""
from core import main

EXPERIMENT = 'drop035_v650_s43'
CONFIG = {'exclude_features': ['ch_ch_t035_equatorial_area', 'strip_v650_dark'], 'tail_weight': 4, 'spectral': True, 'offset': True, 'offset_scope': 'horizon', 'offset_window': 336, 'alpha': 0.25, 'block_size': 24, 'clip_component': True, 'clip_final': True, 'seed': 43, 'spectral_hessian_only': False, 'experiment': 'drop035_v650_s43'}

if __name__ == "__main__":
    main(locked_experiment=EXPERIMENT, locked_config=CONFIG)
