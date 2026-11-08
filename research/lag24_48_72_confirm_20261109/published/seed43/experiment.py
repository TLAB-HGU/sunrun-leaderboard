"""Frozen single-experiment entrypoint."""
from core import main

EXPERIMENT = 'lag24_48_72_s43'
CONFIG = {'add_features': ['v_lag24', 'v_lag48', 'v_lag72'], 'tail_weight': 4, 'spectral': True, 'offset': True, 'offset_scope': 'horizon', 'offset_window': 336, 'alpha': 0.25, 'block_size': 24, 'clip_component': True, 'clip_final': True, 'seed': 43, 'experiment': 'lag24_48_72_s43'}

if __name__ == "__main__":
    main(locked_experiment=EXPERIMENT, locked_config=CONFIG)
