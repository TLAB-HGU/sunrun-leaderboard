"""Baseline-only entrypoint; schedulers generate each run's locked entrypoint."""
from core import main

EXPERIMENT = "baseline_s42"
CONFIG = {"seed": 42}

if __name__ == "__main__":
    main(locked_experiment=EXPERIMENT, locked_config=CONFIG)
