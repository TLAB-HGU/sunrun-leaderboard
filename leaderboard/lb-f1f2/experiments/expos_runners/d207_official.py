"""Official-window forecasts for the E0207 recipe (E0102-line D33 hard switch, rocv-adopted 8/8 folds -1.93%).

Seed cell 0 (preregistered first seed; all three heads random_state=0), params from d133_t1rocv/rocv_runner.fit_params.
"""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/expos_runners"))
sys.path.insert(0, str(LB / "experiments/d133_t1rocv"))
import rocv_runner as R  # noqa: E402  (E0207 code, unchanged)
import hsplit_official as HO  # noqa: E402

SEED = 0


def main():
    p = R.fit_params(SEED)
    HO.run(R, p, p)


if __name__ == "__main__":
    main()
