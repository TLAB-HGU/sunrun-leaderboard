"""Official-window forecasts for the E0212 recipe (E0105-structure hard switch with per-head seeds, rocv-adopted 6/8 folds -1.40%).

Seed cell 0 (preregistered first seed): level experts random_state=level_seed(0)=0, long head random_state=event_seed(0)=100,
params from d133_t1rocv/e0105r_rocv.fit_params.
"""
import sys
from pathlib import Path

LB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LB / "experiments/expos_runners"))
sys.path.insert(0, str(LB / "experiments/d133_t1rocv"))
import e0105r_rocv as R  # noqa: E402  (E0212 code, unchanged)
import hsplit_official as HO  # noqa: E402

SEED = 0


def main():
    base = R.fit_params()
    HO.run(R, {**base, "random_state": R.level_seed(SEED)}, {**base, "random_state": R.event_seed(SEED)})


if __name__ == "__main__":
    main()
