"""Cut the 654 MB HUXt Earth cache (solar_lhx72_omni2_20261018/cache/huxt_earth.npz) down to the F1/F2 origins.

Only ``origins_ns`` and ``traj`` (Earth-point speed, 72 h) are used by E1604; the ring arrays are dropped.

    python reproduce/e1604_hsspeak25/make_huxt_slim.py --cache <huxt_earth.npz> --out <DATA_ROOT>/e1604/huxt_traj_slim.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--out', default=str(paths.DATA_ROOT / 'e1604/huxt_traj_slim.npz'))
    args = parser.parse_args()
    with np.load(paths.data('e1604/prepared_inputs.npz')) as z:
        want = [z['origins_ns']]
    with np.load(paths.data('e1604/hsspeak_f1_inputs.npz')) as z:
        want.append(z['f1_origins_ns'])
    want = np.unique(np.concatenate(want))
    with np.load(args.cache, allow_pickle=True) as z:
        origins = z['origins_ns'].astype('int64')
        index = {int(v): i for i, v in enumerate(origins)}
        rows = np.array([index[int(v)] for v in want])  # KeyError: cache does not cover an origin
        traj = z['traj'][rows]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, origins_ns=want, traj=traj)
    print(f'{len(want)} origins -> {args.out}')


if __name__ == '__main__':
    main()
