"""Compare the shared runner with the original baseline and ablation runner, bitwise."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np  # noqa: E402
import core  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--period', choices=tuple(core.baseline.CUTOFFS), default='F1')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    old = load('reference_ablation', args.reference_root / 'experiment.py')
    baseline = load('reference_baseline', args.reference_root / 'baseline.py')
    old.baseline = baseline
    checks = {}

    def same(name, left, right):
        equal = bool(np.array_equal(left, right))
        checks[name] = dict(equal=equal, max_abs=float(np.max(np.abs(left - right))))
        if not equal:
            raise ValueError(f'Smoke equivalence failed: {name}')

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        a, b, c = root / 'original', root / 'old_runner', root / 'new_runner'
        p = args.period
        baseline.run_period('e2000fs_calib_offset_roll', args.data, a, p, args.threads, True)
        old.run_period(old.normalize_config({}), args.data, b, p, args.threads, True)
        core.run_period(core.normalize_config({}), args.data, c, p, args.threads, True)
        with np.load(a / f'{p}_predictions.npz') as x, np.load(c / f'{p}_predictions.npz') as y:
            same('original_final', x['pred'], y['pred'])
        for suffix, keys in [('models/prior_predictions', ('raw', 'training_truth', 'origins')),
                             ('raw_predictions', ('raw', 'component', 'preclip', 'offset', 'origins')),
                             ('predictions', ('pred',)), ('no_offset_predictions', ('pred',))]:
            with np.load(b / f'{p}_{suffix}.npz') as x, np.load(c / f'{p}_{suffix}.npz') as y:
                for key in keys:
                    same(f'{suffix}.{key}', x[key], y[key])
        original_state = json.loads((a / f'{p}_models/frozen_calibration.json').read_text())
        new_state = json.loads((c / f'{p}_models/frozen_calibration.json').read_text())
        same('original_offset', np.array(original_state['offset']), np.array(new_state['offset']))
        for key in ('reference_start', 'reference_end', 'calibration_start', 'calibration_end', 'split_boundary', 'purge_hours'):
            if original_state[key] != new_state[key]:
                raise ValueError(f'State differs: {key}')
    result = dict(all_equal=True, period=args.period, checks=checks,
                  xgboost=core.xgb.__version__, data=core.file_identity(args.data))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        core.write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
