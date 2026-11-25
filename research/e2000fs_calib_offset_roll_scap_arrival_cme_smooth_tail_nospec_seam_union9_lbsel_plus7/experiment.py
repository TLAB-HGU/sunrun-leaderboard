"""Union9 plus4 with ENLIL Earth speed, onset hazard, and rec54 consensus.

Supply the preassembled feature table with --data. The complete three-seed run
retains union9's seam, ridge, phase advance, and seed-agreement uplift unchanged.
Partial-seed runs and --smoke produce unscored raw GBM predictions; finalize with
--predict-only --seeds 42 43 44 using all GBM and frozen ridge checkpoints.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.name
PARENT = 'e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel_plus4'
_loader = importlib.util.spec_from_file_location('union9_plus7_parent', HERE.parent / PARENT / 'experiment.py')
BASE = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(BASE)
UNION = BASE.BASE
np, pd, common, gbm = BASE.np, BASE.pd, BASE.common, BASE.gbm

ADDED_VARYING = ['enlil_earth_speed', 'onset700_hazard', 'rec54_consensus']


class PlusSevenSpec(gbm.GBMSpec):
    def design(self, grid, start, size):
        parent_grid = {**grid, 'E': grid['E'][:, :, :len(BASE.SPEC.varying)]}
        parent = BASE.SPEC.design(parent_grid, start, size)
        extra = grid['E'][:, start:start + size, len(BASE.SPEC.varying):].reshape(len(parent), -1).copy()
        extra[extra == UNION.MISSING] = np.nan
        return np.column_stack((parent, extra)).astype(np.float32, copy=False)

    def feature_names(self, start, size):
        added = [f'{name}_h{lead}' for lead in range(start + 1, start + size + 1) for name in ADDED_VARYING]
        return [*BASE.SPEC.feature_names(start, size), *added]


_launch = {**BASE.SPEC.launch, 'parent_experiment': PARENT,
           'added_constant': [], 'added_varying': ADDED_VARYING,
           'removed_features': [], 'block_widths': [345, 345, 350],
           'source_ablation': 'user-requested three additions to completed plus4',
           'input_contract': 'preassembled plus4 union plus three lead-varying features',
           'preserved_parent_constant': BASE.SPEC.constant,
           'preserved_parent_varying': BASE.SPEC.varying}
_fields = {field.name: getattr(BASE.SPEC, field.name) for field in dataclasses.fields(BASE.SPEC)}
_fields.update(experiment=EXPERIMENT, constant=list(BASE.SPEC.constant),
               varying=[*BASE.SPEC.varying, *ADDED_VARYING], launch=_launch)
SPEC = PlusSevenSpec(**_fields)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(common.SEEDS))
    parser.add_argument('--periods', nargs='+', default=list(common.CUTOFFS))
    parser.add_argument('--predict-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args, rest = parser.parse_known_args()
    if '--help' in rest or '-h' in rest:
        gbm.main(SPEC, __doc__, HERE)
        return
    if args.data is None:
        parser.error('--data must name the preassembled union9-plus7 table')
    common.output_paths(args, HERE, EXPERIMENT)
    original_score = gbm.score
    try:
        gbm.score = lambda *a, **k: None
        gbm.main(SPEC, __doc__, HERE)
    finally:
        gbm.score = original_score
    if args.smoke or sorted(args.seeds) != list(common.SEEDS) or args.periods != list(common.CUTOFFS):
        return
    logs = common.run_logs(args, EXPERIMENT)
    results = args.out / 'results'
    raw = [pd.read_parquet(results / f's{seed}_predictions.parquet') for seed in common.SEEDS]
    ridge = UNION.ridge_forecast(args.data, args.weights, args.predict_only, logs)
    final = UNION.postprocess([f.pred_kms.to_numpy(np.float64).reshape(-1, common.H) for f in raw], ridge)
    observations = common.observations_path(args.observations)
    frames = []
    for seed, frame, matrix in zip(common.SEEDS, raw, final):
        out = frame[common.KEYS].copy()
        out['pred_kms'] = matrix.ravel()
        scores = common.score(out, observations)
        common.write_results(results, f's{seed}', out, scores)
        logs(f's{seed}', 'scored', total_s=scores['competition']['total']['s'])
        frames.append(out)
    ensemble = common.mean3(frames)
    scores = common.score(ensemble, observations)
    common.write_results(results, 'mean3', ensemble, scores)
    logs('run', 'complete', postprocess=_launch['postprocess'], total_s=scores['competition']['total']['s'])


if __name__ == '__main__':
    main()
