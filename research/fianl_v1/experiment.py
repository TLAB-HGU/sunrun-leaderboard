"""Retrain plus7 after dropping unused lead columns and two block-specific inputs, without ridge."""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENT = 'fianl_v1'
PARENT = 'e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel_plus7'
ARCHIVED_DATA = Path.home() / 'sunrun-union9-plus7/input/model-table-union9-plus7.parquet'

loader = importlib.util.spec_from_file_location('fianl_v1_parent', HERE.parent / PARENT / 'experiment.py')
parent = importlib.util.module_from_spec(loader)
loader.loader.exec_module(parent)
gbm, common, np, pd = parent.gbm, parent.common, parent.np, parent.pd

# Each column below was unused by all archived full models (F1–F3, seeds 42–44).
# Continuation stages with zero weight do not contribute to the forecast.
UNUSED_LEADS = {
    0: {
        'strip_v550_dark': (7, 8, 9, 10),
        'strip_v750_logmed': (14, 15, 16, 17),
        'strip_v350_logmed': (22, 23, 24),
        'strip_v450_logmed': (23, 24),
        'onset700_hazard': (4, 5, 6),
    },
    24: {
        'strip_v550_dark': (35, 36, 37),
        'strip_v750_logmed': (41, 42, 43, 44),
        'strip_v350_logmed': (45, 47, 48),
        'cme_phase_rec27': (32, 33, 38, 39, 43, 44, 45),
        'onset700_hazard': range(29, 49),
    },
    48: {
        'strip_v450_logmed': (53, 54),
        'strip_v550_dark': (60, 62, 63, 64),
        'strip_v750_logmed': range(68, 73),
        'strip_v350_logmed': (72,),
        'cme_phase_rec27': (50, 51, 52, 55, 56, 57, 62),
        'onset700_hazard': range(49, 73),
    },
}

BLOCK_REMOVALS = {0: ('enlil_earth_speed',), 24: (), 48: ('rec54_consensus',)}


def removed_columns(start):
    return {
        f'{name}_h{lead}'
        for name, leads in UNUSED_LEADS[start].items()
        for lead in leads
    } | {
        f'{name}_h{lead}'
        for name in BLOCK_REMOVALS[start]
        for lead in range(start + 1, start + 25)
    }


class FinalSpec(gbm.GBMSpec):
    def design(self, grid, start, size):
        original = parent.SPEC.design(grid, start, size)
        return np.ascontiguousarray(original[:, KEEP[start]])

    def feature_names(self, start, size):
        original = parent.SPEC.feature_names(start, size)
        return [original[i] for i in KEEP[start]]


KEEP = {}
for start, expected_width in zip((0, 24, 48), (305, 308, 283), strict=True):
    names = parent.SPEC.feature_names(start, 24)
    removed = removed_columns(start)
    if not removed.issubset(names):
        raise ValueError(f'Unknown removed columns for block starting at {start + 1}')
    KEEP[start] = [i for i, name in enumerate(names) if name not in removed]
    if len(KEEP[start]) != expected_width:
        raise ValueError(f'Unexpected input width for block starting at {start + 1}')

fields = {field.name: getattr(parent.SPEC, field.name) for field in dataclasses.fields(parent.SPEC)}
fields.update(
    experiment=EXPERIMENT,
    launch={
        **parent.SPEC.launch,
        'parent_experiment': PARENT,
        'source_ablation': 'frozen plus7 split audit and two block-specific removals',
        'removed_features': {start: sorted(removed_columns(start)) for start in KEEP},
        'block_widths': [len(keep) for keep in KEEP.values()],
        'postprocess': ['seam xf8', 'phase advance 3 h', 'seed-agreement uplift 40'],
    },
)
SPEC = FinalSpec(**fields)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(common.SEEDS))
    parser.add_argument('--periods', nargs='+', default=list(common.CUTOFFS))
    parser.add_argument('--smoke', action='store_true')
    args, rest = parser.parse_known_args()
    if '-h' in rest or '--help' in rest:
        gbm.main(SPEC, __doc__, HERE)
        return
    if args.data is None:
        if not ARCHIVED_DATA.is_file():
            parser.error('--data must name the archived plus7 input table with the original ENLIL values')
        sys.argv.extend(('--data', str(ARCHIVED_DATA)))

    # The runner writes offset-calibrated predictions per seed. Score only after
    # applying the common postprocessing to the complete set of three seeds.
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
    observations = common.observations_path(args.observations)
    union = parent.UNION
    frames = [pd.read_parquet(results / f's{seed}_predictions.parquet') for seed in common.SEEDS]
    lead_profiles = [frame.pred_kms.to_numpy(np.float64).reshape(-1, common.H) for frame in frames]
    final = union.uplift([union.phase_advance(union.SEAM.smooth(profile)) for profile in lead_profiles])
    for seed, frame, forecast in zip(common.SEEDS, frames, final, strict=True):
        frame['pred_kms'] = forecast.ravel()
        scores = common.score(frame, observations)
        common.write_results(results, f's{seed}', frame, scores)
        logs(f's{seed}', 'scored', total_s=scores['competition']['total']['s'])
    ensemble = common.mean3(frames)
    scores = common.score(ensemble, observations)
    common.write_results(results, 'mean3', ensemble, scores)
    logs('mean3', 'scored', total_s=scores['competition']['total']['s'])


if __name__ == '__main__':
    main()
