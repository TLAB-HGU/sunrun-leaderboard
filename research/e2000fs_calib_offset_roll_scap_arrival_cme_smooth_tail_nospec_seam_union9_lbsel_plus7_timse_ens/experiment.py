"""plus7 GBM (smooth-tail MSE 600 + S-cap 180) and a pure-TIMSE GBM (780 rounds) on the same 345/345/350 design,
fixed 50:50 average of the two raw forecasts per seed, then plus7's unchanged post-processing
(seam xf8 -> ridge 5 % -> 3 h phase advance -> seed-agreement uplift) and the equal mean of seeds 42/43/44.

The 0.5 weight is fixed before scoring. The plus7 features were selected on F1-F3 leaderboard scores, so the result
is exploratory. All model fitting uses pre-cutoff training data only.

Usage (CPU only, CUDA_VISIBLE_DEVICES=''):
  python experiment.py --component plus7 --seeds 42 --data TABLE --observations OBS --out RUN    # raw plus7 GBM
  python experiment.py --component timse --seeds 42 --data TABLE --observations OBS --out RUN    # raw TIMSE GBM
  python experiment.py --finalize --plus7-runs DIR --timse-runs DIR --data TABLE --weights RIDGE --observations OBS --out DIR
--plus7-runs / --timse-runs hold one folder per seed ({DIR}/s{seed}/results/s{seed}_predictions.parquet, or the campaign
names union9_plus7_s{seed} / timse_s{seed}). RIDGE holds F{1,2,3}/ridge.npz (union9_lbsel's frozen ridge).
Outputs: OUT/results/{s42,s43,s44,mean3}_{predictions.parquet,score.json} and OUT/logs/.
"""
import argparse
import dataclasses
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.name
PLUS7_ROW = 'e2000fs_calib_offset_roll_scap_arrival_cme_smooth_tail_nospec_seam_union9_lbsel_plus7'
TIMSE_ROW = PLUS7_ROW + '_mse_timse'


def _load(row):
    loader = importlib.util.spec_from_file_location(f'{row}_experiment', HERE.parent / row / 'experiment.py')
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


PLUS7, TIMSE = _load(PLUS7_ROW), _load(TIMSE_ROW)
gbm, common, pd, np, UNION = PLUS7.gbm, PLUS7.common, PLUS7.pd, PLUS7.np, PLUS7.UNION
WEIGHT = .5  # plus7 share; TIMSE gets 1 - WEIGHT
TIMSE_ROUNDS = 780


def timse_spec():
    """TIMSE component: plus7 design, no continuation, gradient = w * error with the competition TIMSE weights."""
    launch = dict(PLUS7.SPEC.launch)
    for key in ('a_stage', 'continuation', 'source_ablation'):
        launch.pop(key, None)
    launch.update(parent_experiment=PLUS7_ROW, tree_budget=TIMSE_ROUNDS, loss='timse', component_weight=1 - WEIGHT,
                  classification='plus7 and pure-TIMSE components, fixed 50:50 forecast average')
    return dataclasses.replace(PLUS7.SPEC, experiment=EXPERIMENT, objective='timse', component='timse',
                               fit_block=None, predict_block=None, rounds=TIMSE_ROUNDS, launch=launch,
                               make_objective=lambda target, context: TIMSE.objective(target, context, 'timse'))


def fit(component, remaining):
    sys.argv[1:] = remaining
    if component == 'plus7':
        PLUS7.main()  # partial-seed runs write raw GBM predictions only
        return
    original = gbm.score
    gbm.score = lambda *args: None
    try:
        gbm.main(timse_spec(), __doc__, HERE)
    finally:
        gbm.score = original


def raw(folder, names, seed):
    for name in names:
        path = folder / name / 'results' / f's{seed}_predictions.parquet'
        if path.is_file():
            return pd.read_parquet(path)
    raise FileNotFoundError(f'No seed {seed} predictions under {folder}')


def finalize(args):
    keys = common.KEYS
    plus7 = [raw(args.plus7_runs, (f's{seed}', f'union9_plus7_s{seed}'), seed) for seed in common.SEEDS]
    timse = [raw(args.timse_runs, (f's{seed}', f'timse_s{seed}'), seed) for seed in common.SEEDS]
    for left, right in zip(plus7, timse):
        if not left[keys].equals(right[keys]):
            raise ValueError('Component prediction keys differ')
    if any(not plus7[0][keys].equals(frame[keys]) for frame in plus7[1:]):
        raise ValueError('Seed prediction keys differ')
    forecasts = [(WEIGHT * a.pred_kms.to_numpy(np.float64) + (1 - WEIGHT) * b.pred_kms.to_numpy(np.float64))
                 .reshape(-1, common.H) for a, b in zip(plus7, timse)]
    logs = common.run_logs(args, EXPERIMENT)
    ridge = UNION.ridge_forecast(args.data, args.weights, True, logs)
    final = UNION.postprocess(forecasts, ridge)
    outputs = []
    for seed, frame, matrix in zip(common.SEEDS, plus7, final):
        output = frame[keys].assign(pred_kms=matrix.ravel())
        scores = common.score(output, args.observations)
        common.write_results(args.out / 'results', f's{seed}', output, scores)
        logs(f's{seed}', 'scored', total_s=scores['competition']['total']['s'])
        outputs.append(output)
    mean = common.mean3(outputs)
    scores = common.score(mean, args.observations)
    common.write_results(args.out / 'results', 'mean3', mean, scores)
    logs('run', 'complete', weight_plus7=WEIGHT, weight_timse=1 - WEIGHT, total_s=scores['competition']['total']['s'])


def main():
    parser = argparse.ArgumentParser(add_help=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--component', choices=('plus7', 'timse'))
    mode.add_argument('--finalize', action='store_true')
    args, remaining = parser.parse_known_args()
    if args.component:
        fit(args.component, remaining)
        return
    settings = argparse.ArgumentParser()
    settings.add_argument('--experiment', choices=[EXPERIMENT], default=EXPERIMENT)
    settings.add_argument('--device', choices=['cpu'], default='cpu')
    settings.add_argument('--threads', type=int, default=1)
    for key in ('plus7-runs', 'timse-runs', 'data', 'weights', 'out', 'observations'):
        settings.add_argument(f'--{key}', type=Path, required=True)
    finalize(settings.parse_args(remaining))


if __name__ == '__main__':
    main()
