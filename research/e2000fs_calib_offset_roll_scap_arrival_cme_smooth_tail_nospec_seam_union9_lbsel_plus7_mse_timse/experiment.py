"""Fit separate MSE and TIMSE GBMs on the plus7 design and combine forecasts."""
import argparse
import dataclasses
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROW = HERE.name
PARENT = ROW.removesuffix('_mse_timse')
loader = importlib.util.spec_from_file_location('plus7_parent', HERE.parent/PARENT/'experiment.py')
parent = importlib.util.module_from_spec(loader)
loader.loader.exec_module(parent)
gbm, common, pd, np, union = parent.gbm, parent.common, parent.pd, parent.np, parent.UNION
from _module.objectives import timse_coefficients

LOSSES = ('mse', 'timse')


def objective(target, context, loss):
    weights = (len(target)*np.column_stack([timse_coefficients(y) for y in target.T])
               if loss == 'timse' else np.ones_like(target))
    weights = weights.astype(np.float32)

    def gradients(prediction, matrix):
        residual = np.asarray(prediction).reshape(target.shape) - target
        return (weights*residual).astype(np.float32), weights
    return gradients


def fit(component, remaining):
    shared = dict(parent.SPEC.launch)
    for key in ('a_stage', 'continuation', 'source_ablation'):
        shared.pop(key, None)
    shared.update(parent_experiment=PARENT, tree_budget=780,
                  classification='separate pure objectives and fixed forecast average',
                  loss=component, component_weight=.5)
    spec = dataclasses.replace(parent.SPEC, experiment=ROW, objective=component,
                               component=component, fit_block=None, predict_block=None,
                               rounds=780, launch=shared,
                               make_objective=lambda target, context: objective(target, context, component))
    sys.argv[1:] = remaining
    original = gbm.score
    gbm.score = lambda *args: None
    try:
        gbm.main(spec, __doc__, HERE)
    finally:
        gbm.score = original


def finalize(args):
    component_frames = {}
    for loss in LOSSES:
        component_frames[loss] = [
            pd.read_parquet(args.campaign/'runs'/f'{loss}_s{seed}'/'results'/f's{seed}_predictions.parquet')
            for seed in common.SEEDS]
    frames = component_frames['mse']
    keys = common.KEYS
    for left, right in zip(frames, component_frames['timse']):
        if not left[keys].equals(right[keys]):
            raise ValueError('Component prediction keys differ')
    if any(not frames[0][keys].equals(frame[keys]) for frame in frames[1:]):
        raise ValueError('Seed prediction keys differ')
    forecasts = [(.5*a.pred_kms.to_numpy(np.float64)+.5*b.pred_kms.to_numpy(np.float64)).reshape(-1, common.H)
                 for a,b in zip(frames, component_frames['timse'])]
    logs = common.run_logs(args, ROW)
    ridge = union.ridge_forecast(args.data, args.weights, True, logs)
    final = union.postprocess(forecasts, ridge)
    outputs = []
    for seed, frame, matrix in zip(common.SEEDS, frames, final):
        output = frame[keys].assign(pred_kms=matrix.ravel())
        common.write_results(args.out/'results', f's{seed}', output, common.score(output,args.observations))
        outputs.append(output)
    mean = common.mean3(outputs)
    common.write_results(args.out/'results', 'mean3', mean, common.score(mean,args.observations))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--loss', choices=LOSSES)
    mode.add_argument('--finalize', action='store_true')
    args, remaining = parser.parse_known_args()
    if args.loss:
        fit(args.loss, remaining)
        return
    settings = argparse.ArgumentParser()
    for key in ('campaign','data','weights','out','observations'):
        settings.add_argument(f'--{key}', type=Path, required=True)
    finalize(settings.parse_args(remaining))


if __name__ == '__main__':
    main()
