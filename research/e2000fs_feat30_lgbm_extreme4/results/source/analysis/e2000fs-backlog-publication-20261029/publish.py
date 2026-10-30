"""Publish finished e2000fs runs that had official-grid forecasts but no leaderboard result.

User request 2026-10-29: upload every finished experiment missing from the leaderboard.
Forecasts are the runs' original frozen predictions; nothing is retrained or retuned here.
Each step (build, dryrun, push, github, submit, verify) is one durable stage, as in the
earlier e2000fs publishers.
"""
import argparse
import importlib.util
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = HERE / 'worktree'
spec = importlib.util.spec_from_file_location('backlog_publication_base', ROOT / 'analysis/forecast_strategies_kumo/publish.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
read, write, sha = P.read_json, P.write_json, P.sha

DIST = ROOT / 'analysis/e2000fs_feat30_lgbm_distribution/runs/full_20261028_v1'
NEXT = ROOT / 'analysis/e2000fs_feat30_next/runs/full_20261028_v1'
HIST = ROOT / 'analysis/e2000fs_feat30_extreme_histprior/runs/full_20261028_v2'
COMMON = ('Inherited Top30 features were selected using previously scored official periods; post-hoc official-exposed '
          'research, not a fresh holdout. Targets mature 72h before cutoff; F1 cutoff 2026-02-26, F2 cutoff 2026-05-29. '
          'ACE lag 0, causal forward fill. Published script replays hash-pinned forecasts; training source archived. '
          'Uploaded on 2026-10-29 at user request for complete leaderboard reporting; no settings changed after the run. '
          'Inference timing: saved CPU models, two threads, median of 3 batches of 32 real query origins; '
          'excludes distribution/histogram decoding, feature assembly, loading and I/O.')
DIST_SOURCES = [ROOT / 'analysis/e2000fs_feat30_lgbm_distribution' / n for n in ('run.py', 'methods.py', 'launch.py')]


def dist_lane(method, experiment, text):
    run = DIST / ('controls' if method == 'extreme4' else method)
    pattern = ('{stage}/extreme4/h*/attempt0000/model.txt' if method == 'extreme4'
               else '{stage}/%s/h*/attempt0000/[mr]*.txt' % method)
    return dict(experiment=experiment, run=run, manifest=DIST / 'manifest.json',
                predictions=run / f'{method}_predictions.parquet',
                result=lambda: read(run / 'results.json')[method], models=pattern, kind='lgb',
                sources=DIST_SOURCES, text=text, method=method)


LANES = {
    'histprior': dict(
        experiment='e2000fs_feat30_extreme_histprior', run=HIST / 'official', manifest=HIST / 'manifest.json',
        predictions=HIST / 'official/predictions.parquet', result=lambda: read(HIST / 'official/official_score.json'),
        models='models/{stage}/b*/h*.ubj', kind='xgb', method='hist_weighted_raw',
        sources=[ROOT / 'analysis/e2000fs_feat30_extreme_histprior' / n for n in ('run.py', 'hist.py', 'run_all.sh', 'PREREGISTRATION.md')]
                + [HIST / 'official_script.py', HIST / 'official_score_script.py'],
        text=('HL-Gauss 17-bin histogram classification XGBoost per 24h block (300 rounds, seed 42) with pointwise factor-4 '
              'fast-wind (>=550) sample weights; point forecast is the raw histogram mean (arm hist_weighted_raw, no '
              'target-shift posterior correction). Transferred on the internal climatology-reweighted gate; the '
              'pre-registered corrected arm failed internally.')),
    'smooth_error_tail': dict(
        experiment='e2000fs_feat30_smooth_error_tail', run=NEXT, manifest=NEXT / 'manifest.json',
        predictions=NEXT / 'predictions.parquet', result=lambda: read(NEXT / 'results.json'),
        models='{stage}/h*/model.txt', kind='lgb', method='smooth_error_tail',
        sources=[ROOT / 'analysis/e2000fs_feat30_next' / n for n in ('run.py', 'methods.py', 'README.md')],
        text=('Direct LightGBM per horizon, frozen HPO, custom smooth error-tail loss 0.5e^2 + 1.5*50^2*softplus((e^2-100^2)/50^2), '
              'adapted from Ramaswamy & Prashanth 2025 (arXiv 2505.17777), not the exact shortfall algorithm. '
              'Internal low/high acceptance FAILED; not promoted.')),
    'lognormal_nll': dist_lane('lognormal_nll', 'e2000fs_feat30_lognormal_nll',
        'Direct LightGBM lognormal (mu, rho) per horizon trained by NLL; point forecast is the lognormal predictive mean. Frozen HPO; no official-based selection.'),
    'lognormal_crps': dist_lane('lognormal_crps', 'e2000fs_feat30_lognormal_crps',
        'Direct LightGBM lognormal (mu, rho) per horizon trained by closed-form CRPS; point forecast is the predictive mean. Frozen HPO; no official-based selection.'),
    'smooth_wcrps': dist_lane('smooth_wcrps', 'e2000fs_feat30_smooth_wcrps',
        'Direct LightGBM lognormal per horizon trained by smooth threshold-weighted CRPS (center 550, width 100 km/s, floor 1, extra weight 1); point forecast is the predictive mean.'),
    'tailcal_crps': dist_lane('tailcal_crps', 'e2000fs_feat30_tailcal_crps',
        'Direct LightGBM lognormal per horizon trained by CRPS plus smooth batch CDF tail calibration (strength 1; adaptation, not exact TMCB); point forecast is the predictive mean.'),
    'lgbm_extreme4': dist_lane('extreme4', 'e2000fs_feat30_lgbm_extreme4',
        'Control: direct LightGBM L2 per horizon, frozen HPO, factor-4 block-event weighting (24h block containing >=550 km/s). Distinct from the XGBoost e2000fs_feat30_extreme.'),
}


def query_rows():
    """32 fixed real official-grid origins in the parent's 237-column block design."""
    hpo = P_load('backlog_hpo', ROOT / 'analysis/e2000fs_dirmo_24_feat30_lgbm_HPO/run.py')
    names = read(hpo.CACHE / 'names.json')
    selected = read(ROOT / 'analysis/e2000fs_dirmo_24_feat30/logs/full_01/manifest.json')['selected']
    bundle = dict(np.load(hpo.CACHE / 'official.npz'))
    idx = np.linspace(0, len(bundle['C']) - 1, 32).astype(int)
    return [hpo.base.design(bundle, names, selected, b)[idx] for b in (0, 24, 48)]


def P_load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(s)
    sys.modules[name] = module
    s.loader.exec_module(module)
    return module


def horizon_key(path):
    return [int(n) for n in re.findall(r'\d+', str(path.relative_to(path.parents[3])))]


def measure(lane, query):
    """Seconds per origin and model hashes over the f1 and final stage models."""
    timing, hashes = {}, {}
    for stage, origins in (('f1', 2137), ('final', 2833)):
        files = sorted(lane['run'].glob(lane['models'].format(stage=stage)), key=lambda p: horizon_key(p))
        assert files, (lane['experiment'], stage)
        hashes.update({str(p.relative_to(ROOT)): sha(p) for p in files})
        if lane['kind'] == 'xgb':
            models = []
            for p in files:
                booster = xgb.Booster(model_file=str(p))
                booster.set_param({'nthread': 2})
                models.append((int(p.parent.name[1:]), booster))
            assert len(models) == 72
            data = [xgb.DMatrix(q, nthread=2) for q in query]
            run = lambda: [m.predict(data[b]) for b, m in models]
        else:
            models = [lgb.Booster(model_file=str(p)) for p in files]
            per_h = len(models) // 72
            assert per_h * 72 == len(models), (lane['experiment'], stage, len(models))
            run = lambda: [m.predict(query[(i // per_h) // 24], num_threads=2) for i, m in enumerate(models)]
        run()
        elapsed = []
        for _ in range(3):
            start = time.perf_counter()
            run()
            elapsed.append(time.perf_counter() - start)
        timing[stage] = dict(seconds=elapsed, seconds_per_origin=float(np.median(elapsed) / 32), origins=origins,
                             model_files=len(files))
    seconds = sum(v['seconds_per_origin'] * v['origins'] for v in timing.values()) / 4970
    return seconds, timing, hashes


def build(names=None):
    built = read(HERE / 'build.json', {})
    query = query_rows()
    for name in names or LANES:
        lane = LANES[name]
        if name in built:
            Publisher(HERE, HERE, WORK).check_item(built[name])
            continue
        experiment = lane['experiment']
        frame = pd.read_parquet(lane['predictions'])
        origins, values = P.validate_frame(frame)
        manifest = read(lane['manifest'])
        result = lane['result']()
        seconds, timing, model_hashes = measure(lane, query)
        relative = Path('releases') / experiment
        release = WORK / relative
        assert not release.exists(), release
        release.mkdir(parents=True)
        archived = {}
        for source in lane['sources'] + [Path(__file__)]:
            target = release / 'source' / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            archived[str(source.relative_to(ROOT))] = sha(source)
        np.savez_compressed(release / 'forecast.npz', origins_ns=origins.asi8, predictions=values)
        config = dict(
            data=dict(feature_set='e2000fs_dirmo_24_feat30 Top30 (21 constant + 24h x 9 varying per block)',
                      ace_lag_hours=0, prefill='causal forward fill; leading missing retained',
                      periods={p: dict(start=a, end=b, count=n) for p, (a, b, n) in P.GRID.items()},
                      asset=dict(filename='forecast.npz', sha256=sha(release / 'forecast.npz')),
                      archived_source_sha256=archived, model_sha256=model_hashes,
                      original_prediction_sha256=sha(lane['predictions']),
                      run_manifest_sha256=sha(lane['manifest'])),
            hyperparameters=dict(method=lane['method'], frozen_run_manifest=manifest,
                                 selection='frozen original run; no official-score tuning; user-requested reporting upload'))
        csha = P.config_sha(config)
        description = f'{experiment}. {lane["text"]} {COMMON}'
        meta = dict(team_member=P.MEMBER, experiment=experiment, description=description, no_future_leakage=True,
                    f1_training_data_end_utc='2026-02-26T00:00:00Z', config=config, inference_seconds_per_fold=seconds)
        for filename, value in [('config.json', config), ('meta.yaml', meta), ('run-manifest.json', manifest),
                                ('result.json', result), ('timing.json', timing)]:
            write(release / filename, value)
        (release / 'README.md').write_text(description + '\n')
        script_relative = Path('scripts') / P.MEMBER / experiment / (csha + '.py')
        script = WORK / script_relative
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(P.REPLAY.replace('Kumo', experiment).replace('__SHA__', repr(csha))
                          .replace('__RELEASE__', repr(str(relative))))
        replay = HERE / 'replay' / f'{name}.parquet'
        subprocess.run([str(P.PYTHON), str(script), '--output', str(replay)], check=True, cwd=WORK)
        assert pd.read_parquet(replay).equals(frame)
        paths = [script] + [p for p in release.rglob('*') if p.is_file()]
        built[name] = dict(experiment=experiment, config_sha256=csha, script=str(script_relative),
                           release=str(relative), predictions=str(replay), prediction_sha256=sha(replay),
                           meta=str(release / 'meta.yaml'), reported_mse=reported_mse(result),
                           release_sha256={str(p.relative_to(WORK)): sha(p) for p in paths})
        write(HERE / 'build.json', built)
        print(f'{name}: replay verified, {seconds:.3g} s/origin, built', flush=True)
    write(HERE / 'batch.json', list(built))


def reported_mse(result):
    """Total official-grid MSE as the run reported it."""
    if 'official_metric' in result:
        return result['official_metric']['mse']
    return result['combined']['mse']


class Publisher(P.Publisher):
    """Scores are compared with each run's own reported total MSE; the leaderboard scorer defines F1/F2/550."""

    def compare_scores(self, item, result):
        if result['config_sha256'] != item['config_sha256']:
            raise ValueError('client configuration hash differs')
        if not np.isclose(item['reported_mse'], result['mse'], rtol=1e-6):
            raise ValueError(f'{item["experiment"]}: leaderboard MSE {result["mse"]} differs from reported {item["reported_mse"]}')

    def git(self, *args, capture=False):
        if args and args[0] == 'commit':
            args = ('commit', '-m', 'E2000FS: publish finished feat30 follow-ups missing from the leaderboard')
        return super().git(*args, capture=capture)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step', choices=('build', 'dryrun', 'push', 'github', 'submit', 'verify'))
    parser.add_argument('--lane', choices=list(LANES))
    args = parser.parse_args()
    if args.step == 'build':
        build([args.lane] if args.lane else None)
    else:
        getattr(Publisher(HERE, HERE, WORK), args.step)()
