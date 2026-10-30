"""Publish all three rejected transfer candidates after explicit reporting evaluation."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = HERE.parents[2]
WORK = HERE / 'worktree'
EVAL = HERE / 'evaluation_full'
ORIGINAL = STUDY / 'runs/full_20261028_v1'
LANES = dict(hyper_gain='e2000fs_feat30_hyper_gain',
             gaussian='e2000fs_feat30_gaussian_pilot',
             mc_calibration='e2000fs_feat30_mc_calibration')
sys.path.insert(0, str(WORK))
spec = importlib.util.spec_from_file_location('transfer2_publication_base', ROOT / 'analysis/forecast_strategies_kumo/publish.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
read, write, sha = P.read_json, P.write_json, P.sha


class Publisher(P.Publisher):
    def git(self, *args, capture=False):
        if args and args[0] == 'commit':
            args = ('commit', '-m', 'E2000FS: publish three transfer-rejected candidates for complete reporting')
        return super().git(*args, capture=capture)


def build(lanes=None):
    built = read(HERE/'build.json', {})
    for lane in lanes or LANES:
        experiment = LANES[lane]
        if lane in built:
            Publisher(HERE,HERE,WORK).check_item(built[lane])
            continue
        directory = EVAL / lane
        assert read(directory / 'status.json')['status'] in ('completed', 'completed_evaluated')
        original_manifest = read(ORIGINAL / lane / 'manifest.json')
        original_result = read(ORIGINAL / lane / 'results.json')
        assert original_result['development_passed'] is False
        manifest = read(directory / 'manifest.json')
        for field in ('source_sha256', 'input_sha256'):
            for path, expected in manifest[field].items():
                assert sha(path) == expected, path
        for path, expected in read(directory / 'completion.json')['sha256'].items():
            assert sha(path) == expected, path
        for field in ('source_sha256', 'input_sha256'):
            for path, expected in original_manifest[field].items():
                assert sha(path) == expected, path
        result = read(directory / 'results.json')
        frame = pd.read_parquet(directory / 'predictions.parquet')
        origins, predictions = P.validate_frame(frame)
        stage_predictions = []
        for stage in ('f1', 'final'):
            with np.load(directory / f'{stage}_predictions.npz') as values:
                stage_predictions.append(values['pred'])
        np.testing.assert_array_equal(predictions, np.vstack(stage_predictions))
        relative = Path('releases') / experiment
        release = WORK / relative
        release.mkdir(parents=True, exist_ok=True)
        archived = {}
        sources = [STUDY/'run.py', STUDY/'methods.py', STUDY/'PLAN.md',
                   HERE/'evaluate.py', Path(__file__), ROOT/'analysis/e2000fs_feat30_lgbm_distribution/run.py',
                   ROOT/'analysis/e2000fs_dirmo_24_feat30_lgbm_HPO/run.py',
                   ROOT/'analysis/e2000fs_dirmo_24_feat30_lgbm/run.py',
                   ROOT/'analysis/e2000fs_feat30_followups/evaluation.py']
        for source in sources:
            target = release/'source'/source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source,target)
            archived[str(source.relative_to(ROOT))] = sha(source)
        for loss in directory.rglob('loss.csv'):
            target = release/'loss_history'/loss.relative_to(directory)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(loss,target)
        model_hashes = {str(path.relative_to(ROOT)):sha(path) for path in directory.rglob('model.txt')}
        assert model_hashes
        np.savez_compressed(release/'forecast.npz',origins_ns=origins.asi8,predictions=predictions)
        config = dict(data=dict(feature_names=original_manifest['selected_groups'],ace_lag_hours=0,
            prefill='causal forward fill; leading missing retained',
            periods={p:dict(start=a,end=b,count=n) for p,(a,b,n) in P.GRID.items()},
            asset=dict(filename='forecast.npz',sha256=sha(release/'forecast.npz')),
            archived_source_sha256=archived,model_sha256=model_hashes,
            original_manifest_sha256=sha(ORIGINAL/lane/'manifest.json'),
            evaluation_manifest_sha256=sha(directory/'manifest.json'),
            input_sha256=original_manifest['input_sha256']),
            hyperparameters=dict(lane=lane,parameters=original_manifest['params'],
                method=original_manifest[lane],development_passed=False,
                selection='fixed original settings; no actual-score tuning; user-requested reporting evaluation'))
        csha = P.config_sha(config)
        description = (f'{experiment}. '+original_manifest[lane]+'. '
            'DEVELOPMENT TRANSFER GATE FAILED; NOT PROMOTED. Actual F1/F2 evaluation was explicitly requested afterward for complete leaderboard reporting. '
            'No model settings changed after internal rejection. Inherited Top30 features used previously scored official periods; post-hoc official-exposed research, not a fresh holdout. '
            'Only mature training targets: F1 before2026-02-26, F2 before2026-05-29. '
            'Published script replays hash-pinned forecasts; training/evaluation sources archived. '
            'Inference timing is decoded model prediction on both full query grids; excludes loading, training, feature assembly and I/O.')
        timing = read(directory/'timing.json')
        inference = timing['inference_seconds_per_origin']
        assert inference > 0 and np.isfinite(inference)
        meta = dict(team_member='seongeun',experiment=experiment,description=description,
                    no_future_leakage=True,f1_training_data_end_utc='2026-02-26T00:00:00Z',
                    config=config,inference_seconds_per_fold=inference)
        for filename,value in [('config.json',config),('meta.yaml',meta),('run-manifest.json',manifest),
                               ('result.json',result),('original-validation.json',original_result),
                               ('original-manifest.json',original_manifest),('timing.json',timing)]:
            write(release/filename,value)
        (release/'README.md').write_text(description+'\n')
        script_relative = Path('scripts/seongeun')/experiment/(csha+'.py')
        script = WORK/script_relative
        script.parent.mkdir(parents=True,exist_ok=True)
        script.write_text(P.REPLAY.replace('Kumo',experiment).replace('__SHA__',repr(csha)).replace('__RELEASE__',repr(str(relative))))
        replay = HERE/'replay'/f'{lane}.parquet'
        subprocess.run([str(P.PYTHON),str(script),'--output',str(replay)],check=True,cwd=WORK)
        assert pd.read_parquet(replay).equals(frame)
        score = result['combined']['official']
        paths = [script]+[p for p in release.rglob('*') if p.is_file()]
        built[lane] = dict(experiment=experiment,config_sha256=csha,script=str(script_relative),
            release=str(relative),predictions=str(replay),prediction_sha256=sha(replay),meta=str(release/'meta.yaml'),
            scores={k:score[v] for k,v in P.METRICS.items()},
            release_sha256={str(p.relative_to(WORK)):sha(p) for p in paths})
        print(lane,'full-grid replay verified; built',flush=True)
    write(HERE/'build.json',built)
    write(HERE/'batch.json',list(built))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=('build','dryrun','push','github','submit','verify'))
    parser.add_argument('--lane', choices=list(LANES))
    args = parser.parse_args()
    if args.step == 'build':
        build([args.lane] if args.lane else None)
    else:
        getattr(Publisher(HERE,HERE,WORK),args.step)()
