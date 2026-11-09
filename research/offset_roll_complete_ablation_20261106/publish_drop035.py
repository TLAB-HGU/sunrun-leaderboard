"""Publish the explicitly requested t035 equatorial-area removal, all three seeds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
WORK = ROOT / 'lb_publish_drop035'
CLIENT = WORK / 'submit-client'
MEMBER = 'seongeun'
PARENT = 'e2000fs_calib_offset_roll'
SEEDS = (42, 43, 44)
MODULES = ('experiment.py', 'core.py', 'baseline.py', 'competition.py')
CUTOFF = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}


def setup_client():
    CLIENT.mkdir(parents=True, exist_ok=True)
    for name in ('submit.py', 'board.py', 'architecture_code.py', 'metric_assets.py', 'lbstatus.py'):
        shutil.copy2(REPO / 'leaderboard' / name, CLIENT / name)
    shutil.copytree(REPO / 'leaderboard/scorer', CLIENT / 'scorer', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    sys.path.insert(0, str(CLIENT))
    import submit
    return submit


def metadata(run, revision):
    seed = json.loads((run / 'config.json').read_text())['seed']
    reports = [json.loads((run / f'{p}_metadata.json').read_text()) for p in CUTOFF]
    assert (run / 'train.exit').read_text().strip() == '0'
    assert (run / 'training-validation.json').is_file()
    assert (run / 'experiment.py').read_bytes() == (run / 'runner_at_launch.py').read_bytes()
    for period, report in zip(CUTOFF, reports):
        assert not report['smoke'] and report['training_cutoff'] == CUTOFF[period]
        assert report['maturity_hours'] == 144 and report['feature_count'] == 29
        assert report['config']['exclude_features'] == ['ch_ch_t035_equatorial_area']
    config = {'data': {'campaign': ROOT.name, 'parent': PARENT,
                       'source': 'rerun_df_ffill_20261103/input/df_ffill_trainy.parquet',
                       'maturity_hours': 144, 'training_cutoffs': CUTOFF},
              'hyperparameters': {'run_config': json.loads((run / 'config.json').read_text()),
                                  'resolved_config': json.loads((run / 'resolved_config.json').read_text()),
                                  'parameters': reports[0]['parameters'], 'rounds': 900,
                                  'features': reports[0]['features'], 'feature_count': 29}}
    seconds = sum(r['inference_seconds_per_fold'] * r['query_origins'] for r in reports) / sum(r['query_origins'] for r in reports)
    return {'team_member': MEMBER, 'experiment': f'{PARENT}_drop_t035_equatorial_s{seed}',
            'description': f'Original A single-feature ablation: remove ch_ch_t035_equatorial_area only; seed {seed}. '
                           'All preregistered seeds 42/43/44 are published together on explicit user request 2026-11-07. '
                           '29 retained source features; unchanged tail weight 4, spectral term, frozen training-only offset, block24 and clipping. '
                           'Official F1/F2/F3 evaluation only; 144h target maturity/purge. No B1 blend. '
                           'Diagnostic publication; full campaign statistical contribution/pruning decision remains pending.',
            'no_future_leakage': True, 'config': config, 'inference_seconds_per_fold': seconds,
            'architecture_url': f'https://github.com/TLAB-HGU/sunrun-leaderboard/blob/{revision}/research/{ROOT.name}/published_drop035/core.py',
            **{f'{p.lower()}_training_data_end_utc': d + 'T00:00:00Z' for p, d in CUTOFF.items()}}


def replay(run):
    sources = {n: (run / n).read_text() for n in MODULES}
    return ('"""Exact launch-time replay. Pass --data, --out and --observations; CPU only."""\n'
            'import os, runpy, sys, tempfile\nfrom pathlib import Path\n'
            f'SOURCES = {sources!r}\n'
            'if __name__ == "__main__":\n'
            '    os.environ["CUDA_VISIBLE_DEVICES"] = ""\n'
            '    work = Path(tempfile.mkdtemp(prefix="drop035_replay_"))\n'
            '    for name, body in SOURCES.items():\n'
            '        (work / name).write_text(body)\n'
            '    sys.path.insert(0, str(work))\n'
            '    runpy.run_path(str(work / "experiment.py"), run_name="__main__")\n').encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    submit = setup_client()
    import architecture_code
    from huggingface_hub import HfApi
    spec = importlib.util.spec_from_file_location('historical_publisher', REPO / 'research/offset_roll_ablation2_20261105/publish_lb.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    runs = {seed: ROOT / 'runs' / f'drop_ch_ch_t035_equatorial_area_s{seed}' for seed in SEEDS}
    prefix = f'research/{ROOT.name}/published_drop035'
    files = {f'{prefix}/{name}': (runs[42] / name).read_bytes() for name in MODULES if name != 'experiment.py'}
    for name in MODULES[1:]:
        assert all((run / name).read_bytes() == (runs[42] / name).read_bytes() for run in runs.values())
    files[f'{prefix}/strategy.md'] = (ROOT / 'strategy.md').read_bytes()
    for seed, run in runs.items():
        files[f'{prefix}/seed{seed}/experiment.py'] = (run / 'experiment.py').read_bytes()
        meta = metadata(run, '0' * 40)
        path = submit.replay_script_path(MEMBER, meta['experiment'], meta['config'])
        files[path] = replay(run)
        local = CLIENT / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(files[path])
        compile(files[path], path, 'exec')
        dry = submit.submit(str(run / 'predictions.parquet'), meta, dry_run=True)
        expected = json.loads((run / 'local_score.json').read_text())['competition']['total']['s']
        assert abs(dry['competition']['total']['s'] - expected) <= 1e-6
        assert dry['competition']['total']['replacement_fraction'] == 0
        (WORK / f'dry-run-s{seed}.json').write_text(json.dumps(dry, indent=2) + '\n')
        print(json.dumps({'dry_run_seed': seed, 's': expected}), flush=True)
    if not args.apply:
        return
    revision_file = WORK / 'source-revision.txt'
    revision = revision_file.read_text().strip() if revision_file.exists() else old.publish_sources(files)
    revision_file.write_text(revision + '\n')
    print(json.dumps({'source_revision': revision, 'files': len(files)}), flush=True)
    api = HfApi()
    # Read the same immutable dataset snapshot concurrently; scoring is unchanged.
    def snapshot_results(api, repo, revision=None):
        revision = revision or api.repo_info(repo, repo_type='dataset').sha
        paths = [p for p in api.list_repo_files(repo, repo_type='dataset', revision=revision)
                 if p.startswith('results/')]
        def read(path):
            return json.loads(Path(submit._download(repo, path, revision=revision)).read_text())
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(read, paths))
    submit.all_results = snapshot_results
    receipts = []
    for seed, run in runs.items():
        meta = metadata(run, revision)
        (WORK / f'meta-s{seed}.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
        existing = [r for r in submit.all_results(api, submit.DATASET)
                    if r['team_member'] == MEMBER and r['experiment'] == meta['experiment']]
        if existing:
            assert len(existing) == 1 and existing[0]['config'] == meta['config']
            result = existing[0]
        else:
            result = submit.submit(str(run / 'predictions.parquet'), meta)
        stored = next(r for r in submit.all_results(api, submit.DATASET)
                      if r['team_member'] == MEMBER and r['experiment'] == meta['experiment']
                      and r['submitted_utc'] == result['submitted_utc'])
        expected = json.loads((run / 'local_score.json').read_text())['competition']['total']['s']
        assert abs(stored['competition']['total']['s'] - expected) <= 1e-6
        assert stored['architecture_url'] == meta['architecture_url']
        result = dict(stored, published=True, local_only=False)
        body = json.dumps(result, indent=2) + '\n'
        if not (run / 'submission.local.json').exists():
            shutil.copy2(run / 'submission.json', run / 'submission.local.json')
        (run / 'submission.json').write_text(body)
        if seed in (42, 44):
            old.remote(f'import pathlib,shutil\np=pathlib.Path("/home/yehwan/sunrun-offset-roll-complete-ablation-20261106/runs/{run.name}")\nif not (p/"submission.local.json").exists(): shutil.copy2(p/"submission.json",p/"submission.local.json")\n(p/"submission.json").write_text({body!r})', timeout=120)
        receipts.append({'seed': seed, 'experiment': meta['experiment'], 's': expected,
                         'submitted_utc': stored['submitted_utc'], 'architecture_url': meta['architecture_url']})
        (WORK / 'receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
        print(json.dumps({'published': meta['experiment'], 's': expected}), flush=True)
    page, _, _ = architecture_code.space_files(submit.SPACE)
    assert not architecture_code.script_rows(page or '')
    for receipt in receipts:
        assert receipt['experiment'] in page
    assert architecture_code.reachable(receipts[0]['architecture_url'])
    (WORK / 'verified.json').write_text(json.dumps({'remote_scores_match': True, 'visible_rows': 3,
        'missing_architecture_links': 0, 'architecture_reachable': True}, indent=2) + '\n')
    print('Verified all 3 stored results, page rows and architecture links.', flush=True)


if __name__ == '__main__':
    with (REPO / 'research/rerun_df_ffill_20261103/.publish.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
