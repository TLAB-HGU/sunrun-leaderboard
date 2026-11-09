"""Publish the three validated lag-addition seeds; no ensemble or seed selection."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import fcntl
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
WORK = ROOT / 'publication'
MEMBER = 'seongeun'
SEEDS = (42, 43, 44)
ADDITIONS = ['v_lag24', 'v_lag48', 'v_lag72']
MODULES = ('experiment.py', 'core.py', 'baseline.py', 'competition.py', 'inventory.json')
CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}
SCOPES = ('total', 'f1', 'f2', 'f3')
METRICS = ('s', 'rmse', 'trmse', 'timse')
GLOBAL_LOCK = REPO / 'research/rerun_df_ffill_20261103/.publish.lock'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def utc_now():
    return datetime.now(UTC).isoformat()


def assigned_recipes():
    recipes = read(ROOT / 'assignment.global.json')['experiments']
    if len(recipes) != 3 or sorted(r['seed'] for r in recipes) != list(SEEDS):
        raise ValueError('Publication requires exactly the frozen three seeds')
    if len({r['experiment'] for r in recipes}) != 3:
        raise ValueError('Duplicate run experiment')
    for recipe in recipes:
        if recipe['config'].get('add_features') != ADDITIONS or recipe['config'].get('seed') != recipe['seed']:
            raise ValueError('Only the requested joint lag additions may be published')
        if not recipe.get('leaderboard_experiment'):
            raise ValueError('Freeze each leaderboard experiment name before registration/publication')
    return sorted(recipes, key=lambda r: r['seed'])


def load_core():
    """Load this campaign without contaminating other campaign module bindings."""
    spec = importlib.util.spec_from_file_location('lag_publication_baseline', ROOT / 'baseline.py')
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    previous = sys.modules.get('baseline')
    sys.modules['baseline'] = baseline
    try:
        spec = importlib.util.spec_from_file_location('lag_publication_core', ROOT / 'core.py')
        core = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
    finally:
        if previous is None:
            sys.modules.pop('baseline', None)
        else:
            sys.modules['baseline'] = previous
    return core


def expected_contract(recipe, core):
    expected = dict(core.normalize_config(recipe['config']), experiment=recipe['experiment'])
    names = core.feature_names(expected)
    expanded = {str(start): core.expanded_names(names, start, 24) for start in (0, 24, 48)}
    retained = core.baseline.PARENT_NAMES
    for kind, values in retained.items():
        if names[kind][:len(values)] != values:
            raise ValueError('Parent feature prefix was not retained')
    actual = set(sum(names.values(), []))
    if actual != set(sum(retained.values(), [])) | set(ADDITIONS):
        raise ValueError('Training inputs are not the exact parent/addition union')
    return expected, names, expanded


def validate_runs():
    recipes = assigned_recipes()
    runs = {}
    # Check all completion receipts before importing scientific modules or publishing anything.
    for recipe in recipes:
        run = ROOT / 'runs' / recipe['experiment']
        if not (run / 'train.exit').is_file() or (run / 'train.exit').read_text().strip() != '0':
            raise RuntimeError(f'{run.name}: successful training exit required')
        if not (run / 'training-validation.json').is_file():
            raise RuntimeError(f'{run.name}: training validation receipt required')
        runs[recipe['seed']] = run
    core = load_core()
    for recipe in recipes:
        run = runs[recipe['seed']]
        expected, names, expanded = expected_contract(recipe, core)
        for name in MODULES[1:]:
            if (run / name).read_bytes() != (ROOT / name).read_bytes():
                raise ValueError(f'Frozen scientific snapshot changed: {run.name}/{name}')
        if (run / 'experiment.py').read_bytes() != (run / 'runner_at_launch.py').read_bytes():
            raise ValueError('Single-experiment entrypoint changed after launch')
        if read(run / 'config.json') != expected or read(run / 'resolved_config.json') != expected:
            raise ValueError('Resolved launch configuration differs from frozen assignment')
        proof = read(run / 'training-validation.json')
        if proof.get('reference_only') or proof.get('fresh_training') is False:
            raise ValueError('Historical reference import cannot be published as a new fit')
        if proof.get('rows') != 454680:
            raise ValueError('Incomplete validated official prediction grid')
        for period, cutoff in CUTOFFS.items():
            report = read(run / f'{period}_metadata.json')
            if (report.get('smoke') or report['training_cutoff'] != cutoff
                    or report['maturity_hours'] != 144 or report['seed'] != recipe['seed']
                    or report['rounds'] != 900 or report['config'] != expected
                    or report['features'] != names or report['expanded_features'] != expanded
                    or report['feature_count'] != sum(map(len, names.values()))
                    or report['expanded_feature_count'] != len(expanded['0'])
                    or report['parameters']['device'] != 'cpu'):
                raise ValueError(f'{run.name}/{period}: frozen scientific contract mismatch')
            replay = proof.get('saved_model_replay', {}).get(period, {})
            if (replay.get('rounds') != 900 or replay.get('boosters') != 3
                    or not np.isfinite(replay.get('max_abs_error', np.nan))
                    or replay['max_abs_error'] > 1e-6):
                raise ValueError(f'{run.name}/{period}: saved-model replay proof required')
    return runs


def parent_identifier():
    provenance = read(ROOT / 'provenance.json')
    parent = provenance.get('parent')
    if not isinstance(parent, str) or not parent:
        raise ValueError('Frozen provenance.parent must identify the audited leaderboard parent')
    return parent


def seed_metadata(run, revision):
    config = read(run / 'config.json')
    recipe = next(r for r in assigned_recipes() if r['experiment'] == run.name)
    reports = [read(run / f'{period}_metadata.json') for period in CUTOFFS]
    first = reports[0]
    assignment = read(ROOT / 'assignment.global.json')
    parent = parent_identifier()
    return dict(team_member=MEMBER, experiment=recipe['leaderboard_experiment'],
        description=f'Fresh joint addition of v_lag24, v_lag48 and v_lag72 to {parent}; seed {config["seed"]}. '
                    'Parent features and scientific settings retained. All seeds 42/43/44 published; no seed selection or ensemble. '
                    'Exploratory user-selected combination after individual seed42 official-period ablations; '
                    'no independent generalization or additive-gain claim. Official F1/F2/F3, 144h maturity, '
                    'training-only calibration. Published on explicit user request regardless of improvement gate.',
        no_future_leakage=True,
        inference_seconds_per_fold=sum(r['inference_seconds_per_fold'] * r['query_origins'] for r in reports)
                                   / sum(r['query_origins'] for r in reports),
        architecture_url=f'https://github.com/TLAB-HGU/sunrun-leaderboard/blob/{revision}/research/{ROOT.name}/published/core.py',
        config={'data': {'campaign': ROOT.name, 'parent': parent, 'source': assignment['data'],
                         'maturity_hours': 144, 'training_cutoffs': CUTOFFS,
                         'observation_revision': assignment['observation_revision']},
                'hyperparameters': {'run_config': config, 'resolved_config': read(run / 'resolved_config.json'),
                                    'parameters': first['parameters'], 'rounds': first['rounds'],
                                    'features': first['features'], 'feature_count': first['feature_count'],
                                    'expanded_features': first['expanded_features'],
                                    'expanded_feature_count': first['expanded_feature_count'],
                                    'add_features': ADDITIONS}},
        **{f'{p.lower()}_training_data_end_utc': cutoff + 'T00:00:00Z' for p, cutoff in CUTOFFS.items()})


def seed_replay(run):
    sources = {name: (run / name).read_text() for name in MODULES}
    return ('"""Reproduce one frozen CPU lag-addition model; pass --data --observations --out."""\n'
            'import os, runpy, sys, tempfile\nfrom pathlib import Path\n'
            f'SOURCES = {sources!r}\n'
            'if __name__ == "__main__":\n'
            '    os.environ["CUDA_VISIBLE_DEVICES"] = ""\n'
            '    work = Path(tempfile.mkdtemp(prefix="lag24_48_72_replay_"))\n'
            '    for name, body in SOURCES.items():\n'
            '        (work / name).write_text(body)\n'
            '    sys.path.insert(0, str(work))\n'
            '    sys.argv[0] = str(work / "experiment.py")\n'
            '    runpy.run_path(str(work / "experiment.py"), run_name="__main__")\n').encode()


def existing_result(records, config, experiment):
    matches = [r for r in records if r['team_member'] == MEMBER and r['experiment'] == experiment]
    if not matches:
        return None
    if len(matches) != 1 or matches[0]['config'] != config:
        raise RuntimeError('Existing experiment has conflicting config/multiple rows; refuse overwrite')
    return matches[0]


def setup_client():
    client = WORK / 'submit-client'
    client.mkdir(parents=True, exist_ok=True)
    for name in ('submit.py', 'board.py', 'architecture_code.py', 'metric_assets.py', 'lbstatus.py'):
        shutil.copy2(REPO / 'leaderboard' / name, client / name)
    shutil.copytree(REPO / 'leaderboard/scorer', client / 'scorer', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    sys.path.insert(0, str(client))
    import submit
    def snapshot_results(api, repo, revision=None):
        revision = revision or api.repo_info(repo, repo_type='dataset').sha
        files = [p for p in api.list_repo_files(repo, repo_type='dataset', revision=revision) if p.startswith('results/')]
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(lambda p: read(submit._download(repo, p, revision=revision)), files))
    submit.all_results = snapshot_results
    return submit


def source_publisher():
    spec = importlib.util.spec_from_file_location('lag_historical_source_publisher',
        REPO / 'research/offset_roll_ablation2_20261105/publish_lb.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.publish_sources


def require_score_parity(stored, expected):
    for scope in SCOPES:
        if stored[scope].get('n_replaced', 0) or stored[scope].get('replacement_fraction', 0):
            raise ValueError('Official scorer replaced forecasts')
        for metric in METRICS:
            if not np.isfinite(stored[scope][metric]) or abs(stored[scope][metric] - expected[scope][metric]) > 1e-6:
                raise ValueError(f'Official score mismatch: {scope}/{metric}')


def register(status):
    parent = parent_identifier()
    records = []
    for recipe in assigned_recipes():
        name = recipe['leaderboard_experiment']
        command = [sys.executable, str(REPO / 'leaderboard/lbstatus.py'), 'set', name,
                   '--member', MEMBER, '--status', status, '--desc',
                   f'{parent} + v_lag24/v_lag48/v_lag72; fresh seed{recipe["seed"]}; exploratory user-selected joint addition; all3seeds publish']
        result = subprocess.run(command, capture_output=True, text=True)
        records.append(dict(experiment=name, status=status, exit_code=result.returncode,
                            stdout=result.stdout, stderr=result.stderr))
        write(WORK / f'registration-{status}.json', records)
        if result.returncode:
            raise RuntimeError(f'Leaderboard status registration failed for {name}: {result.stderr[-1000:]}')
    return records


def publish_result(submit, runs):
    import architecture_code
    from huggingface_hub import HfApi
    api = HfApi()
    assignment = read(ROOT / 'assignment.global.json')
    observed = pd.read_parquet(assignment['observations'])
    revision = api.repo_info(submit.DATASET, repo_type='dataset').sha
    official = pd.read_parquet(submit._download(submit.DATASET, 'competition-observations.parquet', revision=revision))
    pd.testing.assert_frame_equal(observed, official, check_exact=True)
    context = submit.prepare_context(observed)
    prefix = f'research/{ROOT.name}/published'
    files = {f'{prefix}/{name}': (runs[42] / name).read_bytes() for name in MODULES[1:]}
    for name in ('strategy.md', 'provenance.json'):
        files[f'{prefix}/{name}'] = (ROOT / name).read_bytes()
    scores = {}
    for seed, run in runs.items():
        files[f'{prefix}/seed{seed}/experiment.py'] = (run / 'experiment.py').read_bytes()
        meta = seed_metadata(run, '0' * 40)
        replay = seed_replay(run)
        replay_path = submit.replay_script_path(MEMBER, meta['experiment'], meta['config'])
        compile(replay, replay_path, 'exec')
        files[replay_path] = replay
        local = submit.HERE / replay_path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(replay)
        score = submit.score_competition(pd.read_parquet(run / 'predictions.parquet'), context)
        require_score_parity(read(run / 'local_score.json')['competition'], score)
        dry = submit.submit(str(run / 'predictions.parquet'), meta, dry_run=True)
        require_score_parity(dry['competition'], score)
        scores[seed] = score
        write(WORK / f'{meta["experiment"]}.dry-run.json', dry)
    # All three full runs/dry scores are validated before source or result publication.
    revision_file = WORK / 'source-revision.txt'
    if revision_file.exists():
        source_revision = revision_file.read_text().strip()
    else:
        source_revision = source_publisher()(files)
        revision_file.write_text(source_revision + '\n')
    receipts = []
    for seed, run in runs.items():
        meta = seed_metadata(run, source_revision)
        experiment = meta['experiment']
        (WORK / f'{experiment}.meta.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
        result = existing_result(submit.all_results(api, submit.DATASET), meta['config'], experiment)
        if result is None:
            result = submit.submit(str(run / 'predictions.parquet'), meta)
        stored = existing_result(submit.all_results(api, submit.DATASET), meta['config'], experiment)
        if not stored or stored['submitted_utc'] != result['submitted_utc']:
            raise ValueError('Submission could not be independently read back')
        require_score_parity(stored['competition'], scores[seed])
        if stored['architecture_url'] != meta['architecture_url']:
            raise ValueError('Pinned architecture URL differs from official record')
        entry = architecture_code.entry(MEMBER, experiment, stored['config_sha256'], stored['architecture_url'])
        page, _, _ = architecture_code.space_files(submit.SPACE)
        if experiment not in (page or '') or stored['architecture_url'] not in page:
            submit.publish(api, [], submit.SPACE, architecture={'models': [entry]}, dataset=submit.DATASET)
            page, _, _ = architecture_code.space_files(submit.SPACE)
        if experiment not in (page or '') or stored['architecture_url'] not in page or not architecture_code.reachable(stored['architecture_url']):
            raise ValueError('Visible Space row/pinned architecture verification failed')
        receipt = run / 'submission.json'
        if receipt.exists() and not (run / 'submission.local.json').exists():
            shutil.copy2(receipt, run / 'submission.local.json')
        write(receipt, dict(stored, published=True, local_only=False))
        complete = read(run / 'complete.json') if (run / 'complete.json').exists() else {}
        write(run / 'complete.json', dict(complete, publish='complete', publication_verified=True))
        receipts.append(dict(experiment=experiment, seed=seed, submitted_utc=stored['submitted_utc'],
                             total_s=stored['competition']['total']['s'], remote_score_matches=True,
                             page_row_present=True, architecture_reachable=True))
        write(WORK / 'receipts.json', receipts)
        print(json.dumps({'stage': 'published', **receipts[-1]}), flush=True)
    return receipts



def apply_remote_receipts(root, payload):
    """Validate all owned runs before changing only their publication receipts."""
    import fcntl
    import json
    import math
    from pathlib import Path
    import shutil

    root = Path(root).resolve()
    def read_json(path):
        return json.loads(path.read_text())
    def atomic_json(path, value):
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    with (root / '.publication-receipts.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assignment = read_json(root / 'assignment.json')
        recipes = {row['experiment']: row for row in assignment['experiments']}
        if assignment.get('host') != 'golab2' or len(recipes) != 3 or set(recipes) != {item['run'] for item in payload} or len(payload) != 3:
            raise ValueError('Remote publication scope differs from exact three assigned runs')
        prepared = []
        for item in payload:
            name = item['run']
            run = root / 'runs' / name
            if Path(name).name != name or run.resolve().parent != (root / 'runs').resolve():
                raise ValueError('Remote run path escaped the owned campaign')
            recipe = recipes[name]
            record, expected = item['record'], item['config']
            launch = read_json(run / 'launch.json')
            proof = read_json(run / 'training-validation.json')
            if (launch.get('campaign_root') != str(root) or launch.get('experiment') != name
                    or launch.get('training_host') != 'golab2'
                    or (run / 'train.exit').read_text().strip() != '0'
                    or proof.get('rows') != 454680 or proof.get('reference_only')):
                raise ValueError('Remote fresh-run ownership/completion proof mismatch')
            if (read_json(run / 'config.json') != expected or read_json(run / 'resolved_config.json') != expected
                    or recipe['seed'] != expected['seed'] or recipe['config']['add_features'] != expected['add_features']
                    or record.get('experiment') != recipe['leaderboard_experiment']
                    or record.get('team_member') != 'seongeun'
                    or record.get('config', {}).get('hyperparameters', {}).get('run_config') != expected
                    or record.get('published') is not True or record.get('local_only') is not False):
                raise ValueError('Remote configuration or official experiment identity mismatch')
            local_score = read_json(run / 'local_score.json')['competition']
            for scope in ('total', 'f1', 'f2', 'f3'):
                if record['competition'][scope].get('n_replaced', 0):
                    raise ValueError('Official forecasts were replaced')
                for metric in ('s', 'rmse', 'trmse', 'timse'):
                    value = record['competition'][scope][metric]
                    if not math.isfinite(value) or abs(value - local_score[scope][metric]) > 1e-6:
                        raise ValueError('Remote/local official metric mismatch')
            for period in ('F1', 'F2', 'F3'):
                replay = proof.get('saved_model_replay', {}).get(period, {})
                if (replay.get('rounds') != 900 or replay.get('boosters') != 3
                        or not math.isfinite(replay.get('max_abs_error', math.inf))
                        or replay['max_abs_error'] > 1e-6):
                    raise ValueError('Remote saved-model replay proof mismatch')
            previous = read_json(run / 'submission.json')
            if previous.get('published') is True and any(previous.get(k) != record.get(k)
                    for k in ('experiment', 'team_member', 'config', 'submitted_utc', 'architecture_url')):
                raise ValueError('Refuse replacing a different already-published remote result')
            complete = read_json(run / 'complete.json')
            prepared.append((run, record, previous, dict(complete, publish='complete',
                             publication_verified=True, submitted_utc=record['submitted_utc'])))
        for run, record, previous, complete in prepared:
            backup = run / 'submission.local.json'
            if previous.get('published') is not True and not backup.exists():
                shutil.copy2(run / 'submission.json', backup)
            atomic_json(run / 'submission.json', record)
            atomic_json(run / 'complete.json', complete)
        verified = []
        for run, record, _, complete in prepared:
            if read_json(run / 'submission.json') != record or read_json(run / 'complete.json') != complete:
                raise ValueError('Remote publication receipt readback failed')
            verified.append(dict(run=run.name, experiment=record['experiment'],
                                 submitted_utc=record['submitted_utc'], published=True))
        return dict(verified=True, records=verified)


def sync_remote_receipts(runs, receipts):
    if len(receipts) != 3 or {row['seed'] for row in receipts} != set(SEEDS):
        raise ValueError('All three official publication verifications are required before remote sync')
    payload = []
    for row in receipts:
        if not all(row.get(key) is True for key in ('remote_score_matches', 'page_row_present', 'architecture_reachable')):
            raise ValueError('Official publication has not passed all verification gates')
        run = runs[row['seed']]
        record = read(run / 'submission.json')
        if record['experiment'] != row['experiment'] or record['submitted_utc'] != row['submitted_utc']:
            raise ValueError('Local official receipt differs from verified publication')
        payload.append(dict(run=run.name, config=read(run / 'config.json'), record=record))
    spec = importlib.util.spec_from_file_location('lag_publication_transport', ROOT / 'transport_support.py')
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    code = inspect.getsource(apply_remote_receipts) + '\nimport json\nprint(json.dumps(apply_remote_receipts(' + repr(transport.REMOTE) + ', ' + repr(payload) + ')))\n'
    result = json.loads(transport.remote(code, timeout=60).stdout)
    expected = {row['experiment']: row['submitted_utc'] for row in receipts}
    if not result.get('verified') or {row['experiment']: row['submitted_utc'] for row in result['records']} != expected:
        raise ValueError('Remote publication sync verification differs from official receipts')
    write(WORK / 'remote-receipts-verified.json', dict(result, verified_utc=utc_now()))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--register-only', action='store_true')
    parser.add_argument('--status', choices=('대기', '진행'), default='대기')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / '.publication.lock').open('a') as campaign_lock:
        fcntl.flock(campaign_lock, fcntl.LOCK_EX)
        if args.register_only:
            print(json.dumps({'registrations': register(args.status)}), flush=True)
            return
        runs = validate_runs()
        if not args.apply:
            for seed, run in runs.items():
                write(WORK / f'seed{seed}.metadata.json', seed_metadata(run, '0' * 40))
                compile(seed_replay(run), f'seed{seed}.replay.py', 'exec')
            print(json.dumps({'validated_seeds': list(runs), 'published': False}), flush=True)
            return
        submit = setup_client()
        with GLOBAL_LOCK.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            receipts = publish_result(submit, runs)
        remote_receipts = sync_remote_receipts(runs, receipts)
        evidence = dict(complete=True, status='complete', seeds=list(SEEDS), records=receipts,
                        visible_rows=3, all_scores_match=True, no_ensemble=True, verified_utc=utc_now(),
                        remote_receipts=remote_receipts,
                        classification='exploratory user-selected joint lag addition', publication=str(WORK))
        write(WORK / 'verified.json', evidence)
        write(ROOT / 'campaign-complete.json', evidence)
        print(json.dumps({'stage': 'complete', 'published_seed_results': 3}), flush=True)


if __name__ == '__main__':
    main()
