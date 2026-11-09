"""Show 3-seed means on the leaderboard and hide single-seed rows.

For every seed group (prefix_s42/_s43/_s44), publish one fixed equal-weight mean
of the three validated seed forecasts as `prefix_mean3`, then hide each
single-seed row with explicit `visibility.json` rules. No weight is fitted and no
label is read; this matches the earlier drop_t035 mean3 rows.

    python publish_mean3.py                # validate + build means + dry-run scores (no external change)
    python publish_mean3.py --apply        # publish missing means, then push the hide policy
    python publish_mean3.py --apply --set union   # the ablation-union arms, after their campaign completes
    python publish_mean3.py --apply --set all     # the all-negative stack, after its campaign completes
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

from ensemble import average_predictions

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
R = REPO / 'research'
WORK = ROOT / 'publication'
MEMBER = 'seongeun'
SEEDS = (42, 43, 44)
KEYS = ['origin_last_input_utc', 'horizon_hours']
CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}
GLOBAL_LOCK = R / 'rerun_df_ffill_20261103/.publish.lock'
ARCHITECTURE = f'research/{ROOT.name}/published/ensemble.py'
OFFSET = 'e2000fs_calib_offset_roll'
COMPLETE = R / 'offset_roll_complete_ablation_20261106/runs'
ABL2 = R / 'offset_roll_ablation2_20261105/lb_publish/abl2'


def group(prefix, runs, note):
    return dict(prefix=prefix, runs={s: Path(str(runs).format(seed=s)) for s in SEEDS}, note=note)


# Seed groups whose single-seed rows are on the board. Constituent runs are the exact
# predictions of the published seed rows (checked value-for-value against the store).
SETS = {
    'board': [
        group(f'{OFFSET}_abl1_loss_t1_sp1_offset', COMPLETE / 'loss_t1_sp1_offset_s{seed}', 'tail weight 1, spectral on, offset on'),
        group(f'{OFFSET}_abl1_loss_t4_sp0_offset', COMPLETE / 'loss_t4_sp0_offset_s{seed}', 'tail 4, spectral off, offset on'),
        group(f'{OFFSET}_abl1_loss_t4_sp1_nooffset', COMPLETE / 'loss_t4_sp1_nooffset_s{seed}', 'tail 4, spectral on, offset off'),
        group(f'{OFFSET}_abl2_baseline', COMPLETE / 'baseline_s{seed}', 'original e2000fs_calib_offset_roll configuration'),
        group(f'{OFFSET}_abl2_c1_rounds300', ABL2 / 'c1_rounds300_s{seed}', '300 boosting rounds'),
        group(f'{OFFSET}_abl2_f1_drop_strip', ABL2 / 'f1_drop_strip_s{seed}', 'coronal-hole strip features removed'),
        group(f'{OFFSET}_abl2_l1_spectral_c03', ABL2 / 'l1_spectral_c03_s{seed}', 'spectral coefficient 0.3'),
        group(f'{OFFSET}_drop_t035_v650_lag24_48_72', R / 'lag24_48_72_confirm_20261109/runs/lag24_48_72_s{seed}',
              'D + v_lag24/v_lag48/v_lag72 joint addition'),
    ],
    'union': [
        group(f'{OFFSET}_drop_t035_v650_union_core', R / 'ablation_union_20261109/runs/union_core_s{seed}',
              'D + ablation-union Core arm (15 added inputs)'),
        group(f'{OFFSET}_drop_t035_v650_union_broad', R / 'ablation_union_20261109/runs/union_broad_s{seed}',
              'D + ablation-union Broad arm (22 added inputs)'),
    ],
    'all': [
        group(f'{OFFSET}_all_negative_stack', R / 'ablation_all_20261109/runs/all_negative_s{seed}',
              'D minus 9 further negative removals plus the 22 union-Broad additions (41 inputs)'),
    ],
}
# Means already on the board; only their single-seed rows still need hiding.
EXISTING_MEANS = [f'{OFFSET}_drop_t035_equatorial', f'{OFFSET}_drop_t035_v650']


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


def setup_client():
    client = WORK / 'submit-client'
    client.mkdir(parents=True, exist_ok=True)
    for name in ('submit.py', 'board.py', 'architecture_code.py', 'metric_assets.py', 'lbstatus.py'):
        shutil.copy2(REPO / 'leaderboard' / name, client / name)
    shutil.copytree(REPO / 'leaderboard/scorer', client / 'scorer', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(REPO / 'leaderboard/visibility.json', client / 'visibility.json')
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
    spec = importlib.util.spec_from_file_location('historical_publisher', R / 'offset_roll_ablation2_20261105/publish_lb.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.publish_sources


def sources(run):
    return {p.name: p.read_bytes() for p in sorted(run.iterdir())
            if p.is_file() and ((p.suffix == '.py' and p.name != 'runner_at_launch.py') or p.name == 'config.json')}


def validate_group(spec, board):
    """Same configuration except the seed, normal exits, full metadata, exact match with any published seed row."""
    runs, prefix = spec['runs'], spec['prefix']
    configs = {}
    for seed, run in runs.items():
        for name in ('train.exit', 'config.json', 'predictions.parquet', 'experiment.py', 'F1_metadata.json'):
            if not (run / name).is_file():
                raise RuntimeError(f'{run}: missing {name}')
        if (run / 'train.exit').read_text().strip() != '0':
            raise RuntimeError(f'{run}: training did not exit 0')
        config = read(run / 'config.json')
        if config.get('seed') != seed:
            raise ValueError(f'{run}: config seed {config.get("seed")} != {seed}')
        # Seed-tied references (e.g. reuse_run=baseline_s43) compare equal after removing their own seed suffix.
        configs[seed] = {k: (v[:-len(f'_s{seed}')] + '_s{seed}' if isinstance(v, str) and v.endswith(f'_s{seed}') else v)
                         for k, v in config.items() if k not in ('seed', 'experiment')}
        for period, cutoff in CUTOFFS.items():
            report = read(run / f'{period}_metadata.json')
            if report.get('smoke') or report.get('seed') != seed or report.get('training_cutoff', cutoff) != cutoff:
                raise ValueError(f'{run}/{period}: smoke, seed or cutoff mismatch')
    if len({json.dumps(c, sort_keys=True) for c in configs.values()}) != 1:
        raise ValueError(f'{prefix}: seeds differ in more than the seed')
    frames = {seed: pd.read_parquet(run / 'predictions.parquet') for seed, run in runs.items()}
    for seed, frame in frames.items():
        stored = board.get(f'{prefix}_s{seed}')
        if stored is None:
            continue
        a = frame.assign(**{KEYS[0]: pd.to_datetime(frame[KEYS[0]], utc=True)}).sort_values(KEYS).reset_index(drop=True)
        b = stored.assign(**{KEYS[0]: pd.to_datetime(stored[KEYS[0]], utc=True)}).sort_values(KEYS).reset_index(drop=True)
        if not (a[KEYS].equals(b[KEYS]) and np.array_equal(a.pred_kms.to_numpy(), b.pred_kms.to_numpy())):
            raise ValueError(f'{prefix}_s{seed}: local constituent differs from the published seed row')
    return frames


def inference_seconds(run):
    reports = [read(run / f'{p}_metadata.json') for p in CUTOFFS]
    if not all('inference_seconds_per_fold' in r and 'query_origins' in r for r in reports):
        return None
    return sum(r['inference_seconds_per_fold'] * r['query_origins'] for r in reports) / sum(r['query_origins'] for r in reports)


def metadata(spec, ensemble, revision):
    members, seconds = [], ensemble['prediction_mean_seconds_per_fold']
    for seed, run in spec['runs'].items():
        report = read(run / 'F1_metadata.json')
        per_fold = inference_seconds(run)
        seconds = None if per_fold is None or seconds is None else seconds + per_fold
        members.append(dict(seed=seed, run=str(run.relative_to(REPO)), run_config=read(run / 'config.json'),
                            parameters=report.get('parameters'), rounds=report.get('rounds'),
                            features=report.get('features'), feature_count=report.get('feature_count')))
    meta = dict(team_member=MEMBER, experiment=f"{spec['prefix']}_mean3",
                description=(f"Fixed arithmetic mean of seeds 42/43/44 of {spec['prefix']} ({spec['note']}). "
                             'Equal 1/3 weights; no fitted weights, no labels. Constituents are the same validated runs '
                             'as the individual seed rows (exact prediction match where published). The board shows '
                             '3-seed means; single-seed rows are hidden by policy. Official F1/F2/F3, 144h maturity.'),
                no_future_leakage=True,
                architecture_url=f'https://github.com/TLAB-HGU/sunrun-leaderboard/blob/{revision}/{ARCHITECTURE}',
                config={'data': {'constituent_runs': [m['run'] for m in members], 'maturity_hours': 144,
                                 'training_cutoffs': CUTOFFS},
                        'hyperparameters': {'aggregation': 'arithmetic_mean', 'weights': [1 / 3] * 3,
                                            'seeds': list(SEEDS), 'constituents': members}},
                **{f'{p.lower()}_training_data_end_utc': d + 'T00:00:00Z' for p, d in CUTOFFS.items()})
    if seconds is not None:
        meta['inference_seconds_per_fold'] = seconds
    return meta


def experiment_name(run):
    config = read(run / 'config.json')
    if config.get('experiment'):
        return config['experiment']
    return read(run / 'launch.json')['experiment']


def replay(spec):
    bundle = {'ensemble.py': (ROOT / 'ensemble.py').read_text()}
    names = {}
    for seed, run in spec['runs'].items():
        for name, body in sources(run).items():
            bundle[f's{seed}/{name}'] = body.decode()
        names[seed] = experiment_name(run)
    return ('"""Reproduce the three constituent seed models and their fixed mean; CPU only."""\n'
            'import argparse, os, subprocess, sys, tempfile\nfrom pathlib import Path\n'
            f'SOURCES = {bundle!r}\nNAMES = {names!r}\n'
            'if __name__ == "__main__":\n'
            '    parser = argparse.ArgumentParser(description=__doc__)\n'
            '    parser.add_argument("--data", required=True)\n'
            '    parser.add_argument("--observations", required=True)\n'
            '    parser.add_argument("--out", required=True, type=Path)\n'
            '    parser.add_argument("--threads", type=int, default=4)\n'
            '    args = parser.parse_args()\n'
            '    os.environ["CUDA_VISIBLE_DEVICES"] = ""\n'
            '    work = Path(tempfile.mkdtemp(prefix="seed_mean3_"))\n'
            '    for name, body in SOURCES.items():\n'
            '        path = work / name\n'
            '        path.parent.mkdir(parents=True, exist_ok=True)\n'
            '        path.write_text(body)\n'
            '    predictions = []\n'
            '    for seed in (42, 43, 44):\n'
            '        out = args.out / f"seed{seed}"\n'
            '        subprocess.run([sys.executable, str(work / f"s{seed}/experiment.py"), "--experiment", NAMES[seed],\n'
            '                        "--config", str(work / f"s{seed}/config.json"), "--data", args.data,\n'
            '                        "--observations", args.observations, "--out", str(out), "--device", "cpu",\n'
            '                        "--threads", str(args.threads)], check=True, cwd=work / f"s{seed}")\n'
            '        predictions.append(str(out / "predictions.parquet"))\n'
            '    subprocess.run([sys.executable, str(work / "ensemble.py"), "--predictions", *predictions,\n'
            '                    "--out", str(args.out / "predictions.parquet")], check=True)\n').encode()


def stored_predictions(submit, api, wanted):
    """Stored prediction parquet of each wanted published seed row (located by its unique store folder)."""
    revision = api.repo_info(submit.DATASET, repo_type='dataset').sha
    files = api.list_repo_files(submit.DATASET, repo_type='dataset', revision=revision)
    out = {}
    for experiment in wanted:
        paths = [f for f in files if f.startswith(f'submissions/{MEMBER}/{experiment}/') and f.endswith('.parquet')]
        if len(paths) > 1:
            raise RuntimeError('Multiple stored rows for ' + experiment)
        if paths:
            out[experiment] = pd.read_parquet(submit._download(submit.DATASET, paths[0], revision=revision))
    return out


def prepare(submit, api, groups):
    wanted = {f"{g['prefix']}_s{s}" for g in groups for s in SEEDS}
    board = stored_predictions(submit, api, wanted)
    observations = read(R / 'feature_addition_all_20261108/assignment.global.json')['observations']
    context = submit.prepare_context(pd.read_parquet(observations))
    items = []
    for spec in groups:
        frames = validate_group(spec, board)
        folder = WORK / spec['prefix']
        folder.mkdir(parents=True, exist_ok=True)
        output, seconds = average_predictions([frames[s] for s in SEEDS])
        path = folder / 'predictions.parquet'
        if path.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(path), output, check_exact=True)
        else:
            output.to_parquet(path, index=False)
        ensemble = dict(seeds=list(SEEDS), weights=[1 / 3] * 3, rows=len(output), keys_identical=True,
                        prediction_mean_seconds_per_fold=seconds / output[KEYS[0]].nunique(),
                        policy='fixed arithmetic mean; no labels or fitted weights')
        write(folder / 'ensemble.json', ensemble)
        score = submit.score_competition(pd.read_parquet(path), context)
        if score['total']['n_replaced']:
            raise ValueError(spec['prefix'] + ': official scorer replaced forecasts')
        seeds = [submit.score_competition(frames[s], context)['total']['s'] for s in SEEDS]
        items.append(dict(spec=spec, path=path, ensemble=ensemble, score=score, seed_scores=seeds))
        print(json.dumps(dict(stage='prepared', experiment=spec['prefix'] + '_mean3', s=score['total']['s'],
                              seed_s=seeds, seed_mean=float(np.mean(seeds)),
                              published_seeds=sorted(int(n[-2:]) for n in board if n.startswith(spec['prefix'] + '_s')))),
              flush=True)
    return items


def existing(records, experiment, config=None):
    rows = [r for r in records if r['team_member'] == MEMBER and r['experiment'] == experiment]
    if len(rows) > 1 or (rows and config is not None and rows[0]['config'] != config):
        raise RuntimeError(f'{experiment}: conflicting existing rows; refuse overwrite')
    return rows[0] if rows else None


def publish_means(submit, api, items):
    records = submit.all_results(api, submit.DATASET)
    pending = [it for it in items if existing(records, it['spec']['prefix'] + '_mean3') is None]
    files = {ARCHITECTURE: (ROOT / 'ensemble.py').read_bytes(),
             f'research/{ROOT.name}/published/publish_mean3.py': Path(__file__).read_bytes()}
    for it in pending:
        meta = metadata(it['spec'], it['ensemble'], '0' * 40)
        body = replay(it['spec'])
        location = submit.replay_script_path(MEMBER, meta['experiment'], meta['config'])
        compile(body, location, 'exec')
        files[location] = body
        local = submit.HERE / location
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(body)
        dry = submit.submit(str(it['path']), meta, dry_run=True)
        if abs(dry['competition']['total']['s'] - it['score']['total']['s']) > 1e-6:
            raise ValueError(meta['experiment'] + ': dry-run score differs from local official score')
        write(WORK / f"{meta['experiment']}.dry-run.json", dry)
    revision = None
    if pending:
        revision_file = WORK / 'source-revision.json'
        saved = read(revision_file) if revision_file.exists() else {}
        if saved and all(it['spec']['prefix'] in saved.get('prefixes', []) for it in pending):
            revision = saved['revision']
        else:
            revision = source_publisher()(files)
            write(revision_file, dict(revision=revision, prefixes=[it['spec']['prefix'] for it in pending], utc=utc_now()))
    receipts = read(WORK / 'receipts.json') if (WORK / 'receipts.json').exists() else {}
    for it in items:
        experiment = it['spec']['prefix'] + '_mean3'
        record = existing(submit.all_results(api, submit.DATASET), experiment)
        if record is None:
            meta = metadata(it['spec'], it['ensemble'], revision)
            (WORK / f'{experiment}.meta.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
            submit.submit(str(it['path']), meta)  # also registers the architecture link and refreshes the page
            record = existing(submit.all_results(api, submit.DATASET), experiment, meta['config'])
            if record is None:
                raise RuntimeError(experiment + ': submission could not be read back')
        if abs(record['competition']['total']['s'] - it['score']['total']['s']) > 1e-6:
            raise ValueError(experiment + ': stored score differs from local official score')
        receipts[experiment] = dict(submitted_utc=record['submitted_utc'], total_s=record['competition']['total']['s'],
                                    seed_s=it['seed_scores'], architecture_url=record['architecture_url'])
        write(WORK / 'receipts.json', receipts)
        print(json.dumps(dict(stage='mean_published', experiment=experiment, **receipts[experiment])), flush=True)
    return receipts


def hide_singles(submit, api, prefixes):
    """Add hidden rules for every single-seed row of these prefixes; push the Space policy of record."""
    import architecture_code
    from huggingface_hub import hf_hub_download
    space_revision = api.repo_info(submit.SPACE, repo_type='space').sha
    policy = read(hf_hub_download(submit.SPACE, 'visibility.json', repo_type='space', revision=space_revision))
    local_path = REPO / 'leaderboard/visibility.json'
    if read(local_path) != policy:
        raise RuntimeError('Local leaderboard/visibility.json differs from the Space policy of record; resolve first')
    present = {r['experiment'] for r in submit.all_results(api, submit.DATASET) if r['team_member'] == MEMBER}
    hidden = {(h['member'], h['experiment']) for h in policy['hidden']}
    means = [f'{p}_mean3' for p in prefixes]
    if any(m not in present or (MEMBER, m) in hidden for m in means):
        raise RuntimeError('Every 3-seed mean must be published and visible before hiding its seed rows')
    added = [dict(member=MEMBER, experiment=f'{p}_s{s}') for p in prefixes for s in SEEDS
             if f'{p}_s{s}' in present and (MEMBER, f'{p}_s{s}') not in hidden]
    if added:
        backup = WORK / f'visibility.before-{space_revision[:10]}.json'
        if not backup.exists():
            shutil.copy2(local_path, backup)
        policy = dict(policy, hidden=policy['hidden'] + added)
        for path in (local_path, Path(submit.__file__).with_name('visibility.json')):
            path.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + '\n')
        submit.publish(api, [], submit.SPACE, dataset=submit.DATASET, push_policy=True)
    page, _, revision = architecture_code.space_files(submit.SPACE)
    visible_singles = [f'{p}_s{s}' for p in prefixes for s in SEEDS if f'{p}_s{s}<' in page or f'{p}_s{s}"' in page]
    missing_means = [m for m in means if m not in page]
    if visible_singles or missing_means:
        raise RuntimeError(f'Page check failed: visible singles {visible_singles}, missing means {missing_means}')
    return dict(added_hidden=[a['experiment'] for a in added], space_revision=revision, means_visible=means)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--set', choices=sorted(SETS), default='board')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import HfApi
    api = HfApi()
    with (WORK / '.publication.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        submit = setup_client()
        items = prepare(submit, api, SETS[args.set])
        if not args.apply:
            print(json.dumps(dict(stage='dry', published=False, groups=len(items))), flush=True)
            return
        with GLOBAL_LOCK.open('a') as global_lock:
            fcntl.flock(global_lock, fcntl.LOCK_EX)
            receipts = publish_means(submit, api, items)
            prefixes = [it['spec']['prefix'] for it in items] + (EXISTING_MEANS if args.set == 'board' else [])
            hidden = hide_singles(submit, api, prefixes)
        write(WORK / f'verified-{args.set}.json', dict(set=args.set, verified_utc=utc_now(), means=receipts, policy=hidden,
              rule='board shows fixed equal-weight 3-seed means; single-seed rows hidden'))
        print(json.dumps(dict(stage='complete', set=args.set, means=len(receipts), hidden=len(hidden['added_hidden']))), flush=True)


if __name__ == '__main__':
    main()
