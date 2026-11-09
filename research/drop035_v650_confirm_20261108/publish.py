"""Resumable publication of the two-removal seeds and fixed three-seed mean."""
import argparse
import fcntl
import importlib.util
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from ensemble import average_predictions

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
WORK = ROOT / 'publication'
MEMBER = 'seongeun'
EXPERIMENT = 'e2000fs_calib_offset_roll_drop_t035_v650_mean3'
SEEDS = (42, 43, 44)
MODULES = ('experiment.py', 'core.py', 'baseline.py', 'competition.py')
CUTOFFS = {'F1': '2025-12-29', 'F2': '2026-03-29', 'F3': '2026-06-29'}
EXCLUDED = ['ch_ch_t035_equatorial_area', 'strip_v650_dark']
RETAINED_FEATURES = ['v_lag0', 'ace_speed_kms_last', 'ace_density_cm3_last', 'ace_density_cm3_m6', 'ace_temperature_k_last', 'ace_temperature_k_m24', 'ace_bt_nt_last', 'ace_bt_nt_m6', 'ace_by_gsm_nt_m24', 'ace_epam_p_channel3_last', 'ace_sis_gt10_flux_last', 'ch_ch_t035_top2_area', 'ch_ch_t035_meridian15_equatorial_area', 'ch_ch_t035_meridian30_equatorial_area', 'ch_ch_t055_meridian15_area', 'ch_ch_t055_meridian30_equatorial_area', 'ch_ch_t055_equatorial_area', 'img_age_h', 'enlil_n_pending', 'enlil_kp', 'rec26', 'rec27', 'rec28', 'rec_mean', 'strip_v350_logmed', 'strip_v450_logmed', 'strip_v550_dark', 'strip_v750_logmed']
ARCHITECTURE = f'research/{ROOT.name}/published/ensemble.py'
GLOBAL_LOCK = REPO / 'research/rerun_df_ffill_20261103/.publish.lock'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def utc_now():
    return datetime.now(UTC).isoformat()


def validate_runs():
    runs = {seed: ROOT / 'runs' / f'drop035_v650_s{seed}' for seed in SEEDS}
    for seed, run in runs.items():
        if not (run / 'train.exit').is_file() or (run / 'train.exit').read_text().strip() != '0':
            raise RuntimeError(f'{run.name}: successful training exit required before publication')
        if not (run / 'training-validation.json').is_file():
            raise RuntimeError(f'{run.name}: training validation receipt required')
        assert (run / 'experiment.py').read_bytes() == (run / 'runner_at_launch.py').read_bytes()
        config = read(run / 'config.json')
        expected = {'exclude_features': EXCLUDED, 'tail_weight': 4, 'spectral': True, 'offset': True,
                        'offset_scope': 'horizon', 'offset_window': 336, 'alpha': .25, 'block_size': 24,
                        'clip_component': True, 'clip_final': True, 'seed': seed,
                        'spectral_hessian_only': False, 'experiment': run.name}
        assert config == expected
        assert read(run / 'resolved_config.json') == expected
        for period, cutoff in CUTOFFS.items():
            report = read(run / f'{period}_metadata.json')
            assert not report['smoke'] and report['training_cutoff'] == cutoff
            assert report['maturity_hours'] == 144 and report['seed'] == seed
            assert report['feature_count'] == 28 and report['rounds'] == 900
            assert report['config'] == expected
            assert report['period'] == period
            names = [name for group in report['features'].values() for name in group]
            assert len(names) == 28 and set(names) == set(RETAINED_FEATURES)
    for module in MODULES[1:]:
        assert all((r / module).read_bytes() == (runs[42] / module).read_bytes() for r in runs.values())
    return runs


def prepare_predictions(runs):
    frames = [pd.read_parquet(run / 'predictions.parquet') for run in runs.values()]
    output, seconds = average_predictions(frames)
    path = WORK / 'predictions.parquet'
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path), output, check_exact=True)
        assert (WORK / 'ensemble.json').is_file()
    else:
        output.to_parquet(path, index=False)
        origins = output['origin_last_input_utc'].nunique()
        write(WORK / 'ensemble.json', {'seeds': list(SEEDS), 'weights': [1 / 3] * 3,
              'keys_identical': True, 'rows': len(output), 'origins': origins,
              'prediction_mean_seconds': seconds, 'prediction_mean_seconds_per_fold': seconds / origins,
              'policy': 'fixed arithmetic mean; no labels or fitted weights'})
    return path


def metadata(runs, revision):
    members = []
    seconds = 0.0
    for seed, run in runs.items():
        reports = [read(run / f'{p}_metadata.json') for p in CUTOFFS]
        seconds += sum(r['inference_seconds_per_fold'] * r['query_origins'] for r in reports) / sum(r['query_origins'] for r in reports)
        members.append({'seed': seed, 'run_config': read(run / 'config.json'),
                            'resolved_config': read(run / 'resolved_config.json'),
                            'parameters': reports[0]['parameters'], 'rounds': reports[0]['rounds'],
                            'features': reports[0]['features'], 'feature_count': reports[0]['feature_count']})
    seconds += read(WORK / 'ensemble.json')['prediction_mean_seconds_per_fold']
    return dict(team_member=MEMBER, experiment=EXPERIMENT,
                description='Exploratory combination of two independently 3/3-seed improving removals: '
                'ch_ch_t035_equatorial_area and strip_v650_dark. Original A otherwise unchanged, '
                'alpha=.25, full spectral term, no B1. Fixed arithmetic mean of fresh seeds 42/43/44; '
                'equal 1/3 weights, no label-fitted weights. Official F1/F2/F3, training-only calibration '
                'and 144h maturity. Selected after prior official LOCO scores; no independent confirmation claim.',
                no_future_leakage=True, inference_seconds_per_fold=seconds,
                architecture_url=f'https://github.com/TLAB-HGU/sunrun-leaderboard/blob/{revision}/{ARCHITECTURE}',
                config={'data': {'campaign': ROOT.name, 'parent': 'e2000fs_calib_offset_roll',
                                 'source': 'rerun_df_ffill_20261103/input/df_ffill_trainy.parquet',
                                 'maturity_hours': 144, 'training_cutoffs': CUTOFFS},
                        'hyperparameters': {'aggregation': 'arithmetic_mean', 'weights': [1 / 3] * 3,
                                            'seeds': list(SEEDS), 'constituents': members}},
                **{f'{p.lower()}_training_data_end_utc': d + 'T00:00:00Z' for p, d in CUTOFFS.items()})


def replay_source(runs):
    sources = {'ensemble.py': (ROOT / 'ensemble.py').read_text()}
    for seed, run in runs.items():
        for name in MODULES:
            sources[f's{seed}/{name}'] = (run / name).read_text()
    return ('"""Reproduce all three exploratory constituent models and their fixed mean; CPU only."""\n'
            'import argparse, os, subprocess, sys, tempfile\nfrom pathlib import Path\n'
            f'SOURCES = {sources!r}\n'
            'if __name__ == "__main__":\n'
            '    parser = argparse.ArgumentParser(description=__doc__)\n'
            '    parser.add_argument("--data", required=True)\n'
            '    parser.add_argument("--observations", required=True)\n'
            '    parser.add_argument("--out", required=True, type=Path)\n'
            '    parser.add_argument("--threads", type=int, default=4)\n'
            '    args = parser.parse_args()\n'
            '    os.environ["CUDA_VISIBLE_DEVICES"] = ""\n'
            '    work = Path(tempfile.mkdtemp(prefix="drop035_v650_mean3_"))\n'
            '    for name, body in SOURCES.items():\n'
            '        path = work / name\n'
            '        path.parent.mkdir(parents=True, exist_ok=True)\n'
            '        path.write_text(body)\n'
            '    args.out.mkdir(parents=True, exist_ok=True)\n'
            '    predictions = []\n'
            '    for seed in (42, 43, 44):\n'
            '        out = args.out / f"seed{seed}"\n'
            '        subprocess.run([sys.executable, str(work / f"s{seed}/experiment.py"), "--data", args.data,\n'
            '                        "--observations", args.observations, "--out", str(out), "--device", "cpu",\n'
            '                        "--threads", str(args.threads)], check=True)\n'
            '        predictions.append(str(out / "predictions.parquet"))\n'
            '    subprocess.run([sys.executable, str(work / "ensemble.py"), "--predictions", *predictions,\n'
            '                    "--out", str(args.out / "predictions.parquet")], check=True)\n').encode()


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
        def download(path):
            return read(submit._download(repo, path, revision=revision))
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(download, files))
    submit.all_results = snapshot_results
    return submit


def source_publisher():
    spec = importlib.util.spec_from_file_location('historical_publisher', REPO / 'research/offset_roll_ablation2_20261105/publish_lb.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.publish_sources


def existing_result(records, config, experiment=EXPERIMENT):
    matches = [r for r in records if r['team_member'] == MEMBER and r['experiment'] == experiment]
    if not matches:
        return None
    if len(matches) != 1 or matches[0]['config'] != config:
        raise RuntimeError('Existing experiment has a different config or multiple rows; refuse overwrite')
    return matches[0]


def seed_metadata(run, revision):
    seed = read(run / 'config.json')['seed']
    report = read(run / 'F1_metadata.json')
    reports = [read(run / f'{p}_metadata.json') for p in CUTOFFS]
    return dict(team_member=MEMBER,
                experiment=f'e2000fs_calib_offset_roll_drop_t035_v650_s{seed}',
                description=f'Fresh original-A retraining with t035 equatorial-area and v650-dark removed, seed {seed}. '
                'All three seeds 42/43/44 published, no seed selection. 28 source features; alpha=.25, full spectral, '
                'tail4, training-only offset horizon/recent336, block24, clips and 900 rounds unchanged. '
                'No B1. Exploratory combination chosen after prior official LOCO scores; no independent confirmation claim.',
                no_future_leakage=True,
                inference_seconds_per_fold=sum(r['inference_seconds_per_fold'] * r['query_origins'] for r in reports)
                / sum(r['query_origins'] for r in reports),
                architecture_url=f'https://github.com/TLAB-HGU/sunrun-leaderboard/blob/{revision}/research/{ROOT.name}/published/core.py',
                config={'data': {'campaign': ROOT.name, 'parent': 'e2000fs_calib_offset_roll',
                                 'source': 'rerun_df_ffill_20261103/input/df_ffill_trainy.parquet',
                                 'maturity_hours': 144, 'training_cutoffs': CUTOFFS},
                        'hyperparameters': {'run_config': read(run / 'config.json'),
                                            'resolved_config': read(run / 'resolved_config.json'),
                                            'parameters': report['parameters'], 'rounds': report['rounds'],
                                            'features': report['features'], 'feature_count': 28}},
                **{f'{p.lower()}_training_data_end_utc': d + 'T00:00:00Z' for p, d in CUTOFFS.items()})


def seed_replay(run):
    sources = {name: (run / name).read_text() for name in MODULES}
    return ('"""Exact launch-time CPU replay; pass --data, --observations, --out."""\n'
            'import os, runpy, sys, tempfile\nfrom pathlib import Path\n'
            f'SOURCES = {sources!r}\n'
            'if __name__ == "__main__":\n'
            '    os.environ["CUDA_VISIBLE_DEVICES"] = ""\n'
            '    work = Path(tempfile.mkdtemp(prefix="drop035_v650_replay_"))\n'
            '    for name, body in SOURCES.items():\n'
            '        (work / name).write_text(body)\n'
            '    sys.path.insert(0, str(work))\n'
            '    runpy.run_path(str(work / "experiment.py"), run_name="__main__")\n').encode()


def publication_items(runs, prediction_path, revision):
    items = [(seed_metadata(run, revision), run / 'predictions.parquet', seed_replay(run), run)
             for run in runs.values()]
    items.append((metadata(runs, revision), prediction_path, replay_source(runs), WORK))
    return items


def publish_result(submit, runs, prediction_path):
    import architecture_code
    from huggingface_hub import HfApi
    api = HfApi()
    prefix = f'research/{ROOT.name}/published'
    files = {ARCHITECTURE: (ROOT / 'ensemble.py').read_bytes(),
             f'{prefix}/strategy.md': (ROOT / 'strategy.md').read_bytes()}
    for name in MODULES[1:]:
        files[f'{prefix}/{name}'] = (runs[42] / name).read_bytes()
    for seed, run in runs.items():
        files[f'{prefix}/seed{seed}/experiment.py'] = (run / 'experiment.py').read_bytes()
    observations = read(ROOT / 'assignment.global.json')['observations']
    context = submit.prepare_context(pd.read_parquet(observations))
    scores = {}
    # Validate all four dry runs before any external mutation.
    for meta, path, replay, _ in publication_items(runs, prediction_path, '0' * 40):
        experiment = meta['experiment']
        replay_path = submit.replay_script_path(MEMBER, experiment, meta['config'])
        compile(replay, replay_path, 'exec')
        files[replay_path] = replay
        local = submit.HERE / replay_path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(replay)
        score = submit.score_competition(pd.read_parquet(path), context)
        dry = submit.submit(str(path), meta, dry_run=True)
        assert abs(dry['competition']['total']['s'] - score['total']['s']) <= 1e-6
        assert dry['competition']['total']['replacement_fraction'] == 0
        if path.parent != WORK:
            recorded = read(path.parent / 'local_score.json')['competition']['total']['s']
            assert abs(recorded - score['total']['s']) <= 1e-6
        scores[experiment] = score
        write(WORK / f'{experiment}.dry-run.json', dry)
    revision_file = WORK / 'source-revision.txt'
    if revision_file.exists():
        revision = revision_file.read_text().strip()
    else:
        revision = source_publisher()(files)
        revision_file.write_text(revision + '\n')
    receipts = []
    for meta, path, _, destination in publication_items(runs, prediction_path, revision):
        experiment = meta['experiment']
        (WORK / f'{experiment}.meta.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
        result = existing_result(submit.all_results(api, submit.DATASET), meta['config'], experiment)
        if result is None:
            result = submit.submit(str(path), meta)
        stored = existing_result(submit.all_results(api, submit.DATASET), meta['config'], experiment)
        assert stored and stored['submitted_utc'] == result['submitted_utc']
        assert abs(stored['competition']['total']['s'] - scores[experiment]['total']['s']) <= 1e-6
        assert stored['architecture_url'] == meta['architecture_url']
        receipt = destination / 'submission.json'
        if receipt.exists() and not (destination / 'submission.local.json').exists():
            shutil.copy2(receipt, destination / 'submission.local.json')
        write(receipt, dict(stored, published=True, local_only=False))
        entry = architecture_code.entry(MEMBER, experiment, stored['config_sha256'], stored['architecture_url'])
        page, _, _ = architecture_code.space_files(submit.SPACE)
        if experiment not in (page or '') or stored['architecture_url'] not in page:
            submit.publish(api, [], submit.SPACE, architecture={'models': [entry]}, dataset=submit.DATASET)
            page, _, _ = architecture_code.space_files(submit.SPACE)
        assert experiment in page and stored['architecture_url'] in page
        assert architecture_code.reachable(stored['architecture_url'])
        receipts.append({'experiment': experiment, 'submitted_utc': stored['submitted_utc'],
                             'total_s': stored['competition']['total']['s'], 'remote_score_matches': True,
                             'page_row_present': True, 'architecture_reachable': True})
        write(WORK / 'receipts.json', receipts)
        print(json.dumps({'stage': 'published', **receipts[-1]}), flush=True)
    write(WORK / 'result-verified.json', {'records': receipts, 'visible_rows': 4, 'all_scores_match': True})
    return stored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / '.publication.lock').open('w') as campaign_lock:
        fcntl.flock(campaign_lock, fcntl.LOCK_EX)
        runs = validate_runs()
        prediction_path = prepare_predictions(runs)
        if not args.apply:
            write(WORK / 'config.json', metadata(runs, '0' * 40)['config'])
            print('Local prediction mean/config prepared; use --apply for official publication and visualization.')
            return
        submit = setup_client()
        with GLOBAL_LOCK.open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            stored = publish_result(submit, runs, prediction_path)
        # publish_links.py owns GLOBAL_LOCK; release it before the subprocess.
        subprocess.run([sys.executable, str(ROOT / 'publish_visualization.py'),
                        '--submitted-utc', stored['submitted_utc']], check=True)
        visualization = read(WORK / 'visualization-verified.json')
        assert visualization['submitted_utc'] == stored['submitted_utc']
        assert visualization['space_visualization_link_verified']
        evidence = {'experiment': EXPERIMENT, 'submitted_utc': stored['submitted_utc'],
                        'total_s': stored['competition']['total']['s'], 'seeds': list(SEEDS),
                        'records': read(WORK / 'receipts.json'),
                        'weights': [1 / 3] * 3, 'official_score_matches': True, 'visualization': visualization,
                        'verified_utc': utc_now(), 'classification': 'exploratory user-selected combination'}
        write(WORK / 'verified.json', evidence)
        write(ROOT / 'campaign-complete.json', dict(evidence, status='complete', publication=str(WORK)))
        print(json.dumps({'stage': 'complete', 'experiment': EXPERIMENT, 's': evidence['total_s']}), flush=True)


if __name__ == '__main__':
    main()
