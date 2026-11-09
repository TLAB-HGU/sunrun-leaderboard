"""Publish the user-requested fixed three-seed mean without fitting ensemble weights."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
from pathlib import Path

import yaml

from publish_drop035 import ROOT, REPO, MEMBER, SEEDS, MODULES, setup_client, metadata

WORK = ROOT / 'lb_publish_drop035/ensemble'
EXPERIMENT = 'e2000fs_calib_offset_roll_drop_t035_equatorial_mean3'
SOURCE = ROOT / 'ensemble_drop035.py'
ARCHITECTURE = f'research/{ROOT.name}/published_drop035/ensemble.py'


def build_meta(revision):
    constituent = [metadata(ROOT / 'runs' / f'drop_ch_ch_t035_equatorial_area_s{s}', revision) for s in SEEDS]
    aggregation = json.loads((WORK / 'predictions.ensemble.json').read_text())
    meta = dict(constituent[0])
    meta.update(experiment=EXPERIMENT,
                description='Fixed equal-weight arithmetic prediction mean of drop_ch_ch_t035_equatorial_area seeds 42/43/44. '
                            'All three constituents remove only ch_ch_t035_equatorial_area from original A. '
                            'Weights are exactly 1/3 each, not fitted or selected using evaluation targets. '
                            'Each seed retains its training-only frozen calibration. No B1 blend. '
                            'Published on explicit user request 2026-11-07; this is a three-model ensemble, not a single model. '
                            'Inference time is the sum of measured constituent inference times plus measured mean computation.',
                architecture_url=f'https://github.com/TLAB-HGU/sunrun-leaderboard/blob/{revision}/{ARCHITECTURE}',
                inference_seconds_per_fold=sum(m['inference_seconds_per_fold'] for m in constituent)
                                           + aggregation['prediction_mean_seconds_per_fold'],
                config={'data': constituent[0]['config']['data'],
                        'hyperparameters': {'aggregation': 'arithmetic_mean', 'seeds': list(SEEDS),
                                            'weights': [1 / 3] * 3,
                                            'constituents': [m['config']['hyperparameters'] for m in constituent]}})
    return meta


def replay_source():
    sources = {'ensemble.py': SOURCE.read_text()}
    for seed in SEEDS:
        run = ROOT / 'runs' / f'drop_ch_ch_t035_equatorial_area_s{seed}'
        for name in MODULES:
            sources[f's{seed}/{name}'] = (run / name).read_text()
    return ('"""Reproduce all three fixed constituents and then their equal-weight mean. CPU only."""\n'
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
            '    work = Path(tempfile.mkdtemp(prefix="drop035_mean3_"))\n'
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    submit = setup_client()
    import architecture_code
    from huggingface_hub import HfApi
    import pandas as pd
    meta = build_meta('0' * 40)
    (WORK / 'config.json').write_text(json.dumps(meta['config'], indent=2) + '\n')
    path = submit.replay_script_path(MEMBER, EXPERIMENT, meta['config'])
    replay = replay_source()
    compile(replay, path, 'exec')
    local = submit.HERE / path
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(replay)
    local_competition = submit.score_competition(pd.read_parquet(WORK / 'predictions.parquet'),
                                                submit.prepare_context(pd.read_parquet(ROOT / 'competition-observations.parquet')))
    (WORK / 'local_score.json').write_text(json.dumps({'competition': local_competition}, indent=2) + '\n')
    dry = submit.submit(str(WORK / 'predictions.parquet'), meta, dry_run=True)
    assert abs(dry['competition']['total']['s'] - local_competition['total']['s']) <= 1e-6
    assert dry['competition']['total']['replacement_fraction'] == 0
    (WORK / 'dry-run.json').write_text(json.dumps(dry, indent=2) + '\n')
    print(json.dumps({'dry_run': EXPERIMENT, 's': dry['competition']['total']['s']}), flush=True)
    if not args.apply:
        return
    spec = importlib.util.spec_from_file_location('historical_publisher', REPO / 'research/offset_roll_ablation2_20261105/publish_lb.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    revision_file = WORK / 'source-revision.txt'
    revision = revision_file.read_text().strip() if revision_file.exists() else old.publish_sources({ARCHITECTURE: SOURCE.read_bytes(), path: replay})
    revision_file.write_text(revision + '\n')
    meta = build_meta(revision)
    (WORK / 'meta.yaml').write_text(yaml.safe_dump(meta, sort_keys=False))
    api = HfApi()
    def snapshot_results(api, repo, revision=None):
        revision = revision or api.repo_info(repo, repo_type='dataset').sha
        paths = [p for p in api.list_repo_files(repo, repo_type='dataset', revision=revision) if p.startswith('results/')]
        def read(path):
            return json.loads(Path(submit._download(repo, path, revision=revision)).read_text())
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(read, paths))
    submit.all_results = snapshot_results
    existing = [r for r in submit.all_results(api, submit.DATASET) if r['team_member'] == MEMBER and r['experiment'] == EXPERIMENT]
    if existing:
        assert len(existing) == 1 and existing[0]['config'] == meta['config']
        result = existing[0]
    else:
        result = submit.submit(str(WORK / 'predictions.parquet'), meta)
    stored = next(r for r in submit.all_results(api, submit.DATASET)
                  if r['team_member'] == MEMBER and r['experiment'] == EXPERIMENT and r['submitted_utc'] == result['submitted_utc'])
    assert abs(stored['competition']['total']['s'] - local_competition['total']['s']) <= 1e-6
    assert stored['architecture_url'] == meta['architecture_url']
    (WORK / 'submission.json').write_text(json.dumps(dict(stored, published=True, local_only=False), indent=2) + '\n')
    page, _, _ = architecture_code.space_files(submit.SPACE)
    assert EXPERIMENT in page and architecture_code.reachable(meta['architecture_url'])
    assert not architecture_code.script_rows(page or '')
    (WORK / 'verified.json').write_text(json.dumps({'remote_score_matches': True, 'page_row_present': True,
        'architecture_reachable': True, 'missing_architecture_links': 0}, indent=2) + '\n')
    print(json.dumps({'published': EXPERIMENT, 's': stored['competition']['total']['s'],
                      'submitted_utc': stored['submitted_utc']}), flush=True)


if __name__ == '__main__':
    with (REPO / 'research/rerun_df_ffill_20261103/.publish.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
