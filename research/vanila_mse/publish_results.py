"""Validate completed vanila_mse runs and prepare their fixed mean; --publish posts it.

Run only after all three seed runs finish. Source .env in the calling shell for
HF/GitHub credentials. No training or parameter selection is performed here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (42, 43, 44)
PERIODS = ('F1', 'F2', 'F3')
NAME = 'vanila_mse'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def load_runner():
    spec = importlib.util.spec_from_file_location('vanila_mse_experiment', HERE / 'experiment.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_runs(runner):
    metadata = {}
    for seed in SEEDS:
        run = HERE / 'runs' / f'mse_s{seed}'
        if not (run / 'train.exit').exists() or (run / 'train.exit').read_text().strip() != '0':
            raise ValueError(f'{run}: training has not exited successfully')
        for period in PERIODS:
            meta = json.loads((run / f'{period}_metadata.json').read_text())
            expected = (seed, period, False, 900, 24, 31, 215, runner.CUTOFFS[period], 144)
            actual = tuple(meta[k] for k in ('seed', 'period', 'smoke', 'rounds', 'block',
                                              'feature_count', 'expanded_feature_count',
                                              'training_cutoff', 'maturity_hours'))
            if actual != expected or not meta['objective'].startswith('unweighted_mse'):
                raise ValueError(f'{seed}/{period}: wrong training recipe: {actual}')
            if meta['parameters'] != dict(runner.PARAMS, seed=seed):
                raise ValueError(f'{seed}/{period}: model parameters changed')
            if meta['constant_features'] != runner.CONSTANT or meta['varying_features'] != runner.VARYING:
                raise ValueError(f'{seed}/{period}: features changed')
            for path in (run / f'{period}_predictions.npz',
                         run / 'weights' / period / f's{seed}.npz'):
                if not path.is_file():
                    raise ValueError(f'Missing {path}')
            prior, full, state = runner.load_checkpoint(run / 'weights' / period / f's{seed}.npz')
            if state != meta['frozen_calibration']:
                raise ValueError(f'{seed}/{period}: checkpoint offset differs from metadata')
            for buffer in prior + full:
                if runner.load_booster(buffer, 1).num_boosted_rounds() != 900:
                    raise ValueError(f'{seed}/{period}: checkpoint is not a 900-round model')
            metadata[seed, period] = meta
        for filename in ('predictions.parquet', 'score.json'):
            if not (run / 'results' / f's{seed}' / filename).is_file():
                raise ValueError(f'{seed}: missing {filename}')
    return metadata


def prepare(runner, data, observations, threads):
    metadata = inspect_runs(runner)
    logs = runner.Logs(HERE / 'logs', 'publication_replay')
    table = runner.load_table(data, logs)
    frames, scores, replay = [], {}, []
    for seed in SEEDS:
        run = HERE / 'runs' / f'mse_s{seed}'
        args = SimpleNamespace(weights=run / 'weights', threads=threads)
        parts = {}
        for period in PERIODS:
            origins, prediction = runner.predict_period(table, period, seed, args, logs)
            with np.load(run / f'{period}_predictions.npz', allow_pickle=False) as stored:
                if not np.array_equal(stored['origins'], origins.asi8):
                    raise ValueError(f'{seed}/{period}: replay origin mismatch')
                if not np.array_equal(stored['pred'], prediction):
                    raise ValueError(f'{seed}/{period}: replay prediction mismatch')
            parts[period] = origins, prediction
            replay.append(dict(seed=seed, period=period, max_abs_error=0.0))
            dest = HERE / 'weights' / period / f's{seed}.npz'
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run / 'weights' / period / f's{seed}.npz', dest)
        frame = runner.prediction_frame(parts)
        stored_frame = pd.read_parquet(run / 'results' / f's{seed}' / 'predictions.parquet')
        pd.testing.assert_frame_equal(frame, stored_frame, check_exact=True)
        result = runner.score(frame, observations)
        old = json.loads((run / 'results' / f's{seed}' / 'score.json').read_text())['competition']
        for scope in ('total', 'f1', 'f2', 'f3'):
            for metric in ('s', 'rmse', 'timse'):
                if not np.isclose(result['competition'][scope][metric], old[scope][metric], atol=1e-10, rtol=0):
                    raise ValueError(f'{seed}/{scope}: official score changed')
        runner.write_results(HERE / 'results' / f's{seed}', frame, result)
        scores[str(seed)] = result['competition']
        frames.append(frame)
    del table
    mean = runner.mean3(frames)
    result = runner.score(mean, observations)
    runner.write_results(HERE / 'results' / 'mean3', mean, result)
    values = np.array([scores[str(s)]['total']['s'] for s in SEEDS])
    verification = dict(status='passed', seeds=list(SEEDS), replay=replay,
                        fixed_equal_weights=True, prediction_rows=len(mean),
                        seed_mean_s=float(values.mean()), seed_sd_s=float(values.std(ddof=1)),
                        seed_worst_s=float(values.max()), mean3=result['competition'])
    write(HERE / 'verification.json', verification)
    return metadata, verification


def make_meta(runner, metadata):
    parameters = dict(metadata[42, 'F1']['parameters'])
    parameters.pop('seed', None)
    weighted_seconds = sum(metadata[s, p]['query_origins'] * metadata[s, p]['inference_seconds_per_fold']
                           for s in SEEDS for p in PERIODS)
    origins = sum(metadata[42, p]['query_origins'] for p in PERIODS)
    config = dict(data=dict(source='data-v2/input/df.parquet',
                           feature_policy='causal feature ffill; training labels ffill on target time',
                           features=dict(constant=runner.CONSTANT, varying=runner.VARYING),
                           expanded_feature_count=215, training_maturity_hours=144, periods=list(PERIODS)),
                  hyperparameters=dict(objective='unweighted_mse', tail_weighted_mse=False, spectral=False,
                      strategy='3 blocks x 24 leads, multi_output_tree', parameters=parameters, rounds=900,
                      seeds=list(SEEDS), seed_aggregation='fixed_equal_mean',
                      gradient='prediction-target', hessian=1,
                      offset=dict(scope='per_lead', alpha=.25, window=336,
                                  source='prior residuals on pre-cutoff calibration training rows'), clip=[1, 3000]))
    meta = dict(team_member='seongeun', experiment=NAME,
                description='Fresh unweighted-MSE-only 900-round offset-roll replacement ablation; '
                            'no tail weights, spectral term or direct-S objective. Fixed equal mean of seeds '
                            '42/43/44; 31 source features and training-only prior offset retained. '
                            'Published on explicit user request, independently of adoption gates.',
                no_future_leakage=True, config=config,
                inference_seconds_per_fold=weighted_seconds / origins,
                architecture_url=f'https://github.com/TLAB-HGU/sunrun-leaderboard/blob/{git("rev-parse", "HEAD")}'
                                 '/research/vanila_mse/experiment.py')
    for period in PERIODS:
        meta[f'{period.lower()}_training_data_end_utc'] = runner.CUTOFFS[period] + 'T00:00:00Z'
    return meta


def replay_script(submit, meta):
    relative = submit.replay_script_path('seongeun', NAME, meta['config'])
    text = '''"""Reproduce vanila_mse: fresh pure MSE models, fixed seeds 42/43/44 mean."""
import argparse, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--observations',required=True)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--threads',type=int,default=6)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    work=a.out/'run'
    subprocess.run([sys.executable,str(ROOT/'research/vanila_mse/experiment.py'),'--data',a.data,
                    '--observations',a.observations,'--out',str(work),'--threads',str(a.threads)],
                   check=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),cwd=ROOT)
    shutil.copy(work/'results/mean3/predictions.parquet',a.out/'predictions.parquet')
'''
    paths = [ROOT / relative, ROOT / 'leaderboard' / relative]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return paths


def readme(verification, published=False):
    status = '게시 완료' if published else '게시 준비'
    metrics = verification['mean3']
    rows = '\n'.join(f"| {p} | {metrics[p.lower()]['s']:.6f} | {metrics[p.lower()]['timse']:.6f} | "
                     f"{metrics[p.lower()]['rmse']:.6f} |" for p in ('Total', 'F1', 'F2', 'F3'))
    return f'''# vanila_mse

2026-11-12 사용자 요청에 따른 {status}. 31 source / 215 expanded 특징의 offset-roll 구조에서
손실만 순수 MSE(gradient=prediction-target, Hessian=1)로 바꿔 새로 학습했다.
Tail 가중치, spectral 항, direct-S 항은 없다. 각 블록 900 rounds, seed 42/43/44,
학습 cutoff·144h maturity·prior/full 분리·training-only frozen offset·clip은 유지했다.

세 seed 예측의 고정 등가중 평균이며 공식 평가 정답으로 계수나 가중치를 적합하지 않았다.

| 구간 | S | TIMSE | RMSE |
|---|---:|---:|---:|
{rows}

개별 seed 점수 평균 {verification['seed_mean_s']:.6f}, 표본 SD {verification['seed_sd_s']:.6f},
최악 {verification['seed_worst_s']:.6f}. 위 표의 mean3 예측 점수와 구분한다.
사용자 요청 게시이며, 게시 자체는 성능 개선 판정이나 기준선 승격이 아니다.

모든 seed 학습 exit 0, 9개 seed×기간 저장 모델 예측 재생성 오차 0,
원 parquet와 비트 단위 동일, 공식 scorer 재채점 일치, 누락 대체 0을 확인했다.
상세 검증은 verification.json, 실행 metadata와 로그는 runs/mse_s{{42,43,44}}에 있다.
weights는 로컬에 보관하고 Git에는 올리지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/vanila_mse/experiment.py --threads 6 --out /tmp/vanila-mse-retrain
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/vanila_mse/experiment.py --predict-only --weights research/vanila_mse/weights --out /tmp/vanila-mse-replay
```
'''


def commit_files(paths, message):
    if git('diff', '--cached', '--name-only'):
        raise RuntimeError('Existing staged changes; refusing to commit another task\'s index')
    relative = [str(p.relative_to(ROOT)) for p in paths if p.is_file()]
    if not relative:
        return
    git('add', '-f', '--', *relative)
    if git('diff', '--cached', '--name-only'):
        git('commit', '-m', message)


def sync_and_push():
    git('fetch', 'origin', 'main')
    # publish_plots creates one known local asset while committing its remote copy.
    # Move only this task's untracked copy before integrating the remote commit.
    for path in (ROOT / 'leaderboard/assets/metrics').glob('seongeun--vanila_mse--*.png'):
        if not git('ls-files', '--', str(path.relative_to(ROOT))):
            folder = Path(tempfile.mkdtemp(prefix='vanila-mse-asset-'))
            shutil.move(str(path), str(folder / path.name))
    git('merge', '--no-edit', 'origin/main')
    git('push', 'origin', 'HEAD:main')


def existing_record(submit, api, meta):
    relative = Path(submit.replay_script_path('seongeun', NAME, meta['config']))
    result_path = f'results/seongeun/{NAME}/{relative.stem}.json'
    files = api.list_repo_files(submit.DATASET, repo_type='dataset')
    if result_path not in files:
        return None
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(submit.DATASET, result_path, repo_type='dataset', force_download=True)
    record = json.loads(Path(path).read_text())
    if record['config'] != meta['config']:
        raise ValueError('Existing submission has a different config')
    return record


def verify_live(submit, api, record):
    from huggingface_hub import hf_hub_download
    revision = api.repo_info(submit.SPACE, repo_type='space').sha
    path = hf_hub_download(submit.SPACE, 'index.html', repo_type='space', revision=revision)
    page = Path(path).read_text()
    rows = re.findall(r'<tr\b[^>]*>.*?</tr>', page, re.S)
    matches = [row for row in rows if re.search(r'(?:data-experiment="vanila_mse"|>vanila_mse<)', row)]
    if len(matches) != 1:
        raise ValueError(f'Expected one visible vanila_mse row, got {len(matches)}')
    row = matches[0]
    if '>architecture<' not in row or record['architecture_url'] not in row:
        raise ValueError('Live row lacks the pinned architecture link')
    if f"{record['competition']['total']['s']:.2f}" not in row:
        raise ValueError('Live row displays a different score')
    missing = submit.board.script_cells(page)
    if missing:
        raise ValueError(f'{len(missing)} rows lack architecture links')
    return dict(experiment=NAME, space_revision=revision, displayed_once=True,
                displayed_total_s=record['competition']['total']['s'], architecture_link_present=True,
                rows_missing_architecture=0, submitted_utc=record['submitted_utc'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data-v2/input/df.parquet')
    parser.add_argument('--observations', type=Path)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--publish', action='store_true', help='commit/push code and publish the authorized result')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run with CUDA_VISIBLE_DEVICES=""')
    runner = load_runner()
    observations = runner.observations_path(args.observations)
    metadata, verification = prepare(runner, args.data, observations, args.threads)
    sys.path.insert(0, str(ROOT / 'leaderboard'))
    import submit
    meta = make_meta(runner, metadata)
    wrappers = replay_script(submit, meta)
    write(HERE / 'submission-meta.json', meta)
    (HERE / 'README.md').write_text(readme(verification))
    dry = submit.submit(str(HERE / 'results/mean3/predictions.parquet'), meta, dry_run=True)
    if not np.isclose(dry['competition']['total']['s'], verification['mean3']['total']['s'], atol=1e-10, rtol=0):
        raise ValueError('Live official observations produce a different score')
    if not args.publish:
        print(json.dumps(dict(status='prepared', total_s=dry['competition']['total']['s'])))
        return
    source_files = [HERE / name for name in ('experiment.py', 'publish_results.py', 'test_experiment.py',
                    'README.md', 'strategy.md', 'verification.json', 'registry.json', 'manifest.json')]
    result_files = list((HERE / 'results').glob('*/*'))
    # Training metadata and exit receipts are small reproducibility artifacts, unlike weights.
    receipts = [p for run in (HERE / 'runs').glob('mse_s*') for p in run.glob('F*_metadata.json')]
    receipts += [p for run in (HERE / 'runs').glob('mse_s*') for p in (run / 'train.exit', run / 'launch.json')]
    commit_files(source_files + result_files + receipts + wrappers, 'Publish vanila_mse architecture and verified mean3 forecasts')
    sync_and_push()
    meta['architecture_url'] = (f'https://github.com/TLAB-HGU/sunrun-leaderboard/blob/{git("rev-parse", "HEAD")}'
                                '/research/vanila_mse/experiment.py')
    write(HERE / 'submission-meta.json', meta)
    from huggingface_hub import HfApi
    api = HfApi()
    record = existing_record(submit, api, meta)
    if record is None:
        record = submit.submit(str(HERE / 'results/mean3/predictions.parquet'), meta)
    else:
        if not np.isclose(record['competition']['total']['s'], dry['competition']['total']['s'], atol=1e-10, rtol=0):
            raise ValueError('Stored submission score differs; refusing replacement')
        entry = submit.architecture_code.entry('seongeun', NAME, record['config_sha256'], record['architecture_url'])
        record['_leaderboard'] = submit.publish(api, [], submit.SPACE,
                                                architecture={'models': [entry]}, dataset=submit.DATASET)
    write(HERE / 'submission.json', record)
    check = verify_live(submit, api, record)
    write(HERE / 'publication-verification.json', check)
    (HERE / 'README.md').write_text(readme(verification, published=True))
    commit_files([HERE / p for p in ('submission-meta.json', 'submission.json',
                                     'publication-verification.json', 'README.md')],
                 'Record vanila_mse leaderboard publication and live verification')
    sync_and_push()
    print(json.dumps(dict(status='published', total_s=record['competition']['total']['s'],
                          page=record.get('_leaderboard')), ensure_ascii=False))


if __name__ == '__main__':
    main()
