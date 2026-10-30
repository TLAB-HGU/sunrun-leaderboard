"""Leaderboard publication of the E2000 forecasting-strategy study (team_member seongeun).

Batches: `build` releases every strategy whose F1 and F2 fits are finished and that is not yet submitted
(publication/batch.json); the other steps act on that batch. Records accumulate across batches.
Steps (each writes publication/<step>.json):
  build   releases/e2000fs/<slug>/ + scripts/seongeun/e2000fs_<slug>/<config_sha>.py in the origin/main worktree,
          replay each script and require byte-identical predictions to the notebook's combined file
  dryrun  submit.py dry-run (validate + score, uploads nothing) for every strategy
  github  after the git push: every pinned script must be served byte-identically by raw.githubusercontent.com
  submit  real submissions through the unchanged client, never --replace
  verify  read back stored predictions/results and compare hashes
The HF token is read by the shared client (HF_TOKEN or the local huggingface cache); this file never prints it.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import data
import run
import strategies as S

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get('FS_OUT', 'outputs')       # smoke runs point both at a scratch directory
PUB = HERE / os.environ.get('FS_PUB', 'publication')
CACHE = HERE / 'outputs/cache'                         # shared data cache (notebook uses the same path)
WORK = HERE / 'publication/worktree'                  # leaderboard origin/main (submit.py checks scripts here)
MEMBER = 'seongeun'
RELEASES = WORK / 'releases/e2000fs'
SOURCE_FILES = ('data.py', 'strategies.py', 'run.py', 'evaluate.py', 'make_notebook.py', 'publish.py',
                'replay_template.py', 'test_strategies.py', 'requirements.txt', 'README.md',
                'forecast_strategies_e2000.ipynb')
GRID = {'F1': ('2026-02-28T23:00:00Z', '2026-05-28T23:00:00Z', 2137),
        'F2': ('2026-05-31T23:00:00Z', '2026-09-26T23:00:00Z', 2833)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slug(name):
    return name.lower().replace('-', '_')


def experiment(name):
    return f'e2000fs_{slug(name)}'


def config_sha(config):
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def describe(name):
    sp = S.spec(name)
    if sp['kind'] == 'block':
        kind, sigma = sp['block']
        return {'kind': kind, 'sigma': sigma}
    return {'kind': 'Rectify', 'base': {'kind': sp['base'][0], 'sigma': sp['base'][1]},
            'residual': {'kind': sp['residual'][0], 'sigma': sp['residual'][1]},
            'residual_training': f'purged out-of-fold base forecasts, {S.FOLDS} contiguous folds, {S.PURGE_H}h purge'}


TEXT = {
    'Recursive': 'one shared 1-step model iterated 72 times on its own forecasts',
    'DirRec': '72 horizon models, each also fed the forecasts of earlier horizons',
    'MIMO': 'one joint multi-output tree ensemble producing all 72 horizons',
    'RecMO': 'one shared sigma-output block model iterated 72/sigma times',
    'DirMO': 'one sigma-output model per horizon block',
    'DirRecMO': 'one sigma-output model per block, fed the forecasts of earlier blocks',
    'Rectify': 'Recursive base plus per-horizon residual models',
    'RectifyMO': 'RecMO base plus DirMO residual models',
    'Stratify': 'base and residual block strategies chosen on pre-2026-02-26 validation',
}


def description(name, selection):
    family = name.split('-')[0] if name not in TEXT else name
    chosen = ''
    if name == selection['RectifyMO']:
        chosen = ' sigma chosen among {6,12,24} on the select stage.'
    if name == selection['Stratify']:
        chosen = ' Chosen among 18 base x residual x sigma{6,24} combinations on the select stage.'
    return (f'E2000 forecasting-strategy study ({name}): {TEXT[family]}.{chosen} Frozen E2000 XGBoost recipe '
            '(900 rounds, depth 3, lr 0.03, seed 42) on the 111 E2000 inputs (ACE, 27-day recurrence, SUVI '
            'strips/CH, DONKI WSA-ENLIL) re-timed to the origin hour (no ACE +2h lag), gaps forward-filled. '
            'F1 forecasts from models trained on targets before 2026-02-26; F2 from targets before 2026-05-29. '
            'Strategy choices use only validation 2025-12-04..2026-02-22; no leaderboard score was used for '
            'any choice. Disclosure: the E2000 feature set and recipe were inherited from earlier research '
            'whose own selection used later data, so pre-2026-02-26 tuning provenance of those inherited '
            'choices is not proven; single seed. Script replays the hash-pinned archived forecast.')


def _load(name, default):
    path = PUB / name
    return json.loads(path.read_text()) if path.exists() else default


def stage_meta(name):
    canon = run.canonical(name)
    return {st: OUT / 'predictions' / st / f'{tag}_{canon}.json' for st, tag in (('f1', 'f1'), ('final', 'official'))}


def ready(selection):
    """Strategies with both F1 and F2 fits and the combined file, not yet submitted (in notebook order)."""
    done = _load('submission.json', {})
    return [n for n in selection['final'] if n not in done
            and all(p.exists() for p in stage_meta(n).values())
            and (OUT / 'predictions/combined' / f'{n}.parquet').exists()]


def copy_source(dest):
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for f in SOURCE_FILES:
        shutil.copyfile(HERE / f, dest / f)
        hashes[f] = sha(HERE / f)
    for name, path in (('data-receipt.json', CACHE / 'receipt.json'), ('selection.json', OUT / 'selection.json')):
        shutil.copyfile(path, dest / name)
        hashes[name] = sha(path)
    return hashes


def build(args):
    """Release every ready strategy (or --names); records this batch in batch.json."""
    import evaluate
    selection = json.loads((OUT / 'selection.json').read_text())
    receipt = json.loads((CACHE / 'receipt.json').read_text())
    names = args.names or ready(selection)
    if not names:
        raise SystemExit('no strategy is ready (needs F1 + F2 fits and the combined file)')
    scorer = evaluate.Official()
    cfg = S.recipe()
    params = {k: v for k, v in cfg.params().items() if k != 'n_jobs'}
    built = _load('build.json', {})
    for name in names:
        rel = RELEASES / slug(name)
        rel.mkdir(parents=True, exist_ok=True)
        combined_path = OUT / 'predictions/combined' / f'{name}.parquet'
        combined = pd.read_parquet(combined_path)
        origins = pd.DatetimeIndex(pd.to_datetime(combined.origin_last_input_utc.iloc[::72], utc=True))
        values = combined.pred_kms.to_numpy(np.float64).reshape(-1, 72)
        np.savez(rel / 'forecast.npz', origins_ns=origins.as_unit('ns').asi8, predictions=values)
        meta_by_stage = {st: json.loads(p.read_text()) for st, p in stage_meta(name).items()}
        # Per-release copy: a later batch must not change files whose hashes an earlier config records.
        source_hashes = copy_source(rel / 'source')
        config = {
            'data': {
                'study': 'analysis/forecast_strategies_e2000 (E2000 forecasting-strategy comparison, Direct excluded)',
                'inputs': {k: v['sha256'] for k, v in receipt['inputs'].items()},
                'feature_count': 111, 'ace_lag_hours': 0, 'prefill': 'forward fill along the origin timeline',
                'sources': ['ACE real-time solar wind/plasma/magnetic/EPAM/SIS (hourly)', 'ACE 27-day recurrence',
                            'SUVI longitude strips and coronal-hole areas', 'DONKI WSA-ENLIL runs completed <= origin'],
                'targets': 'hourly ACE filled_speed_kms (the leaderboard truth series), absolute km/s',
                'f1_model_training_target_end_utc': '2026-02-26T00:00:00Z',
                'f2_model_training_target_end_utc': '2026-05-29T00:00:00Z',
                'selection': {'fit_target_end_utc': str(data.SELECT_TRAIN_END),
                              'validation_origins_utc': [str(data.SELECT_VALID_START), str(data.SELECT_VALID_END)],
                              'metric': 'observed-row MSE', 'chosen': {k: selection[k] for k in ('RectifyMO', 'Stratify')}},
                'periods': {p: {'origins_start_utc': a, 'origins_end_utc': b, 'origin_count': n}
                            for p, (a, b, n) in GRID.items()},
                'asset': {'filename': 'forecast.npz', 'sha256': sha(rel / 'forecast.npz')},
                'source_sha256': source_hashes,
                'source_note': 'source/ snapshot taken while the study notebook was still executing; the '
                               'notebook copy is the unexecuted driver',
                'fit_origins': {st: m['n_fit_origins'] for st, m in meta_by_stage.items()},
            },
            'hyperparameters': {
                'strategy': name, 'structure': describe(name), 'xgboost': params,
                'multi_output': 'multi_strategy=multi_output_tree for blocks with sigma > 1',
                'lag_window_hours': data.LAG, 'shared_models_lead_feature': True,
                'teacher_forcing': 'observed (filled) speeds in training, own forecasts at inference',
                'seed': 42,
            },
        }
        csha = config_sha(config)
        write_json(rel / 'config.json', config)
        script = WORK / 'scripts' / MEMBER / experiment(name) / f'{csha}.py'
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text((HERE / 'replay_template.py').read_text().replace('__CONFIG_SHA__', csha)
                          .replace('__SLUG__', slug(name)))
        n1, n2 = GRID['F1'][2], GRID['F2'][2]
        timing = (meta_by_stage['f1']['inference_seconds_per_fold'] * n1
                  + meta_by_stage['final']['inference_seconds_per_fold'] * n2) / (n1 + n2)
        meta = {'team_member': MEMBER, 'experiment': experiment(name), 'description': description(name, selection),
                'no_future_leakage': True, 'f1_training_data_end_utc': '2026-02-26T00:00:00Z',
                'config': config, 'inference_seconds_per_fold': timing}
        write_json(rel / 'meta.yaml', meta)  # JSON is valid YAML
        write_json(rel / 'inference-timing.json', {
            'inference_seconds_per_fold': timing, 'scope': 'model predict call on prepared inputs only; '
            'excludes loading, training, feature preparation and output writing; F1/F2 origin-weighted mean',
            'per_stage': {st: m['inference_seconds_per_fold'] for st, m in meta_by_stage.items()}})
        replay = PUB / 'replay' / f'{slug(name)}.parquet'
        subprocess.run([sys.executable, str(script), '--output', str(replay)], check=True, cwd=WORK)
        if not pd.read_parquet(replay).equals(combined):
            raise SystemExit(f'{name}: replay does not reproduce the notebook forecast')
        row, _ = scorer.row(combined_path)
        built[name] = {'experiment': experiment(name), 'config_sha256': csha, 'script': str(script.relative_to(WORK)),
                       'release': str(rel.relative_to(WORK)), 'predictions': str(replay),
                       'prediction_sha256': sha(replay), 'meta': str(rel / 'meta.yaml'), 'notebook_mse': row['mse'],
                       'notebook_mse_f1': row['mse_f1'], 'notebook_mse_f2': row['mse_f2'],
                       'notebook_mse550': row['mse550']}
        print(f'{name:30s} built  MSE={row["mse"]:.1f}  F1={row["mse_f1"]:.1f}  F2={row["mse_f2"]:.1f}  '
              f'>=550={row["mse550"]:.1f}', flush=True)
    write_json(PUB / 'build.json', built)
    write_json(PUB / 'batch.json', names)


def batch():
    built = json.loads((PUB / 'build.json').read_text())
    return {n: built[n] for n in json.loads((PUB / 'batch.json').read_text())}


def client():
    sys.path.insert(0, str(WORK))
    import submit
    return submit


def dryrun(args):
    built = batch()
    submit = client()
    out = {}
    for name, b in built.items():
        r = submit.submit(b['predictions'], json.loads(Path(b['meta']).read_text()), dry_run=True)
        if not np.isclose(r['mse'], b['notebook_mse'], rtol=1e-9):
            raise SystemExit(f'{name}: dry-run MSE {r["mse"]} differs from notebook {b["notebook_mse"]}')
        out[name] = {k: r[k] for k in ('experiment', 'config_sha256', 'mse', 'mse_f1', 'mse_f2', 'mse_ge550',
                                       'mse_f1_ge550', 'mse_f2_ge550', 'mse_observed', 'skill_vs_naive')}
        print(f'{name:30s} dry-run ok  MSE={r["mse"]:.1f}  F1={r["mse_f1"]:.1f}  F2={r["mse_f2"]:.1f}  '
              f'>=550={r["mse_ge550"]:.1f}', flush=True)
    write_json(PUB / 'dryrun.json', _load('dryrun.json', {}) | out)


def github(args):
    built = batch()
    submit = client()
    out = {}
    for name, b in built.items():
        _, raw, page = submit.script_location(MEMBER, b['experiment'], b['config_sha256'])
        with urllib.request.urlopen(raw, timeout=30) as r:
            remote = r.read()
        out[name] = {'url': page, 'identical': remote == (WORK / b['script']).read_bytes()}
        if not out[name]['identical']:
            raise SystemExit(f'{name}: GitHub script differs from the local pinned script')
    write_json(PUB / 'github.json', _load('github.json', {}) | out)
    print(f'{len(out)} pinned scripts served byte-identically from GitHub main')


def submit_all(args):
    built = batch()
    submit = client()
    done = _load('submission.json', {})
    for name, b in built.items():
        if name in done:
            continue
        r = submit.submit(b['predictions'], json.loads(Path(b['meta']).read_text()), dry_run=False, replace=False)
        done[name] = {k: r.get(k) for k in ('team_member', 'experiment', 'config_sha256', 'submitted_utc', 'mse',
                                            'mse_f1', 'mse_f2', 'mse_ge550', 'mse_f1_ge550', 'mse_f2_ge550', 'code_url')}
        write_json(PUB / 'submission.json', done)
        print(f'{name:30s} submitted  MSE={r["mse"]:.1f}', flush=True)


def verify(args):
    from huggingface_hub import HfApi, hf_hub_download
    built = json.loads((PUB / 'build.json').read_text())
    done = json.loads((PUB / 'submission.json').read_text())
    api = HfApi()
    out = _load('verify.json', {'records': {}})
    out['dataset_revision'] = api.dataset_info(client().DATASET).sha
    for name in batch():
        d = done[name]
        key = f"{d['team_member']}/{d['experiment']}/{d['config_sha256']}"
        stored = hf_hub_download(client().DATASET, f'submissions/{key}.parquet', repo_type='dataset', force_download=True)
        rec = json.loads(Path(hf_hub_download(client().DATASET, f'results/{key}.json', repo_type='dataset',
                                              force_download=True)).read_text())
        same = pd.read_parquet(stored).equals(pd.read_parquet(built[name]['predictions']))
        out['records'][name] = {'key': key, 'predictions_identical': same, 'remote_mse': rec['mse'],
                                'mse_matches': abs(rec['mse'] - d['mse']) < 1e-9}
        if not (same and out['records'][name]['mse_matches']):
            raise SystemExit(f'{name}: remote record does not match')
    write_json(PUB / 'verify.json', out)
    print(f"{len(batch())} remote records verified ({len(out['records'])} total) at dataset revision "
          f"{out['dataset_revision']}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('build', 'dryrun', 'github', 'submit', 'verify'))
    ap.add_argument('--names', nargs='+', help='build only these strategies (default: every ready one)')
    a = ap.parse_args()
    {'build': build, 'dryrun': dryrun, 'github': github, 'submit': submit_all, 'verify': verify}[a.step](a)
