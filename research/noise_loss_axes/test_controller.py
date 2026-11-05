import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('noise_controller', HERE/'controller.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def archive_file(path, entries):
    with tarfile.open(path, 'w:gz') as archive:
        for name, content, kind in entries:
            member = tarfile.TarInfo(name)
            if kind == 'file':
                data = content.encode()
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            else:
                member.type = tarfile.SYMTYPE
                member.linkname = content
                archive.addfile(member)


@pytest.mark.parametrize('name,kind', [('../escape', 'file'), ('/tmp/escape', 'file'),
    ('results/mixture/seed42/run.json', 'file'), ('results/gaussian/seed42/link', 'link'),
    ('results/gaussian/seed42/../../plain_l2/seed42/a', 'file'), ('logs/unrelated.log', 'file')])
def test_secure_extract_rejects_unsafe_member(tmp_path, name, kind):
    archive = tmp_path/'bad.tar.gz'
    archive_file(archive, [(name, 'outside', kind)])
    with pytest.raises(ValueError, match='Unsafe'):
        C.secure_extract(archive, tmp_path/'extract')
    assert not list((tmp_path/'extract').rglob('*'))


def test_secure_extract_and_conflict_free_install(tmp_path):
    archive = tmp_path/'safe.tar.gz'
    archive_file(archive, [('results/gaussian/seed42/run.json', '{}', 'file'),
                           ('logs/lognormal_seed42.log', 'ok', 'file')])
    staging, root = tmp_path/'staging', tmp_path/'root'
    assert C.secure_extract(archive, staging) == 2
    other = root/'results/sera_l2/seed42/keep'
    other.parent.mkdir(parents=True)
    other.write_text('untouched')
    C.install_staged(staging, root)
    C.install_staged(staging, root)
    assert other.read_text() == 'untouched'
    (root/'logs/lognormal_seed42.log').write_text('different')
    with pytest.raises(ValueError, match='conflicts'):
        C.install_staged(staging, root)
    assert (root/'logs/lognormal_seed42.log').read_text() == 'different'


def make_complete_seed(root, arm='gaussian', seed=42):
    source = root/'experiment.py'
    source.write_text((HERE/'experiment.py').read_text())
    out = root/'results'/arm/f'seed{seed}'
    out.mkdir(parents=True)
    identity = {'arm': arm, 'seed': seed, 'rounds': 300, 'smoke': False,
                'maturity_hours': 144, 'target_scale': 500.0, 'sigma_floor': .02,
                'train_cutoff': '2025-06-29 00:00:00+00:00',
                'validation_start': '2025-07-01 00:00:00+00:00',
                'validation_end': '2025-10-01 00:00:00+00:00'}
    origins = pd.date_range('2025-07-01T00:00Z', '2025-09-27T18:00Z', freq='6h')
    y = np.tile(400+np.arange(72), (len(origins), 1)).astype(float)
    y[0, 0] = np.nan
    pred = np.nan_to_num(y, nan=400)+1
    frame = pd.DataFrame({'origin': np.repeat(origins, 72), 'h': np.tile(np.arange(1, 73), len(origins)),
                          'pred_kms': pred.ravel(), 'y': y.ravel()})
    frame.to_parquet(out/'predictions.parquet')
    n_pairs = int(np.isfinite(y).sum())
    metrics = {'identity': identity, 'arm': arm, 'seed': seed, 'W': 1., 'mae': 1., 'rmse': 1., 'bias': 1.,
               'n_pairs': n_pairs, 'by_lead': [{'lead': h, 'n': len(origins)-(h == 1)} for h in range(1, 73)],
               'distribution': {'nll': 5., 'coverage50': .5, 'coverage90': .9, 'coverage95': .95, 'pit_hist': [1]*10}}
    C.write_json(out/'run.json', identity)
    C.write_json(out/'metrics.json', metrics)
    C.write_json(out/'complete.json', {'status': 'complete', 'exit_code': 0, 'identity': identity, 'rows': len(frame)})
    (out/'process.exit').write_text('0\n')
    (out/'source.py').write_text(source.read_text())
    for block in range(3):
        C.write_json(out/f'b{block}.json', {'identity': identity, 'block': block, 'feature_count': 237,
                                          'features': C.expected_features(source, block)})
        C.write_json(out/f'b{block}.loss.json', {'train': {'train_nll_kms': [5.]*300}})
        (out/f'b{block}.ubj').write_bytes(b'model-fixture')
        sl = slice(block*24, (block+1)*24)
        np.savez(out/f'b{block}.npz', origins=origins.asi8, pred=pred[:, sl], truth=y[:, sl],
                 theta=np.zeros((len(origins), 24, 2)))
    return out, source


def test_validates_complete_run_without_remote_actions(tmp_path):
    out, source = make_complete_seed(tmp_path)
    result = C.validate_seed(tmp_path, 'gaussian', 42, source)
    assert result['rows'] == 356*72
    assert result['n_pairs'] == 356*72-1
    (out/'process.exit').write_text('1\n')
    with pytest.raises(ValueError, match='process exit'):
        C.validate_seed(tmp_path, 'gaussian', 42, source)


@pytest.mark.parametrize('damage', ['rounds', 'smoke', 'features', 'source', 'nonfinite', 'missing_model'])
def test_rejects_invalid_completion(tmp_path, damage):
    out, source = make_complete_seed(tmp_path)
    if damage in ('rounds', 'smoke'):
        value = json.loads((out/'run.json').read_text())
        value[damage] = 2 if damage == 'rounds' else True
        C.write_json(out/'run.json', value)
    elif damage == 'features':
        value = json.loads((out/'b0.json').read_text())
        value['features'][0] = 'replacement'
        C.write_json(out/'b0.json', value)
    elif damage == 'source':
        (out/'source.py').write_text('changed')
    elif damage == 'nonfinite':
        value = json.loads((out/'metrics.json').read_text())
        value['W'] = float('nan')
        (out/'metrics.json').write_text(json.dumps(value))
    else:
        (out/'b0.ubj').unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        C.validate_seed(tmp_path, 'gaussian', 42, source)


def test_probe_parser_and_script_compile():
    probe = C.remote_probe_source()
    compile(probe, '<probe>', 'exec')
    assert C.REMOTE_ROOT in probe and C.REMOTE_ARCHIVE in probe
    assert C.parse_probe('CLI chatter\nNOISE_STATUS={"boot_id":"a"}\nfinished')['boot_id'] == 'a'
    with pytest.raises(RuntimeError, match='NOISE_STATUS'):
        C.parse_probe('No valid status')


def test_local_progress_marks_failure_without_completion(tmp_path):
    out = tmp_path/'results/mixture/seed42'
    out.mkdir(parents=True)
    (out/'process.exit').write_text('9')
    progress, failures = C.local_progress(tmp_path)
    assert failures == ['mixture/42: exit=9']
    assert progress['mixture']['seeds']['42']['complete'] is False


def test_monitor_does_not_stop_after_bad_archive(tmp_path, monkeypatch):
    (tmp_path/'experiment.py').write_text((HERE/'experiment.py').read_text())
    calls = []
    payload = {'boot_id': 'runtime-a', 'arms': {a: {'exit': 0, 'seeds': {str(s): {'exit': 0, 'complete': True}
                              for s in C.SEEDS}} for a in C.REMOTE_ARMS},
               'archive': {'path': C.REMOTE_ARCHIVE, 'bytes': 3}}
    def fake_command(command, log, timeout=240):
        calls.append(command)
        if command[1] == 'exec':
            return 'NOISE_STATUS='+json.dumps(payload)
        if command[1] == 'download':
            Path(command[-1]).write_bytes(b'bad')
            return ''
        raise AssertionError('Stop must not occur before valid artifact retrieval')
    monkeypatch.setattr(C, 'run_command', fake_command)
    # Invalid archive is an artifact failure, never permission to stop a runtime.
    assert C.monitor(tmp_path, C.SESSION, 60, once=True) == 2
    assert json.loads((tmp_path/'status.json').read_text())['status'] == 'failed'
    assert not any(command[1] == 'stop' for command in calls)


def test_monitor_rejects_reused_session_identity(tmp_path, monkeypatch):
    evidence = tmp_path/'evidence'
    C.write_json(evidence/'remote_boot_identity.json', {'session': C.SESSION, 'boot_id': 'original'})
    monkeypatch.setattr(C, 'run_command', lambda *a, **kw: 'NOISE_STATUS={"boot_id":"replacement","arms":{},"archive":null}')
    assert C.monitor(tmp_path, C.SESSION, 60, once=True) == 2
    state = json.loads((tmp_path/'status.json').read_text())
    assert state['status'] == 'failed'
    assert 'identity changed' in state['last_tool_error']
