import json
import pytest
from experiments.e0472_ablation_hpo import campaign as c


def test_budget_reserves_last_24_hours():
    state = dict(started_at=100, max_hours=72)
    assert c.remaining_seconds(state, 100, exploration=True) == 48 * 3600
    assert c.remaining_seconds(state, 100 + 48 * 3600, exploration=True) == 0
    assert c.remaining_seconds(state, 100 + 72 * 3600) == 0


@pytest.mark.parametrize('kind,cap', [('xgb', 60), ('analog', 24), ('joint', 20)])
def test_trial_limits(kind, cap):
    assert c.trial_available(cap - 1, kind)
    assert not c.trial_available(cap, kind)
    assert not c.trial_available(-1, kind)


def test_review_not_implicit_self_approved_or_reused():
    cfg, hashes = {'name': 'candidate'}, {'runner.py': 'abc'}
    receipt = dict(reviewer='independent-critic', independent=True, round=1,
                   verdict='accept', high_findings=[], notes='Reviewed causal time splits.',
                   **c.attestable(cfg, hashes))
    receipt['signature'] = c.digest(receipt)
    assert c.review_valid(receipt, cfg, hashes, 'owner')
    assert not c.review_valid({}, cfg, hashes, 'owner')
    assert not c.review_valid(receipt, cfg, hashes, 'independent-critic')
    assert not c.review_valid(receipt, cfg, {'runner.py': 'changed'}, 'owner')
    assert not c.review_valid(receipt, {'name': 'changed'}, hashes, 'owner')
    receipt['notes'] += ' altered'
    assert not c.review_valid(receipt, cfg, hashes, 'owner')


def test_poison_needs_positive_control():
    with pytest.raises(RuntimeError):
        c.require_poison({'passed': True})
    with pytest.raises(RuntimeError):
        c.require_poison({'passed': False, 'positive_control': True})
    c.require_poison({'passed': True, 'positive_control': True})


def test_smoke_never_implicitly_permits_official(tmp_path, monkeypatch):
    controller = c.Campaign(c.parser().parse_args(['--run', '--root', str(tmp_path)]))
    controller.state = dict(started_at=c.time.time(), max_hours=72, n_jobs=2,
                            configs={'baseline': {'config': c.DEFAULT_CONFIG}},
                            units={'baseline:smoke:CAL': {'status': 'complete'}}, locked=None)
    monkeypatch.setattr(c, 'sun_cpu_usage', lambda: (0, []))
    monkeypatch.setattr(controller, 'ensure_review', lambda key: True)
    with pytest.raises(RuntimeError, match='frozen'):
        controller.run_unit('baseline', 'official', ['official'], [0, 1, 2])


def make_process(root, pid, ppid, cmd, env=''):
    entry = root / str(pid)
    entry.mkdir()
    (entry / 'cmdline').write_bytes(cmd.encode().replace(b' ', b'\0'))
    (entry / 'environ').write_bytes(env.encode())
    (entry / 'status').write_text(f'PPid:\t{ppid}\n')
    (entry / 'cwd').symlink_to('/home/test/sunrun/leaderboard')


def test_resource_detection_leaf_and_unknown(tmp_path):
    make_process(tmp_path, 100, 1, 'python experiments/wrapper.py --n-jobs 2')
    make_process(tmp_path, 101, 100, 'python experiments/train.py --n-jobs 2', 'OMP_NUM_THREADS=2')
    assert c.sun_cpu_usage(tmp_path, own_pid=500)[0] == 2
    make_process(tmp_path, 200, 1, 'python experiments/unknown.py')
    assert c.sun_cpu_usage(tmp_path, own_pid=500)[0] == 12


def test_fixed_ablation_and_analog_design():
    variants = c.fixed_configs(c.GROUPS, c.ENLIL_COLUMNS)
    assert len([v for v in variants if v['name'].startswith('analog_')]) == 24
    assert len([v for v in variants if v['drop_columns']]) == 6
    assert variants[0]['name'] == 'baseline'
    for cfg in variants:
        assert cfg['xgb'] == c.DEFAULT_CONFIG['xgb']


def test_dry_run_does_not_create_state(tmp_path, capsys):
    c.main(['--dry-run', '--root', str(tmp_path / 'new')])
    assert not (tmp_path / 'new').exists()
    assert json.loads(capsys.readouterr().out)['limits']['xgb'] == 60


def paired_records(mse=100., f1=.3):
    return [dict(window=w, seed=seed, metrics=dict(mse=mse + seed, f1=f1,
                 key_sha256=w, mask_sha256='same')) for w in ('W0', 'W1') for seed in (0, 1, 2)]


def test_confirmation_uses_all_six_records_and_requires_digests():
    base = paired_records()
    candidate = paired_records(90)
    assert c.paired_confirmation(candidate, base)['pass']
    candidate[-1]['metrics']['mse'] = 150
    assert not c.paired_confirmation(candidate, base)['pass']
    candidate = paired_records(90)
    for row in candidate + base:
        row['metrics'].pop('key_sha256')
    assert not c.paired_confirmation(candidate, base)['pass']


def test_owned_child_timeout_kills_only_its_group(monkeypatch):
    calls = []
    class Child:
        pid = 321
        def wait(self, timeout=None):
            calls.append(('wait', timeout))
    monkeypatch.setattr(c.os, 'killpg', lambda pid, sig: calls.append((pid, sig)))
    c.terminate_owned_group(Child())
    assert calls == [(321, c.signal.SIGTERM), ('wait', 5)]


def test_adoption_ignores_polluted_gate_and_uses_explicit_reference():
    cells = {k: 200 for k in ('low_low_high', 'low_high_high', 'high_low_high', 'high_high_high')}
    reference = dict(summary=dict(n_seeds=3, mse_seedmean=100, mse_by_seed=[99,100,101],
                                  f1_seedmean=.3, cells_seedmean=cells))
    candidate = dict(summary=dict(n_seeds=3, mse_seedmean=90, mse_by_seed=[89,90,91],
                                  f1_seedmean=.3, cells_seedmean=dict(cells)), gate={'pass': False})
    assert c.official_verdict(candidate, reference)['verdict'] == 'adoptable_improvement'
    candidate['summary']['cells_seedmean']['low_low_high'] = 201
    assert c.official_verdict(candidate, reference)['verdict'] == 'local_improvement'


def test_search_gain_cannot_override_scientific_cal_rejection(tmp_path):
    controller = c.Campaign(c.parser().parse_args(['--status', '--root', str(tmp_path)]))
    controller.state = {'configs': {'baseline': {}, 'cal_worse': {}, 'cal_gain': {}, 'alias': {'alias_of': 'cal_gain'}}}
    def metric(key, stage, window):
        values = {'baseline': (1000, 2000), 'cal_worse': (1679.65, 1246.6),
                  'cal_gain': (900, 1500), 'alias': (900, 1500)}
        return dict(mse=values[key][stage == 'search'], per_lead={},
                    key_sha256=window, mask_sha256='observed')
    controller.metric = metric
    assert controller.ranked_search() == ['cal_gain']


def refresh_controller(tmp_path, monkeypatch):
    controller = c.Campaign(c.parser().parse_args(['--refresh-scheduler-code', '--root', str(tmp_path), '--n-jobs', '2', '--max-parallel', '1']))
    scheduler = str(c.Path(c.__file__).resolve().relative_to(c.LB))
    old = {scheduler: 'old', 'runner.py': 'stable'}
    new = {scheduler: 'new', 'runner.py': 'stable'}
    controller.state = dict(code_hashes=old, configs={}, units={}, official_evaluations={}, shortlist=None, locked=None)
    monkeypatch.setattr(c, 'code_hashes', lambda: new)
    return controller, old, new


def test_explicit_refresh_preserves_historical_hashes(tmp_path, monkeypatch):
    controller, old, new = refresh_controller(tmp_path, monkeypatch)
    controller.refresh_scheduler_code()
    assert controller.state['code_hashes'] == new
    assert controller.state['code_revisions'][0]['previous_code_hashes'] == old


@pytest.mark.parametrize('block', ['helper', 'running', 'interrupted', 'official', 'config'])
def test_refresh_fails_closed(tmp_path, monkeypatch, block):
    controller, old, new = refresh_controller(tmp_path, monkeypatch)
    if block == 'helper':
        new['runner.py'] = 'changed'
    elif block in ('running', 'interrupted'):
        controller.state['units']['x:smoke:CAL'] = {'status': block}
    elif block == 'official':
        controller.state['official_evaluations']['x'] = {'status': 'requested'}
    else:
        path = tmp_path / 'config.json'
        path.write_text('{"name":"changed"}')
        controller.state['configs']['x'] = {'path': str(path), 'config': {'name': 'original'}}
    with pytest.raises(RuntimeError):
        controller.refresh_scheduler_code()
    assert controller.state['code_hashes'] == old


def resource_fixture(tmp_path, monkeypatch, external=0, memory=40, load=2):
    monkeypatch.setattr(c, 'sun_cpu_usage', lambda **kw: (external, []))
    monkeypatch.setattr(c, 'machine_capacity', lambda: {'available_bytes': memory * 1024**3, 'load1': load})
    monkeypatch.setattr(c, 'experiment_processes', lambda eid: [])
    return c.CpuReservations(tmp_path / 'reservations.json')


def test_cpu_reservations_include_existing_solar_work(tmp_path, monkeypatch):
    leases = resource_fixture(tmp_path, monkeypatch, external=2)
    assert leases.acquire('first', 5)[0]
    assert not leases.acquire('second', 5)[0]
    leases.release('first')
    assert leases.acquire('second', 5)[0]


@pytest.mark.parametrize('memory,load', [(9, 2), (40, 21)])
def test_memory_and_load_gate_before_reservation(tmp_path, monkeypatch, memory, load):
    leases = resource_fixture(tmp_path, monkeypatch, memory=memory, load=load)
    assert not leases.acquire('first', 5)[0]
    assert c.read_json(leases.path) == {}


def test_reserved_eid_excluded_only_from_external_accounting(tmp_path, monkeypatch):
    leases = resource_fixture(tmp_path, monkeypatch)
    assert leases.acquire('first', 5)[0]
    leases.attach('first', 'OWN')
    seen = []
    def usage(**kwargs):
        seen.append(kwargs['exclude_experiments'])
        return 1, []
    monkeypatch.setattr(c, 'sun_cpu_usage', usage)
    assert not leases.acquire('second', 5)[0]
    assert seen == [{'OWN'}]


def test_reservation_retained_for_surviving_owned_process(tmp_path, monkeypatch):
    leases = resource_fixture(tmp_path, monkeypatch)
    leases.acquire('first', 5)
    leases.attach('first', 'OWN')
    monkeypatch.setattr(c, 'experiment_processes', lambda eid: [123])
    with pytest.raises(RuntimeError, match='retained'):
        leases.release('first')
    assert 'first' in c.read_json(leases.path)


def test_real_subprocess_overlap_two_slots_and_single_writer(tmp_path, monkeypatch):
    import threading
    controller = c.Campaign(c.parser().parse_args(['--run', '--root', str(tmp_path)]))
    controller.state = {'units': {}, 'n_jobs': 5, 'max_parallel': 2}
    controller.reservations = resource_fixture(tmp_path, monkeypatch)
    leader = threading.get_ident()
    mutations, intervals, charges = [], [], []
    original_worker = c.run_expos_process
    script = 'import json,time; t=time.time(); time.sleep(.25); print(json.dumps({"start":t,"end":time.time()}))'
    monkeypatch.setattr(c, 'run_expos_process', lambda eid, timeout, jobs:
                        original_worker(eid, timeout, jobs, command=[c.sys.executable, '-c', script]))
    def prepare(key, stage, windows, seeds):
        assert threading.get_ident() == leader
        mutations.append(('prepare', key))
        uid = f'{key}:{stage}:{"-".join(windows)}'
        admitted, _ = controller.reservations.acquire(uid, 5)
        assert admitted
        controller.reservations.attach(uid, key)
        charges.append(sum(v['jobs'] for v in c.read_json(controller.reservations.path).values()))
        record = dict(status='registered', experiment=key, reservation=uid)
        controller.state['units'][uid] = record
        return dict(uid=uid, key=key, record=record, timeout=2, n_jobs=5)
    def finish(prepared, future):
        assert threading.get_ident() == leader
        mutations.append(('finish', prepared['key']))
        intervals.append(json.loads(future.result()['stdout']))
        prepared['record']['status'] = 'complete'
        controller.reservations.release(prepared['record']['reservation'])
    monkeypatch.setattr(controller, 'prepare_unit', prepare)
    monkeypatch.setattr(controller, 'finish_unit', finish)
    monkeypatch.setattr(controller, 'event', lambda *args: None)
    specs = [(str(i), 'smoke', ['CAL'], [0]) for i in range(3)]
    assert controller.batch(specs)
    intervals.sort(key=lambda item: item['start'])
    assert intervals[1]['start'] < intervals[0]['end'], 'two actual subprocesses must overlap'
    assert intervals[2]['start'] >= min(intervals[0]['end'], intervals[1]['end'])
    assert max(charges) == 10
    assert len([x for x in mutations if x[0] == 'prepare']) == 3
    # Resume with every unit complete does not prepare/register or run again.
    assert controller.batch(specs)
    assert len(mutations) == 6


def test_worker_cannot_mutate_controller_state(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    controller = c.Campaign(c.parser().parse_args(['--status', '--root', str(tmp_path)]))
    with ThreadPoolExecutor(1) as executor:
        with pytest.raises(RuntimeError, match='leader'):
            executor.submit(controller.save).result()


def test_exploration_search_waits_for_successful_smoke(tmp_path, monkeypatch):
    controller = c.Campaign(c.parser().parse_args(['--status', '--root', str(tmp_path)]))
    metric = dict(mse=1000, key_sha256='same', mask_sha256='same', per_lead={})
    controller.state = dict(started_at=c.time.time(), max_hours=72, configs={
        'baseline': {'processed': True}, 'candidate': {'kind': 'analog'}}, units={
        'baseline:smoke:CAL': {'status': 'complete', 'result': {'records': [{'metrics': metric}]}},
        'candidate:smoke:CAL': {'status': 'running'}}, max_parallel=2)
    assert controller.exploration_ready() == []
    controller.state['units']['candidate:smoke:CAL'] = {'status': 'complete', 'result': {'records': [{'metrics': metric}]}}
    assert controller.exploration_ready() == [('candidate', 'search', ['SEARCH'], [0])]


def test_maintenance_blocks_new_dispatch(tmp_path, monkeypatch):
    controller = c.Campaign(c.parser().parse_args(['--run', '--root', str(tmp_path)]))
    controller.state = {'units': {}, 'max_parallel': 2}
    (tmp_path / '.maintenance.json').write_text('{}')
    monkeypatch.setattr(controller, 'prepare_unit', lambda *args: pytest.fail('must not register'))
    assert controller.batch([('x', 'smoke', ['CAL'], [0])]) is False
    with pytest.raises(RuntimeError, match='maintenance'):
        controller.run()


def test_resume_reconciles_completed_graph_evidence_without_refit(tmp_path, monkeypatch):
    controller = c.Campaign(c.parser().parse_args(['--resume', '--root', str(tmp_path)]))
    controller.state = {'units': {'x:smoke:CAL': {'status': 'running', 'experiment': 'EX'}}}
    evidence = {'records': []}
    monkeypatch.setattr(controller, 'verified_result', lambda record: evidence)
    monkeypatch.setattr(controller, 'cli', lambda *args, **kwargs: pytest.fail('resume must not register/refit'))
    controller.reconcile()
    assert controller.state['units']['x:smoke:CAL']['status'] == 'complete'
    assert controller.state['units']['x:smoke:CAL']['result'] == evidence


def test_resume_ambiguous_unit_blocks_instead_of_refitting(tmp_path, monkeypatch):
    controller = c.Campaign(c.parser().parse_args(['--resume', '--root', str(tmp_path)]))
    controller.state = {'units': {'x:smoke:CAL': {'status': 'running', 'experiment': 'EX'}}}
    def still_active(record):
        raise RuntimeError('experiment active')
    monkeypatch.setattr(controller, 'verified_result', still_active)
    with pytest.raises(RuntimeError, match='no duplicate fit'):
        controller.reconcile()
    assert controller.state['units']['x:smoke:CAL']['status'] == 'interrupted'


def test_confirmation_requires_search_and_scientific_cal_then_w2_barrier(tmp_path):
    controller = c.Campaign(c.parser().parse_args(['--status', '--root', str(tmp_path)]))
    metric = lambda mse: dict(mse=mse, per_lead={}, key_sha256='same', mask_sha256='same')
    controller.state = dict(shortlist=['candidate'], locked=None, units={})
    values = {('baseline', 'smoke'): metric(1000), ('baseline', 'search'): metric(1000),
              ('candidate', 'smoke'): metric(1100), ('candidate', 'search'): metric(900)}
    controller.metric = lambda key, stage, windows: values.get((key, stage))
    assert not controller.full_ready('candidate', 'confirm', ['W0', 'W1'])
    values[('candidate', 'smoke')] = metric(900)
    assert controller.full_ready('candidate', 'confirm', ['W0', 'W1'])
    assert not controller.full_ready('candidate', 'confirm', ['W2'])
    controller.state['locked'] = ['candidate']
    assert not controller.full_ready('candidate', 'confirm', ['W2'])
    controller.records = lambda key, stage, windows: [{}] * (6 if windows == 'W0-W1' else 0)
    assert controller.full_ready('candidate', 'confirm', ['W2'])
    assert not controller.full_ready('candidate', 'official', ['official'])


def test_actual_timeout_cleans_owned_subprocess_group(tmp_path):
    import time
    pidfile = tmp_path / 'child-pid'
    script = ('import subprocess,sys,time,pathlib; '
              'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
              'pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)')
    with pytest.raises(c.subprocess.TimeoutExpired):
        c.run_expos_process('TEST', -39.7, 1, command=[c.sys.executable, '-c', script, str(pidfile)])
    pid = int(pidfile.read_text())
    status = c.Path(f'/proc/{pid}/status')
    for _ in range(20):
        if not status.exists() or 'State:\tZ' in status.read_text():
            break
        time.sleep(.02)
    assert not status.exists() or 'State:\tZ' in status.read_text()


def test_thread_migration_requires_actual_control_and_preserves_deadline(tmp_path, monkeypatch):
    controller, old, new = refresh_controller(tmp_path, monkeypatch)
    controller.state.update(n_jobs=2, max_parallel=1, started_at=1234., max_hours=72)
    controller.args.n_jobs = 5
    controller.args.max_parallel = 2
    with pytest.raises(RuntimeError, match='thread-check'):
        controller.refresh_scheduler_code()
    controller.args.thread_check_eid = 'CONTROL'
    calls = []
    def verify(eid, jobs):
        calls.append((eid, jobs))
        return {'identical_predictions': True}
    monkeypatch.setattr(controller, 'verify_thread_equivalence', verify)
    controller.refresh_scheduler_code()
    assert calls == [('CONTROL', 5)]
    assert controller.state['n_jobs'] == 5 and controller.state['max_parallel'] == 2
    assert controller.state['started_at'] == 1234. and controller.state['max_hours'] == 72
    assert controller.state['code_revisions'][-1]['old_scheduling'] == {'n_jobs': 2, 'max_parallel': 1}


def thread_check_fixture(tmp_path, monkeypatch):
    import pandas as pd
    from expos import core
    controller = c.Campaign(c.parser().parse_args(['--status', '--root', str(tmp_path)]))
    controller.state = {'n_jobs': 2, 'units': {'baseline:smoke:CAL': {'experiment': 'BASE'}}}
    runs = tmp_path / 'runs'
    graph = {}
    source_files = {str((c.HERE / name).relative_to(c.LB)): c.file_hash(c.HERE / name)
                    for name in ('configs.py', 'runner.py', 'recipe.py', 'metrics.py')}
    helper = str(c.HERE / 'configs.py')
    for eid, jobs in [('BASE', 2), ('CONTROL', 5)]:
        out = runs / eid
        out.mkdir(parents=True)
        c.write_json(out / 'metrics.json', dict(key_sha256='key', mask_sha256='mask'))
        c.write_json(out / 'receipt.json', dict(stage='smoke', window='CAL', seed=0, device='cpu',
            config=c.DEFAULT_CONFIG, n_jobs=jobs, helper_code_hashes={helper: c.file_hash(helper)},
            source_access={'input': {'sha256': 'same'}}))
        c.write_json(out / 'poison.json', dict(passed=True, positive_control=True))
        pd.DataFrame({'origin_last_input_utc': pd.to_datetime(['2025-12-18'], utc=True),
                      'horizon_hours': [1], 'pred_kms': [400.]}).to_parquet(out / 'predictions_CAL_s0.parquet')
        graph[eid] = dict(code=source_files, runs=[{'rc': 0, 'outputs': {p.name: c.file_hash(p) for p in out.iterdir()}}])
    monkeypatch.setattr(core, 'RUNS', runs)
    monkeypatch.setattr(core, 'state', lambda: {'experiments': graph})
    monkeypatch.setattr(core, '_lines', lambda: [{'event': 'start', 'id': eid} for eid in graph])
    monkeypatch.setattr(c, 'experiment_processes', lambda eid: [])
    return controller, graph, runs


def test_thread_check_accepts_only_actual_exact_prediction_evidence(tmp_path, monkeypatch):
    import pandas as pd
    controller, graph, runs = thread_check_fixture(tmp_path, monkeypatch)
    assert controller.verify_thread_equivalence('CONTROL', 5)['identical_predictions']
    path = runs / 'CONTROL/predictions_CAL_s0.parquet'
    frame = pd.read_parquet(path)
    frame['pred_kms'] += .000001
    frame.to_parquet(path)
    graph['CONTROL']['runs'][0]['outputs'][path.name] = c.file_hash(path)
    with pytest.raises(RuntimeError, match='predictions differ'):
        controller.verify_thread_equivalence('CONTROL', 5)


def test_thread_check_rejects_input_source_mismatch(tmp_path, monkeypatch):
    controller, graph, runs = thread_check_fixture(tmp_path, monkeypatch)
    path = runs / 'CONTROL/receipt.json'
    record = c.read_json(path)
    record['source_access']['input']['sha256'] = 'changed'
    c.write_json(path, record)
    graph['CONTROL']['runs'][0]['outputs'][path.name] = c.file_hash(path)
    with pytest.raises(RuntimeError, match='source_access'):
        controller.verify_thread_equivalence('CONTROL', 5)


def test_thread_check_rejects_duplicate_experiment_starts(tmp_path, monkeypatch):
    from expos import core
    controller, graph, runs = thread_check_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(core, '_lines', lambda: [
        {'event': 'start', 'id': 'BASE'}, {'event': 'start', 'id': 'CONTROL'},
        {'event': 'start', 'id': 'CONTROL'}])
    with pytest.raises(RuntimeError, match='lineage ambiguous'):
        controller.verify_thread_equivalence('CONTROL', 5)
