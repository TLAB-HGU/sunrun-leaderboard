"""Frozen golab2 staging, foreground scheduling, and verified artifact collection."""
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

from scheduler_support import atomic, identity, now

REMOTE = '/home/yehwan/sunrun-followup-feature-axes4-20261109'
ORIGINAL = '/home/yehwan/sunrun-offset-roll-complete-ablation-20261106/input/df_ffill_trainy.parquet'
RPY = '/home/yehwan/sunrun-golab2-20261104-17/.venv/bin/python'
FILES = ('core.py', 'baseline.py', 'competition.py', 'replay.py', 'scheduler_support.py',
         'host_scheduler.py', 'minute_features.py', 'temporal_features.py',
         'build_data.py', 'inventory.json', 'provenance.json', 'strategy.md', 'parent-lock.json',
         'ace-source-manifest.json', 'source-smoke.json')
PREFLIGHT_FILES = ('core.py', 'baseline.py', 'competition.py', 'inventory.json', 'replay.py',
                   'scheduler_support.py', 'host_scheduler.py', 'assignment.json')


def remote(code, **kwargs):
    from transport_support import remote as transport_remote
    return transport_remote(code, **kwargs)


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('Unsafe experiment name')
    return value


def payload(root, assignment, ready_axes):
    """Select complete three-seed ready axes and only explicitly owned inputs."""
    root = Path(root).resolve()
    axes = sorted(set(ready_axes))
    if not axes or any(type(axis) is not int or axis not in range(1, 5) for axis in axes):
        raise ValueError('Ready axes must be distinct integers 1..4')
    selected = []
    for axis in axes:
        recipes = [r for r in assignment['experiments'] if int(r['axis']) == axis]
        if sorted(r['config']['seed'] for r in recipes) != [42, 43, 44]:
            raise ValueError('Each ready axis requires exactly three locked seeds')
        for recipe in recipes:
            row = dict(recipe, data=f'{REMOTE}/input/axis{axis}.parquet', threads=4)
            _name(row['experiment'])
            selected.append(row)
    if len({r['experiment'] for r in selected}) != len(selected):
        raise ValueError('Duplicate experiment names')
    remote_assignment = dict(assignment, experiments=selected, host='golab2', python=RPY,
                             threads=4, max_jobs=6, data=selected[0]['data'])
    remote_assignment['observations'] = assignment['remote_observations']
    files = [*FILES, *(f'input/axis{axis}.features.parquet' for axis in axes),
             *(f'input/axis{axis}.features.json' for axis in axes)]
    for name in files:
        if not (root / name).is_file() or (root / name).is_symlink():
            raise ValueError('Missing or nonregular frozen stage input: ' + name)
    receipt = {'ready_axes': axes, 'assignment': remote_assignment,
               'sources': {name: identity(root / name) for name in files},
               'original_bytes': assignment.get('original_data_identity', {}).get('bytes', 225755615)}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name in files:
            archive.add(root / name, arcname=name, recursive=False)
        for name, value in [('assignment.json', remote_assignment), ('stage-request.json', receipt)]:
            content = (json.dumps(value, indent=2) + '\n').encode()
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    return stream.getvalue(), receipt


# Runs on the remote host with payload on stdin; all mutations stay in REMOTE.
_STAGE = r'''
import io,json,sys,tarfile,tempfile,shutil,subprocess,os,time,fcntl
from pathlib import Path
root=Path(REMOTE)
blob=sys.stdin.buffer.read()
with tempfile.TemporaryDirectory(prefix='sunrun-axis4-stage-') as temporary:
 staging=Path(temporary)
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
  for member in archive.getmembers():
   path=Path(member.name)
   if path.is_absolute() or '..' in path.parts or not member.isfile():
    raise ValueError('Unsafe staging payload')
  archive.extractall(staging,filter='data')
 request=json.loads((staging/'stage-request.json').read_text())
 if root.exists():
  receipt=root/'prepared-remote.json'
  if not receipt.is_file() or json.loads(receipt.read_text())['request']!=request:
   raise ValueError('Existing remote destination has no matching complete receipt; preserve for diagnosis')
  saved=json.loads(receipt.read_text())
  for name,stat in saved['frozen_files'].items():
   p=root/name;s=p.stat()
   if {'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}!=stat:
    raise ValueError('Frozen remote file changed: '+name)
  print(json.dumps({'root':str(root),'resumed':True}));sys.exit(0)
 original=Path(ORIGINAL)
 if original.stat().st_size!=request['original_bytes']:
  raise ValueError('Existing cross-host original input byte identity differs')
 assignment=request['assignment']
 if not Path(assignment['observations']).is_file():raise ValueError('Missing official observations')
 root.mkdir()
 for source in staging.rglob('*'):
  if source.is_file():
   target=root/source.relative_to(staging);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
 prepare="import sys,pyarrow as pa;from pathlib import Path;pa.set_cpu_count(1);pa.set_io_thread_count(1);sys.path.insert(0,sys.argv[1]);from build_data import augment;augment(Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]))"
 for axis in request['ready_axes']:
  # Keep admission locked through this bounded one-thread preparation job.
  # Existing work continues; no second scheduler can admit over this reservation.
  with Path('/tmp/sunrun-cpu-admission.lock').open('a') as admission:
   deadline=time.monotonic()+1200
   while True:
    fcntl.flock(admission,fcntl.LOCK_EX)
    sys.path.insert(0,str(root))
    from scheduler_support import memory_gib,busy_cpu_cores
    from host_scheduler import workloads,capacity
    allowed=capacity(workloads(),memory_gib(),busy_cpu_cores(),threads=1)
    if allowed and shutil.disk_usage(root).free/1024**3>=15:break
    fcntl.flock(admission,fcntl.LOCK_UN)
    if time.monotonic()>deadline:raise TimeoutError('No resource capacity for feature augmentation')
    time.sleep(10)
   subprocess.run([RPY,'-c',prepare,str(root),str(original),str(root/f'input/axis{axis}.features.parquet'),str(root/f'input/axis{axis}.parquet')],env=env,check=True,stdout=sys.stderr,timeout=600)
 def identity(p):
  s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
 def write(name,value):
  p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
 inputs={r['data']:identity(Path(r['data'])) for r in assignment['experiments']}
 sources={name:identity(root/name) for name in PREFLIGHT_FILES}
 write('preflight.json',{'passed':True,'golab2':{'input_identities':inputs,'source_identity':sources},'checks':'parent locked, ready feature cache, original-column equality during augmentation'})
 write('scheduler-state.json',{r['experiment']:{'state':'waiting'} for r in assignment['experiments']})
 frozen_names=[*request['sources'],'assignment.json','preflight.json',*[str(Path(p).relative_to(root)) for p in inputs]]
 frozen={name:{key:identity(root/name)[key] for key in ('bytes','mtime_ns')} for name in frozen_names}
 write('prepared-remote.json',{'request':request,'frozen_files':frozen,'original_input_identity':identity(original)})
 print(json.dumps({'root':str(root),'resumed':False,'experiments':len(assignment['experiments'])}))
'''


def stage(root, assignment, ready_axes):
    data, request = payload(root, assignment, ready_axes)
    header = f'REMOTE={REMOTE!r}\nORIGINAL={ORIGINAL!r}\nRPY={RPY!r}\nPREFLIGHT_FILES={PREFLIGHT_FILES!r}\n'
    result = json.loads(remote(header + _STAGE, input=data, timeout=1800).stdout)
    if result['root'] != REMOTE:
        raise ValueError('Unexpected staging destination')
    atomic(Path(root) / 'remote-prepared.json', dict(result, request=request, prepared_utc=now()))
    return REMOTE


def remote_snapshot():
    code = f'''import json
from pathlib import Path
r=Path({REMOTE!r});state=json.loads((r/'scheduler-state.json').read_text());pid=(r/'scheduler.pid').read_text().strip() if (r/'scheduler.pid').exists() else ''
alive=False
if pid.isdigit():
 p=Path('/proc')/pid/'cmdline'
 alive=p.exists() and str(r/'host_scheduler.py').encode() in p.read_bytes().split(bytes([0]))
print(json.dumps(dict(state=state,scheduler_alive=alive)))
'''
    return json.loads(remote(code, timeout=35).stdout)


def _extract(archive, staging, name):
    for member in archive.getmembers():
        path = Path(member.name)
        if (path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != name
                or not (member.isfile() or member.isdir())):
            raise ValueError('Unsafe remote artifact path')
    archive.extractall(staging, filter='data')


def collect_run(root, name):
    name = _name(name)
    root = Path(root)
    out = root / 'runs' / name
    if out.exists():
        if (out / 'collected.json').is_file():
            return
        raise ValueError('Incomplete local destination; preserve and diagnose: ' + name)
    code = f'''import sys,tarfile,json
from pathlib import Path
out=Path({REMOTE!r})/'runs'/{name!r}
if (out/'train.exit').read_text().strip()!='0':raise ValueError('Remote run did not exit0')
if not (out/'training-validation.json').exists():raise ValueError('Missing remote replay validation')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:archive.add(out,arcname={name!r})
'''
    data = remote(code, timeout=600).stdout
    (root / 'runs').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.collect-', dir=root) as directory:
        staging = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
            _extract(archive, staging, name)
        source = staging / name
        proof = json.loads((source / 'training-validation.json').read_text())
        if (proof['rows'] != 454680 or set(proof['saved_model_replay']) != {'F1', 'F2', 'F3'}
                or not proof.get('official_score_recomputed')):
            raise ValueError('Incomplete remote scoring/model replay receipt')
        for report in proof['saved_model_replay'].values():
            if report['max_abs_error'] > 1e-6 or report['boosters'] != 3 or report['rounds'] != 900:
                raise ValueError('Remote replay validation failed')
        for filename in ('core.py', 'baseline.py', 'competition.py', 'inventory.json'):
            if (source / filename).read_bytes() != (root / filename).read_bytes():
                raise ValueError('Remote scientific source differs: ' + filename)
        if json.loads((source / 'complete.json').read_text()).get('publish') != 'not_required':
            raise ValueError('Unexpected publication state')
        atomic(source / 'collected.json', dict(host='golab2', utc=now(), includes_saved_models=True))
        source.rename(out)


def foreground_scheduler(root, python=RPY):
    """Actual foreground SSH scheduler, called from a root-owned tmux wrapper."""
    root = Path(root)
    code = ('import os\n' + f'os.chdir({REMOTE!r})\n'
            + 'env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",OMP_NUM_THREADS="1")\n'
            + f'os.execve({python!r},[{python!r},"-u",{REMOTE + "/host_scheduler.py"!r},"--root",{REMOTE!r}],env)\n')
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3',
                             'golab2', 'wsl -d Ubuntu -- python3 -'], input=code.encode(), check=False)
    temporary = root / 'remote_scheduler_controller.exit.tmp'
    temporary.write_text(str(result.returncode) + '\n')
    temporary.replace(root / 'remote_scheduler_controller.exit')
    return result.returncode
