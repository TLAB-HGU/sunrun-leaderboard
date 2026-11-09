"""Prepare an isolated two-host campaign, preserving validated historical runs."""
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(ROOT.parent / 'rerun_df_ffill_20261103'))
from golab2_transport import remote as legacy_remote
import base64

def remote(code, *, input=None, timeout=120):
    if input is None:
        return legacy_remote(code, timeout=timeout)
    envelope=json.dumps(dict(code=code, data=base64.b64encode(input).decode())).encode()
    loader="import sys,json,base64,io;p=json.load(sys.stdin);sys.stdin=io.TextIOWrapper(io.BytesIO(base64.b64decode(p['data'])));exec(p['code'])"
    return subprocess.run(["ssh","-o","BatchMode=yes","golab2", "wsl -d Ubuntu -- python3 -c " + '"' + loader.replace('"', '\"') + '"'],input=envelope,capture_output=True,timeout=timeout,check=True)
REMOTE = '/home/yehwan/sunrun-offset-roll-complete-ablation-20261106'
RPY = '/home/yehwan/sunrun-golab2-20261104-17/.venv/bin/python'
DATA = REPO / 'research/rerun_df_ffill_20261103/input/df_ffill_trainy.parquet'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    if (ROOT / 'prepared.json').exists():
        raise RuntimeError('Already prepared; refuse overwrite')
    previous = ROOT.parent / 'calib_offset_roll_ablation_20261104'
    assignment = json.loads((previous / 'assignment.full-superseded.json').read_text())
    for seed in (42, 43, 44):
        assignment['experiments'].append(dict(experiment=f'spectral_hessian_only_s{seed}', variant='spectral_hessian_only', category='mechanism', stage='ablation', seed=seed, family='pooling', threads=4, config=dict(seed=seed, spectral_hessian_only=True)))
    imports = {}
    for seed in (42, 43, 44):
        imports[f'baseline_s{seed}'] = ['sunrun-offset-roll-ablation2-20261105', f'baseline_s{seed}', False]
        imports[f'dropgroup_strip_s{seed}'] = ['sunrun-offset-roll-ablation2-20261105', f'f1_drop_strip_s{seed}', False]
        imports[f'block12_s{seed}'] = ['sunrun-offset-roll-ablation2-20261105', f'm1_block12_s{seed}', False]
        imports[f'loss_t4_sp0_offset_s{seed}'] = ['sunrun-offset-roll-ablation5-20261105', f's0_spectral_off_s{seed}', True]
    imports['loss_t1_sp1_offset_s42'] = ['sunrun-calib-offset-roll-ablation-20261104', 'loss_t1_sp1_offset_s42', False]
    recipes = {r['experiment']: r for r in assignment['experiments']}
    # Put replay dependencies on the same host. Imported sources exist on both hosts.
    loads = {'local': 0, 'golab2': 0}
    owners = {}
    for recipe in assignment['experiments']:
        name = recipe['experiment']
        if name in imports:
            owners[name] = 'imported'
            continue
        dep = recipe['config'].get('reuse_run')
        if dep and owners.get(dep) in loads:
            host = owners[dep]
        else:
            host = min(loads, key=loads.get)
        owners[name] = host
        if not dep:
            loads[host] += 1
        recipe['execution_host'] = host
    assignment.update(campaign=ROOT.name, host='local', python=str(REPO/'.venv/bin/python'), data=str(DATA), observations=str(ROOT/'competition-observations.parquet'), max_jobs=5, threads=4, max_cpu_threads_total=20, reserve_memory_gib=6, estimated_job_memory_gib=5, baseline_reference_predictions=str(ROOT/'reference/predictions.parquet'), run_analysis=False, host_reproduction_gate='host-reproduction.json')
    dump(ROOT/'assignment.global.json', assignment)
    dump(ROOT/'allocation.json', dict(owners=owners, imports=imports, new_training_by_host=loads))
    dump(ROOT/'provenance.json', dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(), observation_revision=assignment['observation_revision'], policy='one fixed global inventory; no score-driven expansion', baseline='original A; no B1', hosts=['local','golab2']))
    shutil.copy2(previous/'competition.py', ROOT/'competition.py')
    sources = ['core.py','baseline.py','experiment.py','competition.py','ablation_scheduler.py','analyze_results.py','provenance.json','smoke_equivalence.py','host_gate.py']
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
        for name in sources:
            tar.add(ROOT/name,arcname=name)
        for name in ('experiment.py','baseline.py'):
            tar.add(previous/name,arcname='reference_source/'+name)
    setup = f'''import pathlib,tarfile,io,sys,json,shutil,os
root=pathlib.Path({REMOTE!r});root.mkdir(exist_ok=True)
if (root/'assignment.json').exists():raise RuntimeError('Remote campaign already prepared')
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:t.extractall(root,filter='data')
(root/'input').mkdir(exist_ok=True);(root/'reference').mkdir(exist_ok=True)
shutil.copy2('/home/yehwan/sunrun-offset-roll-ablation2-20261105/input/df_ffill_trainy.parquet',root/'input/df_ffill_trainy.parquet')
shutil.copy2('/home/yehwan/sunrun-offset-roll-ablation2-20261105/competition-observations.parquet',root/'competition-observations.parquet')
imports={imports!r};recipes={ {n: recipes[n] for n in imports}!r}
for name,(campaign,source,unblend) in imports.items():
 src=pathlib.Path('/home/yehwan')/campaign/'runs'/source;dst=root/'runs'/name
 state=json.loads((src.parent.parent/'scheduler-state.json').read_text())
 if state[source]['state']!='trained_validated' or (src/'train.exit').read_text().strip()!='0':raise RuntimeError('Unvalidated import '+source)
 dst.mkdir(parents=True,exist_ok=True)
 for p in src.iterdir():
  if p.is_file() and p.suffix in ('.json','.npz','.parquet','.py','.exit','.log'):shutil.copy2(p,dst/p.name)
 for period in ('F1','F2','F3'):
  (dst/(period+'_models')).mkdir(exist_ok=True)
  for p in (src/(period+'_models')).glob('*'):
   if p.is_file():shutil.copy2(p,dst/(period+'_models')/p.name)
  if name.startswith('baseline_'):shutil.copytree(src/(period+'_models/full'),dst/(period+'_models/full'),dirs_exist_ok=True)
  if unblend:shutil.copy2(src/(period+'_unblended_predictions.npz'),dst/(period+'_predictions.npz'))
  meta=json.loads((dst/(period+'_metadata.json')).read_text());meta['config']=recipes[name]['config'];meta['experiment']=name
  meta['import_source_metadata']=str(src/(period+'_metadata.json'))
  (dst/(period+'_metadata.json')).write_text(json.dumps(meta,indent=2)+'\\n')
 if unblend:
  shutil.copy2(src/'unblended_predictions.parquet',dst/'predictions.parquet');shutil.copy2(src/'paired_unblended_score.json',dst/'local_score.json')
 (dst/'imported.json').write_text(json.dumps(dict(imported_from=str(src),source_validated=True,unblended=unblend,original_config=json.loads((src/'F1_metadata.json').read_text())['config']),indent=2)+'\\n')
shutil.copy2(root/'runs/baseline_s42/predictions.parquet',root/'reference/predictions.parquet')
print(json.dumps(dict(imported=len(imports),root=str(root))))
'''
    print(remote(setup,input=buffer.getvalue(),timeout=180).stdout.decode())
    # Transfer only immutable imports and data, not live output.
    export=f'''import pathlib,tarfile,sys
root=pathlib.Path({REMOTE!r})
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for n in ['runs','reference','competition-observations.parquet','input/df_ffill_trainy.parquet']:t.add(root/n,arcname=n)
'''
    blob=remote(export,timeout=600).stdout
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as tar:tar.extractall(ROOT,filter='data')
    # Full binary equality establishes relocation; no content hashes required.
    if subprocess.run(['cmp','-s',str(DATA),str(ROOT/'input/df_ffill_trainy.parquet')]).returncode:
        raise RuntimeError('Local/remote training inputs differ')
    from core import file_identity
    for name in imports:
        out=ROOT/'runs'/name
        for period in ('F1','F2','F3'):
            path=out/f'{period}_metadata.json';meta=json.loads(path.read_text());meta['identity']['data']=file_identity(DATA);dump(path,meta)
        proof=json.loads((out/'imported.json').read_text());proof['data_relocation']=dict(binary_equal=True,destination=file_identity(DATA));dump(out/'imported.json',proof)
    for host in ('local','golab2'):
        subset=[r for r in assignment['experiments'] if owners[r['experiment']] in ('imported',host)]
        host_assignment=dict(assignment,experiments=subset,host=host,smoke_proof='smoke-local.json' if host=='local' else 'smoke-remote.json')
        if host=='local':
            dump(ROOT/'assignment.json',host_assignment)
        else:
            host_assignment.update(python=RPY,data=REMOTE+'/input/df_ffill_trainy.parquet',observations=REMOTE+'/competition-observations.parquet',baseline_reference_predictions=REMOTE+'/reference/predictions.parquet')
            code=f'''import pathlib,json
root=pathlib.Path({REMOTE!r});assignment={host_assignment!r}
(root/'assignment.json').write_text(json.dumps(assignment,indent=2)+'\\n')
data=root/'input/df_ffill_trainy.parquet';identity=dict(path=str(data),bytes=data.stat().st_size,mtime_ns=data.stat().st_mtime_ns)
for p in (root/'runs').glob('*/F*_metadata.json'):
 m=json.loads(p.read_text());m['identity']['data']=identity;p.write_text(json.dumps(m,indent=2)+'\\n')
state={{n:dict(state='trained_validated',imported_from=json.loads((root/'runs'/n/'imported.json').read_text())['imported_from']) for n in {list(imports)!r}}}
(root/'scheduler-state.json').write_text(json.dumps(state,indent=2)+'\\n')
'''
            remote(code)
    state={n:dict(state='trained_validated',imported_from=json.loads((ROOT/'runs'/n/'imported.json').read_text())['imported_from']) for n in imports}
    dump(ROOT/'scheduler-state.json',state)
    dump(ROOT/'registry.json',{r['experiment']:dict(r,family='pooling') for r in assignment['experiments'] if owners[r['experiment']] in ('local','imported')})
    dump(ROOT/'manifest.json',dict(campaign=ROOT.name,hosts=['local','golab2'],global_inventory='assignment.global.json',no_automatic_expansion=True))
    dump(ROOT/'runtime.json',state)
    dump(ROOT/'prepared.json',dict(imported=len(imports),binary_equal=True,registered=len(recipes),new_training_by_host=loads))
    print((ROOT/'prepared.json').read_text())


if __name__=='__main__':main()
