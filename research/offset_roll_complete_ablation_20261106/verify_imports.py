"""Audit original imported training metadata before admitting any new work."""
import json
from prepare_campaign import ROOT, REMOTE, remote, RPY


def main():
    allocation=json.loads((ROOT/'allocation.json').read_text())
    global_assignment=json.loads((ROOT/'assignment.global.json').read_text())
    configs={r['experiment']:r['config'] for r in global_assignment['experiments']}
    code=f'''import pathlib,json,subprocess,sys,io,tarfile
root=pathlib.Path({REMOTE!r});configs={ {n:configs[n] for n in allocation['imports']}!r}
code={AUDIT!r}
r=subprocess.run([{RPY!r},'-c',code,str(root),json.dumps(configs)],capture_output=True)
sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr)
if r.returncode:raise RuntimeError('Original import metadata audit failed')
'''
    try:
        result=remote(code,timeout=120)
    except __import__('subprocess').CalledProcessError as error:
        raise RuntimeError(error.stderr.decode()) from None
    proof=json.loads(result.stdout)
    (ROOT/'import-audit.json').write_text(json.dumps(proof,indent=2)+'\n')
    for name,records in proof['runs'].items():
        path=ROOT/'runs'/name/'imported.json';p=json.loads(path.read_text());p['original_metadata_audit']=records;path.write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(validated=proof['validated'],runs=len(proof['runs']))))


AUDIT=r'''
import sys,json,pathlib,numpy,pandas,xgboost,core,subprocess
root=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(root));configs=json.loads(sys.argv[2]);proof=dict(validated=True,runs={})
for name,expected in configs.items():
 imported=json.loads((root/'runs'/name/'imported.json').read_text());src=pathlib.Path(imported['imported_from']);periods={}
 for period in ('F1','F2','F3'):
  m=json.loads((src/(period+'_metadata.json')).read_text());c=m['config'];norm=core.normalize_config(expected)
  for k in core.TRAINING_KEYS:
   actual=c.get(k,core.DEFAULTS[k]);wanted=norm[k]
   if k=='exclude_features':actual,wanted=set(actual),set(wanted)
   if actual!=wanted:raise ValueError(name+' training mismatch '+k)
  for k,v in [('rounds',900),('tail_mode','step'),('offset_estimator','mean'),('offset_condition','none')]:
   if c.get(k,v)!=v:raise ValueError(name+' incompatible '+k)
  if m.get('smoke') or m['training_cutoff']!=core.baseline.CUTOFFS[period] or m['maturity_hours']!=144:raise ValueError(name+' protocol mismatch')
  if m['features']!=core.feature_names(norm):raise ValueError(name+' feature mismatch')
  for key,actual in [('numpy',numpy.__version__),('pandas',pandas.__version__),('xgboost',xgboost.__version__)]:
   if m['versions'][key]!=actual:raise ValueError(name+' version mismatch '+key)
  ident=m['identity']['data'];p=pathlib.Path(ident['path'])
  if p.stat().st_size!=ident['bytes'] or p.stat().st_mtime_ns!=ident['mtime_ns']:raise ValueError(name+' source input identity changed')
  if subprocess.run(['cmp','-s',str(p),str(root/'input/df_ffill_trainy.parquet')]).returncode:raise ValueError(name+' original training input differs')
  periods[period]=dict(original_metadata=m,input_binary_equal=True)
 proof['runs'][name]=periods
(root/'import-audit.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
'''
# The imported core must be available before this audit interpreter imports it.
AUDIT=AUDIT.replace('import sys,json,pathlib,numpy,pandas,xgboost,core,subprocess','import sys,json,pathlib,numpy,pandas,xgboost,subprocess\nsys.path.insert(0,sys.argv[1])\nimport core')

if __name__=='__main__':main()
