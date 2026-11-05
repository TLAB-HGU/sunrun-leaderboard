"""Validate migrated complete blocks without executing any training."""
import json
from pathlib import Path
import resume_run

HERE=Path(__file__).resolve().parent
receipt=json.loads((HERE/'local-migration.json').read_text())
results={}
for name in receipt['migrated']:
    out=HERE/'runs'/name
    launch=json.loads((out/'launch.json').read_text())
    data=launch['input_identity']
    record=resume_run.identity(out/'experiment.py',out/'runner_at_launch.py',data['path'],out,data['bytes'],data['mtime_ns'],launch['family'],name)
    record.update(threads=launch['threads'],smoke=False)
    if (out/'resume_identity.json').exists():
        assert json.loads((out/'resume_identity.json').read_text())==record
    module=resume_run.import_snapshot(out/'experiment.py')
    checks=[]
    for period in module.PERIODS:
        reports=[out/f'{period}_b{b}.json' for b in range(3)]
        if not any(p.exists() for p in reports):continue
        if launch['family']=='calib11':
            training,query=module.load(Path(data['path']),name,period)
            rounds=module.RECIPES[name][3][list(module.PERIODS).index(period)]
        for block,path in enumerate(reports):
            if not path.exists():continue
            try:
                if launch['family']=='calib11':
                    resume_run.validate_calib_block(module,record,out,period,block,rounds,launch['threads'],training,query)
                else:
                    resume_run.validate_top_block(module,record,out,period,block,launch['threads'])
                checks.append({'block':path.stem,'valid':True})
            except Exception as e:
                checks.append({'block':path.stem,'valid':False,'reason':str(e)})
    results[name]=checks
    print(name,json.dumps(checks),flush=True)
(HERE/'local-migration-validation.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(v['valid'] for checks in results.values() for v in checks)
