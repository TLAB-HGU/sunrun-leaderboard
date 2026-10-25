"""Export existing protected roll3 evidence; never scores official predictions."""
import hashlib
import json
from pathlib import Path
import sys

import gate

PACKAGE = Path(__file__).resolve().parent
LB = Path('/home/t-lab01/sunrun/leaderboard')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def verify_registration(m, entries, hashes):
    """Bind assigned experiment ID to the registered immutable launch intent."""
    records = [e for e in entries if e.get('event')=='register' and e.get('id')==m['experiment_id']]
    gate.require(len(records)==1,'exactly one registration required')
    code = records[0]['code']
    gate.require(isinstance(code,dict) and code,'registered code hashes required')
    paths = {}
    for name,sha in code.items():
        path = Path(name)
        if not path.is_absolute(): path = LB/path
        gate.require(file_hash(path)==sha,'registered file changed: '+str(path))
        paths[path.resolve()] = sha
    intents = [p for p in paths if p.name=='intent.json']
    gate.require(len(intents)==1,'registered intent.json required')
    gate.require(gate.read(intents[0])=={k:v for k,v in m.items() if k!='experiment_id'},'registered intent differs from manifest')
    gate.require(any(p.name=='inputs.json' for p in paths),'registered inputs required')
    gate.require(gate.digest(hashes)==m['code_sha256'],'current runner code differs from manifest')
    for name,sha in hashes.items():
        gate.require(paths.get((PACKAGE/name).resolve())==sha,'canonical runner source not registered: '+name)
    return records[0]


def paired_rows(candidate,parent,seeds,folds):
    """Use actual preofficial paired observations; no synthetic fold weights."""
    import numpy as np
    keys = ['fold','origin','h']
    frames = []
    for frame in (candidate,parent):
        gate.require(not frame.duplicated(keys).any(),'duplicate preofficial keys')
        frames.append(frame.sort_values(keys).reset_index(drop=True))
    c,p = frames
    gate.require(c[keys].equals(p[keys]),'paired preofficial keys differ')
    gate.require(c['obs'].equals(p['obs']) and c['y'].equals(p['y']),'paired observations/targets differ')
    gate.require(set(map(str,c['fold'].unique()))==set(folds),'preofficial fold identity differs')
    rows = []
    for fold in folds:
        mask = (c['fold'].astype(str)==fold) & c['obs']
        count = int(mask.sum())
        gate.require(count>0,'empty observed fold')
        for seed in seeds:
            ce = c.loc[mask,f's{seed}'].to_numpy()-c.loc[mask,'y'].to_numpy()
            pe = p.loc[mask,f's{seed}'].to_numpy()-p.loc[mask,'y'].to_numpy()
            gate.require(np.isfinite(ce).all() and np.isfinite(pe).all(),'nonfinite paired errors')
            rows.append(dict(fold=fold,seed=seed,n_observed=count,
                             parent_sse=float(np.square(pe).sum()),candidate_sse=float(np.square(ce).sum())))
    return rows


def export_selection(m,parent,out,factorial=None):
    gate.preflight(m,parent,factorial)
    # Import only the preofficial evaluator. No lbeval/official snapshot call.
    sys.path.insert(0,str(LB))
    from expos import core,roll3
    import runner
    from build_inputs import evaluator_hash
    gate.require(not core.check_protected(),'protected evaluator files changed')
    gate.require(evaluator_hash()==m['contract']['evaluation_sha256'],'evaluator identity differs from frozen manifest')
    entries = gate.graph_entries()
    hashes = runner.code_hashes()
    verify_registration(m,entries,hashes)
    verify_registration(parent,entries,hashes)
    gate.require(not any(e.get('id')==m['experiment_id'] and e.get('event') in ('official','lbeval') for e in entries),'candidate already officially exposed')
    events = [e for e in entries if e.get('event')=='roll3' and e.get('id')==m['experiment_id'] and e.get('control')==parent['experiment_id'] and e.get('baseline') is False]
    gate.require(bool(events),'no matching protected paired roll3 record')
    event = events[-1]
    c,cs = roll3.load(m['experiment_id'])
    p,ps = roll3.load(parent['experiment_id'])
    gate.require(cs==ps==m['contract']['seeds'],'loaded seed set differs')
    # Detect changed prediction files by comparing observed sufficient metrics.
    for frame,seeds,key in ((c,cs,'candidate'),(p,ps,'control_summary')):
        actual = roll3.summarise(frame,seeds)
        recorded = event[key]
        gate.require(actual['rows']==recorded['rows'],'loaded rows differ from scored record')
        for seed in seeds:
            gate.require(abs(actual['mse_by_seed'][seed]-recorded['mse_by_seed'][str(seed)]) <= 1e-8*max(1,actual['mse_by_seed'][seed]),'loaded seed score differs from graph')
        gate.require(abs(actual['mse_seedmean_pred']-recorded['mse_seedmean_pred']) <= 1e-8*max(1,actual['mse_seedmean_pred']),'loaded pooled score differs from graph')
    ev = dict(schema_version=1,stage='preofficial',official_exposed=False,
              manifest_sha256=gate.digest(m),parent_sha256=gate.digest(parent),
              contract_sha256=gate.digest(m['contract']),source_sha256=gate.digest(event),
              rows=paired_rows(c,p,cs,m['contract']['folds']))
    receipt = {key:ev[key] for key in ('manifest_sha256','parent_sha256','contract_sha256','stage','official_exposed','source_sha256')}
    receipt.update(evaluator_sha256=m['contract']['evaluation_sha256'],rows_sha256=gate.digest(ev['rows']),
                   pooled_parent_f1=event['gate']['control_f1_seedmean'],pooled_candidate_f1=event['gate']['f1_seedmean'],
                   ci95_upper_relative_mse=event['gate']['ci95'][1],uncertainty_method='paired_temporal_block_bootstrap',
                   graph_event_sha256=gate.digest(event),graph_event_hash=event['hash'])
    out = Path(out)
    receipt_path = out.with_name(out.name+'.receipt.json')
    gate.require(not out.exists() and not receipt_path.exists(),'evidence destination already exists')
    gate.atomic(receipt_path,receipt)
    ev['evaluator_receipt'] = dict(path=str(receipt_path.resolve()),sha256=file_hash(receipt_path))
    try:
        gate.selection(m,parent,ev)
        gate.atomic(out,ev)
    except Exception:
        receipt_path.unlink(missing_ok=True)
        raise
    return ev


def main():
    parser = gate.JsonParser()
    for key in ('manifest','parent','out'): parser.add_argument('--'+key,required=True)
    parser.add_argument('--factorial')
    args = parser.parse_args()
    try:
        result = export_selection(gate.read(args.manifest),gate.read(args.parent),args.out,
                                  gate.read(args.factorial) if args.factorial else None)
        print(json.dumps(dict(ok=True,evidence_sha256=gate.digest(result),path=args.out)))
        return 0
    except Exception as error:
        print(json.dumps(dict(ok=False,error=str(error))))
        return 2


if __name__=='__main__': raise SystemExit(main())
