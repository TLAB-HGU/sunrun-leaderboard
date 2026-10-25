"""Fail-closed cumulative research gate. Receipts are integrity records, not authority.

Canonical hashes use UTF-8 sorted compact JSON. Selection receipts must be issued
by the trusted evaluator integration; this module cannot authenticate their author.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


TRUSTED_GRAPH = Path('/home/t-lab01/sunrun/research/expos/graph.jsonl')


class GateError(ValueError):
    pass


class JsonParser(argparse.ArgumentParser):
    def error(self, message):
        print(json.dumps(dict(ok=False,error=message)))
        raise SystemExit(2)


def require(condition, message):
    if not condition:
        raise GateError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(GateError('nonfinite JSON')))


def keys(obj, expected):
    require(isinstance(obj, dict) and set(obj) == set(expected.split()), 'unexpected or missing fields: '+expected)


def sha(value):
    require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'invalid sha256')


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), 'invalid finite number')
    return value


def evaluation_identity(contract):
    """Comparison identity excludes training masks and model input sources."""
    return digest({key:contract[key] for key in
                   ('seeds','grid_sha256','evaluation_sha256','aggregation')})


def manifest(m):
    keys(m, 'schema_version experiment_id campaign_id method_class parent_sha256 declared_changes change_kind rationale recipe contract code_sha256 champion')
    require(m['schema_version'] == 1 and m['method_class'] == 'direct', 'only schema 1 direct models accepted')
    for name in ('experiment_id', 'campaign_id', 'rationale'):
        require(isinstance(m[name], str) and bool(m[name].strip()), 'missing '+name)
    sha(m['code_sha256'])
    if m['parent_sha256'] is not None: sha(m['parent_sha256'])
    require(m['change_kind'] in ('baseline', 'ablation', 'rollback', 'factorial'), 'invalid change_kind')
    recipe = m['recipe']
    require(isinstance(recipe, dict) and bool(recipe), 'empty recipe')
    for k, v in recipe.items():
        require(isinstance(k, str) and k and type(v) in (str, int, float, bool), 'recipe axes must be scalar leaves')
        if type(v) in (int, float): number(v)
    changes = m['declared_changes']
    require(isinstance(changes, list) and all(isinstance(k, str) for k in changes) and len(changes) == len(set(changes)), 'invalid declared changes')
    c = m['contract']
    keys(c, 'seeds folds cutoff grid_sha256 mask_sha256 sources evaluation_sha256 aggregation selection_aggregation requires_arrival inputs_sha256')
    require(isinstance(c['seeds'], list) and len(c['seeds']) >= 3 and all(type(s) is int for s in c['seeds']) and len(set(c['seeds'])) == len(c['seeds']), 'at least three unique seeds required')
    require(isinstance(c['folds'], list) and len(c['folds']) >= 3 and all(isinstance(f, str) and f for f in c['folds']) and len(set(c['folds'])) == len(c['folds']), 'at least three unique folds required')
    for key in ('grid_sha256', 'mask_sha256', 'evaluation_sha256', 'inputs_sha256'): sha(c[key])
    require(type(c['requires_arrival']) is bool, 'requires_arrival must be boolean')
    require(c['selection_aggregation'] == 'seed_mean_prediction_mse', 'selection aggregation must match roll3')
    for key in ('cutoff', 'aggregation'): require(isinstance(c[key], str) and c[key], 'missing '+key)
    require(isinstance(c['sources'], list) and c['sources'], 'sources required')
    ids = []
    for source in c['sources']:
        keys(source, 'id sha256 availability_sha256')
        require(isinstance(source['id'], str) and source['id'], 'source id required')
        ids.append(source['id'])
        sha(source['sha256']); sha(source['availability_sha256'])
    require(len(ids) == len(set(ids)), 'duplicate sources')
    ch = m['champion']
    keys(ch, 'experiment_id mse contract_sha256 aggregation source_sha256')
    require(number(ch['mse']) > 0, 'invalid champion mse')
    for key in ('contract_sha256', 'source_sha256'): sha(ch[key])
    require(all(isinstance(ch[k], str) and ch[k] for k in ('experiment_id', 'aggregation')), 'invalid champion identity')
    return digest(m)


def difference(a, b):
    return sorted(k for k in set(a) | set(b) if k not in a or k not in b or type(a[k]) != type(b[k]) or a[k] != b[k])


def preflight(m, parent=None, factorial=None):
    identity = manifest(m)
    if parent is None:
        require(m['change_kind'] == 'baseline' and m['parent_sha256'] is None and not m['declared_changes'], 'baseline only without parent')
        return {'manifest_sha256':identity, 'changes':[]}
    manifest(parent)
    require(m['parent_sha256'] == digest(parent), 'parent identity mismatch')
    require(m['experiment_id'] != parent['experiment_id'], 'experiment identity reused')
    for field in ('campaign_id', 'contract', 'champion', 'method_class'):
        require(m[field] == parent[field], 'frozen '+field+' changed')
    changed = difference(m['recipe'], parent['recipe'])
    require(changed and sorted(m['declared_changes']) == changed, 'no-op or undeclared recipe changes')
    if m['change_kind'] == 'factorial':
        require(len(changed) == 2 and factorial is not None, 'factorial requires two axes and A/B evidence')
        keys(factorial, 'base arm_a arm_b evidence_a evidence_b')
        require(digest(factorial['base']) == digest(parent), 'factorial base mismatch')
        a, b = factorial['arm_a'], factorial['arm_b']
        preflight(a, parent); preflight(b, parent)
        require(a['change_kind'] == b['change_kind'] == 'ablation', 'factorial arms must be ablations')
        axes = a['declared_changes'] + b['declared_changes']
        require(len(set(axes)) == 2 and sorted(axes) == changed, 'factorial axes mismatch')
        expected = dict(parent['recipe'])
        for arm in (a,b):
            axis = arm['declared_changes'][0]
            require(axis in arm['recipe'], 'factorial removal unsupported')
            expected[axis] = arm['recipe'][axis]
        require(expected == m['recipe'], 'factorial combined recipe mismatch')
        selection(a, parent, factorial['evidence_a']); selection(b, parent, factorial['evidence_b'])
    else:
        require(m['change_kind'] in ('ablation','rollback') and len(changed) == 1, 'exactly one axis required')
        require(m['change_kind'] == 'rollback' or set(parent['recipe']) <= set(m['recipe']), 'removal requires rollback')
    return {'manifest_sha256':identity, 'parent_sha256':digest(parent), 'changes':changed}


def evidence_record(m, parent_hash, ev, stage):
    keys(ev, 'schema_version stage official_exposed manifest_sha256 parent_sha256 contract_sha256 source_sha256 rows evaluator_receipt')
    require(ev['schema_version'] == 1 and ev['stage'] == stage, 'evidence stage mismatch')
    require(type(ev['official_exposed']) is bool, 'official_exposed must be boolean')
    require(ev['manifest_sha256'] == digest(m) and ev['parent_sha256'] == parent_hash and ev['contract_sha256'] == digest(m['contract']), 'evidence binding mismatch')
    sha(ev['source_sha256'])
    ref = ev['evaluator_receipt']
    keys(ref, 'path sha256'); sha(ref['sha256'])
    require(hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256'], 'receipt content changed')
    receipt = read(ref['path'])
    keys(receipt, 'manifest_sha256 parent_sha256 contract_sha256 stage official_exposed evaluator_sha256 rows_sha256 source_sha256 pooled_parent_f1 pooled_candidate_f1 ci95_upper_relative_mse uncertainty_method graph_event_sha256 graph_event_hash')
    for key in ('manifest_sha256','parent_sha256','contract_sha256','stage','official_exposed','source_sha256'):
        require(receipt[key] == ev[key], 'receipt binding mismatch: '+key)
    require(receipt['evaluator_sha256'] == m['contract']['evaluation_sha256'] and receipt['rows_sha256'] == digest(ev['rows']), 'evaluator/rows identity mismatch')
    require(isinstance(ev['rows'], list) and ev['rows'], 'paired rows required')
    seen = set()
    totals = {f:[0.,0.,0] for f in (['full'] if stage=='full_period' else m['contract']['folds'])}
    for row in ev['rows']:
        keys(row, 'fold seed n_observed parent_sse candidate_sse')
        pair = (row['fold'], row['seed'])
        require(type(row['seed']) is int and row['fold'] in totals and row['seed'] in m['contract']['seeds'] and pair not in seen, 'invalid/duplicate paired row')
        seen.add(pair)
        require(type(row['n_observed']) is int and row['n_observed'] > 0, 'positive observed count required')
        require(number(row['parent_sse']) > 0 and number(row['candidate_sse']) >= 0, 'invalid SSE')
        for i,key in enumerate(('parent_sse','candidate_sse','n_observed')): totals[row['fold']][i] += row[key]
    require(seen == {(f,s) for f in totals for s in m['contract']['seeds']}, 'missing fold/seed pairs')
    for fold in totals:
        require(len({r['n_observed'] for r in ev['rows'] if r['fold']==fold}) == 1, 'seed masks/counts differ')
    p,c,n = (sum(t[i] for t in totals.values()) for i in range(3))
    return receipt, {'relative_mse':c/p-1, 'parent_mse':p/n, 'candidate_mse':c/n,
                     'folds_improved':sum(t[1] < t[0] for t in totals.values()), 'fold_count':len(totals)}


def graph_entries():
    previous = ''
    entries = []
    with TRUSTED_GRAPH.open() as stream:
        for index,line in enumerate(stream):
            entry = json.loads(line)
            body = {key:value for key,value in entry.items() if key != 'hash'}
            expected = hashlib.sha256((previous + json.dumps(body,sort_keys=True)).encode()).hexdigest()
            require(entry.get('seq') == index and entry.get('prev') == previous and entry.get('hash') == expected, 'trusted graph chain broken')
            previous = expected
            entries.append(entry)
    return entries


def selection(m, parent, ev):
    require(ev.get('official_exposed') is False, 'official-exposed evidence cannot select')
    receipt, diagnostic = evidence_record(m, digest(parent), ev, 'preofficial')
    sha(receipt['graph_event_sha256'])
    require(receipt['graph_event_sha256'] == ev['source_sha256'], 'selection source must identify graph event')
    entries = graph_entries()
    require(not any(entry.get('id') == m['experiment_id'] and entry.get('event') in ('official','lbeval')
                    for entry in entries), 'candidate has official exposure in graph')
    event = next((entry for entry in entries if digest(entry) == receipt['graph_event_sha256']),None)
    require(event is not None, 'receipt not found in trusted graph')
    require(event.get('hash') == receipt['graph_event_hash'], 'graph hash mismatch')
    require(event.get('event') == 'roll3' and event.get('id') == m['experiment_id'] and
            event.get('control') == parent['experiment_id'] and event.get('baseline') is False,
            'graph experiment/control mismatch')
    candidate, control = event['candidate'], event['control_summary']
    for summary in (candidate,control):
        require(summary['seeds'] == m['contract']['seeds'], 'graph seed mismatch')
    require(candidate['rows'] == control['rows'] and candidate['rows'] > 0, 'graph paired rows mismatch')
    require(all(sum(r['n_observed'] for r in ev['rows'] if r['seed']==seed) == candidate['rows']
                for seed in m['contract']['seeds']), 'graph observed counts mismatch')
    g = event['gate']
    require(set(g['rel_by_fold']) == set(m['contract']['folds']), 'graph folds mismatch')
    require(g['n_folds_improved'] == sum(number(v)<0 for v in g['rel_by_fold'].values()), 'graph fold count mismatch')
    require(number(g['pooled_rel']) <= -.01 and g['n_folds_improved']*3 >= len(g['rel_by_fold'])*2,
            'insufficient pooled/fold improvement')
    require(number(control['mse_seedmean_pred']) > 0 and
            math.isclose(candidate['mse_seedmean_pred']/control['mse_seedmean_pred']-1,
                         g['pooled_rel'],rel_tol=1e-8,abs_tol=1e-10), 'graph pooled metric mismatch')
    require(receipt['uncertainty_method'] == 'paired_temporal_block_bootstrap' and
            receipt['ci95_upper_relative_mse'] == g['ci95'][1] and
            number(g['ci95'][1]) < 0 and g['n_blocks'] >= 2, 'missing/failed temporal uncertainty')
    require(receipt['pooled_parent_f1'] == g['control_f1_seedmean'] == control['f1_seedmean'] and
            receipt['pooled_candidate_f1'] == g['f1_seedmean'] == candidate['f1_seedmean'], 'graph F1 mismatch')
    for name in ('pooled_parent_f1','pooled_candidate_f1'):
        require(0 <= number(receipt[name]) <= 1, 'invalid event F1')
    require(receipt['pooled_candidate_f1'] >= receipt['pooled_parent_f1'] and g['pass'] is True, 'event F1/gate regression')
    return dict(relative_mse=g['pooled_rel'], folds_improved=g['n_folds_improved'],
                ci95=g['ci95'], aggregation=m['contract']['selection_aggregation'],
                diagnostic_mean_seed_mse=diagnostic, graph_event_sha256=digest(event))


@contextlib.contextmanager
def locked(path):
    path = Path(path)
    with Path(str(path)+'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield path


def atomic(path, value):
    fd, temp = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp,path)
        fd = os.open(path.parent,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def initialize(path, m):
    preflight(m)
    with locked(path) as path:
        require(not path.exists(), 'state already initialized')
        state = dict(schema_version=1,campaign_id=m['campaign_id'],accepted_sha256=digest(m),
                     accepted_manifest=m,history=[dict(action='initialize',manifest_sha256=digest(m))],
                     accepted_axes={},confirmation=None,parent_ids={digest(m):m['experiment_id']})
        atomic(path,state)
    return state


def promote(path, m, parent, ev, factorial=None):
    check = preflight(m,parent,factorial)
    result = selection(m,parent,ev)
    with locked(path) as path:
        state = read(path)
        require(state['confirmation'] is None, 'campaign frozen for official confirmation')
        require(state['accepted_sha256'] == digest(parent) and state['accepted_manifest'] == parent, 'stale parent')
        require(state['campaign_id'] == m['campaign_id'], 'campaign mismatch')
        for axis in check['changes']:
            require(axis not in state['accepted_axes'] or m['change_kind']=='rollback', 'accepted choice replacement requires documented rollback')
            state['accepted_axes'][axis] = m['recipe'].get(axis)
        state['parent_ids'][digest(m)] = m['experiment_id']
        state.update(accepted_sha256=digest(m),accepted_manifest=m)
        state['history'].append(dict(action='promote',manifest_sha256=digest(m),parent_sha256=digest(parent),evidence_sha256=digest(ev),result=result))
        atomic(path,state)
    return result


def reserve(path, m):
    manifest(m)
    with locked(path) as path:
        state = read(path)
        require(state['accepted_sha256'] == digest(m) and state['accepted_manifest'] == m, 'confirmation requires accepted parent')
        require(state['confirmation'] is None, 'confirmation already reserved')
        require(any(h['action']=='promote' for h in state['history']), 'preofficial promotion required before confirmation')
        if m['contract']['requires_arrival']:
            entries = graph_entries()
            parent_hash = m['parent_sha256']
            parents = [h for h in state['history'] if h.get('manifest_sha256')==parent_hash]
            require(bool(parents), 'arrival parent history missing')
            parent_id = state['parent_ids'].get(parent_hash)
            require(parent_id is not None, 'arrival parent identity missing')
            arrivals = [e for e in entries if e.get('event')=='arrival' and e.get('id')==m['experiment_id'] and e.get('control')==parent_id]
            require(bool(arrivals) and arrivals[-1].get('baseline') is False and arrivals[-1]['gate']['pass'] is True,
                    'required paired arrival gate has not passed')
        state['confirmation'] = dict(manifest_sha256=digest(m),contract_sha256=digest(m['contract']))
        state['history'].append(dict(action='reserve-confirmation',**state['confirmation']))
        atomic(path,state)
    return state['confirmation']


def compare(path, m, ev):
    manifest(m)
    state = read(path)
    require(state['confirmation'] == dict(manifest_sha256=digest(m),contract_sha256=digest(m['contract'])), 'matching confirmation reservation required')
    champion = m['champion']
    require(champion['contract_sha256'] == evaluation_identity(m['contract']) and champion['aggregation'] == m['contract']['aggregation'], 'champion evaluation/aggregation mismatch')
    require(m['contract']['aggregation'] == 'mean_seed_mse', 'full-period SSE evidence only supports mean_seed_mse')
    receipt, result = evidence_record(m,digest(champion),ev,'full_period')
    require(ev['official_exposed'] is True, 'full-period evidence must be marked exposed')
    require(math.isclose(result['parent_mse'],champion['mse'],rel_tol=1e-8), 'champion score/source mismatch')
    entries = graph_entries()
    official_gate = None
    for identity,source,expected in ((m['experiment_id'],ev['source_sha256'],result['candidate_mse']),
                                     (champion['experiment_id'],champion['source_sha256'],champion['mse'])):
        event = next((entry for entry in entries if digest(entry) == source),None)
        require(event is not None and event.get('event') == 'lbeval' and event.get('id') == identity,
                'full-period source must identify trusted lbeval event')
        require(set(event['per_seed']) == set(map(str,m['contract']['seeds'])), 'full-period seeds mismatch')
        require(math.isclose(event['summary']['mse_seedmean'],expected,rel_tol=1e-8), 'full-period graph metric mismatch')
        if identity == m['experiment_id']:
            official_gate = event['gate']
            require(type(official_gate.get('pass')) is bool and isinstance(official_gate.get('reasons'),list), 'invalid native official gate')
            require(receipt['graph_event_sha256'] == source and receipt['graph_event_hash'] == event['hash'],
                    'full-period receipt graph mismatch')
    mse_target_met = result['relative_mse'] <= -.05
    eligible = mse_target_met and official_gate['pass']
    return dict(**result, mse_target_met=mse_target_met,
                official_gate_pass=official_gate['pass'], official_gate_reasons=official_gate['reasons'],
                eligible_breakthrough=eligible, breakthrough=eligible,
                selection_eligible=False, champion_sha256=digest(champion))


def main():
    parser = JsonParser()
    parser.add_argument('command', choices=['init','preflight','promote','reserve-confirmation','compare'])
    for option in ('manifest','parent','factorial','evidence','state'): parser.add_argument('--'+option)
    args = parser.parse_args()
    try:
        require(args.manifest is not None, '--manifest required')
        m = read(args.manifest)
        parent = read(args.parent) if args.parent else None
        factorial = read(args.factorial) if args.factorial else None
        if args.command == 'preflight': result = preflight(m,parent,factorial)
        else:
            require(args.state is not None, '--state required')
            if args.command == 'init': result = initialize(args.state,m)
            elif args.command == 'reserve-confirmation': result = reserve(args.state,m)
            else:
                require(args.evidence is not None, '--evidence required')
                ev = read(args.evidence)
                if args.command == 'compare': result = compare(args.state,m,ev)
                else:
                    require(parent is not None, '--parent required')
                    result = promote(args.state,m,parent,ev,factorial)
        print(json.dumps(dict(ok=True,result=result),sort_keys=True,allow_nan=False))
        return 0
    except (GateError,OSError,ValueError,TypeError,KeyError,IndexError) as error:
        print(json.dumps(dict(ok=False,error=str(error)),sort_keys=True))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
