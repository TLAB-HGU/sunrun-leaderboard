"""Pin existing local feature artifacts and provenance; never collect data or score truth."""
import argparse
import copy
import json
from pathlib import Path
import sys

import pandas as pd

import gate
import runner
from poison import validate_poison
from contracts import validate_predictions

LB = Path('/home/t-lab01/sunrun/leaderboard')
sys.path.insert(0, str(LB))


def evaluator_hash():
    return gate.digest({name: runner.file_hash(LB / name) for name in (
        'expos/roll3.py', 'expos/arrival.py', 'expos/lbeval.py', 'scorer/scoring.py')})


def pin_champion(experiment_id, contract):
    """Use one explicit comparable local scorer record, not a public averaged score.

    The leader must establish the comparator's scientific eligibility from current
    research/source records; a graph pass alone is insufficient (e.g. a blend).
    """
    entries = gate.graph_entries()
    records = [entry for entry in entries if entry.get('event') == 'lbeval'
               and entry.get('id') == experiment_id]
    gate.require(records, 'no local full-period mean-seed scorer record for comparator')
    event = records[-1]
    gate.require(set(event['per_seed']) == set(map(str, contract['seeds'])),
                 'comparator seed aggregation differs')
    return dict(experiment_id=experiment_id, mse=event['summary']['mse_seedmean'],
                contract_sha256=gate.evaluation_identity(contract), aggregation='mean_seed_mse',
                source_sha256=gate.digest(event))


def pin(draft, spec, out, poison_ref=None):
    """Materialized inputs must already obey source admission/issue-time availability.

    Pinning checks integrity and shape, not the scientific truth of supplied source
    provenance. Independent review of the byte-bound availability receipts remains
    mandatory; an on-disk file alone is never an availability receipt.
    """
    gate.keys(spec, 'train features grids sources blocks')
    sources = []
    for item in spec['sources']:
        gate.keys(item, 'id path availability_path')
        evidence = gate.read(item['availability_path'])
        gate.require(evidence.get('source_id') == item['id'] and evidence.get('passed') is True
                     and evidence.get('byte_bound') is True and evidence.get('causal') is True,
                     'missing reviewed byte-bound causal source receipt')
        sources.append(dict(id=item['id'], sha256=runner.file_hash(item['path']),
                            availability_sha256=runner.file_hash(item['availability_path'])))
    gate.require(sources, 'source receipts required')
    train_path = str(Path(spec['train']).resolve())
    train = pd.read_parquet(train_path)
    runner.validate_training(train, spec['features'])
    cutoff = pd.Timestamp('2026-05-29T00:00:00Z')
    gate.require((train.target_t < cutoff).all(), 'training contains official-era targets')
    gate.require(set(spec['grids']) == {'f1', 'f2', 'f3', 'official'},
                 'freeze all three selection grids and whole final grid before experiments')
    from expos.roll3 import fold_bounds
    refs = {}
    for name, path in spec['grids'].items():
        start, end = (fold_bounds(int(name[1:])) if name != 'official' else (
            pd.Timestamp('2026-05-31T23:00:00Z'), pd.Timestamp('2026-09-26T23:00:00Z')))
        frame = pd.read_parquet(path)
        gate.require('y' not in frame and 'label' not in frame,
                     'prediction input grids must be targetless')
        validate_predictions(runner._grid_frame(frame), pd.date_range(start, end, freq='h'))
        refs[name] = dict(path=str(Path(path).resolve()), sha256=runner.file_hash(path),
                          start=start.isoformat(), end=end.isoformat())
    inputs = dict(version=1, train=dict(path=train_path, sha256=runner.file_hash(train_path)),
                  features=spec['features'], grids=refs, source_availability=sources,
                  blocks=spec['blocks'])
    if poison_ref is not None:
        inputs['poison'] = poison_ref
        validate_poison(inputs, draft['recipe'])
    result = copy.deepcopy(draft)
    result['code_sha256'] = gate.digest(runner.code_hashes())
    result['contract'] = dict(seeds=[0, 1, 2], folds=['1', '2', '3'], cutoff=cutoff.isoformat(),
        grid_sha256=gate.digest({name: {'start': v['start'], 'end': v['end']} for name, v in refs.items()}),
        mask_sha256=runner.mask_hash(train), inputs_sha256=gate.digest(inputs),
        sources=sources, evaluation_sha256=evaluator_hash(),
        aggregation='mean_seed_mse', selection_aggregation='seed_mean_prediction_mse',
        requires_arrival=True)
    result['champion'] = pin_champion(draft['champion']['experiment_id'], result['contract'])
    gate.manifest(result)
    runner.select_features(inputs['features'], inputs['blocks'], result['recipe'], train,
                           [pd.read_parquet(ref['path']) for ref in refs.values()])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gate.require(not (out / 'manifest.json').exists() and not (out / 'inputs.json').exists(),
                 'refuse to overwrite frozen inputs')
    # The unpoisoned preparation is intentionally non-executable. Recompute all
    # enabled feature paths with real temporal perturbations before final pinning.
    gate.atomic(out / ('inputs.json' if poison_ref else 'inputs.unchecked.json'), inputs)
    if poison_ref:
        gate.atomic(out / 'manifest.json', result)
    gate.atomic(out / 'provenance-files.json', spec['sources'])
    return result, inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', required=True)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--poison-ref', help='JSON {path,sha256} from actual feature-builder probes')
    args = parser.parse_args()
    pin(gate.read(args.draft), gate.read(args.spec), args.out,
        gate.read(args.poison_ref) if args.poison_ref else None)
    print(json.dumps(dict(ok=True, out=str(Path(args.out).resolve()))))


if __name__ == '__main__':
    main()
