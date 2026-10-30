"""Reporting F1/F2 evaluation of gate-rejected lanes, at user request (2026-10-29).

Runs each study's original, unmodified execute() into evaluation/<study>/ instead of
its runs/ directory. Gates are still computed and recorded in reporting_gates.json,
but a failed gate no longer stops the run before F1/final. Original runs are copied
(not moved), so their dev/confirmation checkpoints are reused under identical manifests.
"""
import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path, PosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVAL = HERE / 'evaluation'
STUDIES = dict(point4=(ROOT / 'analysis/e2000fs_point4_advance', 'full_20261029_v1', ('linear_leaf', 'leaf_refit', 'stable_retrain')),
               transfer=(ROOT / 'analysis/e2000fs_feat30_transfer', 'full_20261028_v1', ('center', 'recency', 'recent_window')))
RUN_STATE = {'status.json', 'results.json', 'REPORT.md', 'completion.json', 'run.lock', 'run.log', 'launch.json'}


class Redirect(PosixPath):
    """The study directory, except that `/ 'runs'` points at the reporting evaluation root."""
    target = None

    def __truediv__(self, key):
        return self.target if key == 'runs' else Path(str(self)) / key


def seed(src, dst):
    if dst.exists():
        return
    shutil.copytree(src, dst, ignore=lambda d, names: [n for n in names if n in RUN_STATE and Path(d) == src])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, choices=STUDIES)
    parser.add_argument('--lane', required=True)
    args = parser.parse_args()
    study, run_name, lanes = STUDIES[args.study]
    assert args.lane in lanes
    root = EVAL / args.study / run_name
    root.mkdir(parents=True, exist_ok=True)
    original = study / 'runs' / run_name
    baseline = root / 'baseline'
    if not baseline.exists():
        shutil.copytree(original / 'baseline', baseline)  # completed comparator, status kept
    seed(original / args.lane, root / args.lane)
    spec = importlib.util.spec_from_file_location(f'reporting_{args.study}_run', study / 'run.py')
    run = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = run
    spec.loader.exec_module(run)
    redirect = Redirect(run.HERE)
    redirect.target = EVAL / args.study
    run.HERE = redirect
    audit = {}
    real_development, real_gate = run.m.development_gate, run.m.gate

    def development_gate(*a):
        audit['development_passed'] = bool(real_development(*a))
        return True

    def gate(*a, **k):
        audit.setdefault('gate_calls', []).append(bool(real_gate(*a, **k)))
        return True

    run.m.development_gate, run.m.gate = development_gate, gate
    if args.study == 'point4':
        # The lane manifest pins baseline files by path; the copies are byte-identical to the originals.
        real_load = run.previous.baseline_load

        def baseline_load(directory, *a):
            loaded, hashes = real_load(directory, *a)
            return loaded, {k.replace(str(root), str(original)): v for k, v in hashes.items()}

        run.previous.baseline_load = baseline_load
    run.execute(args.lane, False)
    out = root / args.lane
    # development_gate calls the patched gate for dev0, dev1, dev2 and pooled, then confirmation and transferred follow.
    calls = audit.get('gate_calls', [])
    assert len(calls) == 6, calls
    audit.update(gate_calls_order=['dev0', 'dev1', 'dev2', 'pooled', 'confirmation', 'transferred'],
                 development_passed=all(calls[:4]), confirmation_passed=calls[4], transferred=calls[5],
                 note='Gates computed by the original code; failures did not stop F1/final for reporting.')
    results = json.loads((out / 'results.json').read_text())
    for key in ('development_passed', 'confirmation_passed', 'transferred'):
        if key in results:
            results[key] = audit[key]
    results['reporting_evaluation'] = audit['note']
    (out / 'results.json').write_text(json.dumps(results, indent=1, default=str) + '\n')
    completion = json.loads((out / 'completion.json').read_text())
    completion['sha256'][str(out / 'results.json')] = run.d.digest(out / 'results.json')
    (out / 'completion.json').write_text(json.dumps(completion, indent=1) + '\n')
    (out / 'reporting_gates.json').write_text(json.dumps(audit, indent=1) + '\n')
    print(json.dumps(dict(lane=args.lane, mse=results['combined']['mse'], **{k: audit[k] for k in ('development_passed', 'confirmation_passed', 'transferred')})))


if __name__ == '__main__':
    main()
