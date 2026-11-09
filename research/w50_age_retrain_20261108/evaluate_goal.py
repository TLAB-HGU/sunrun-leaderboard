"""Fail-closed completion gate for the ongoing official-S reduction campaign.

Exit 0: audited goal PASS; exit 1: completed scores FAIL; exit 2: evidence PARTIAL.
This evaluator never fits a model or changes workflow/hidden Codex goal state.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from analyze_results import (
    BEST_SINGLE_S,
    HALF_S_TARGET,
    ROOT,
    SEEDS,
    completed_runs,
    write,
)

REQUIRED_CHECKS = (
    'causal_fitting', 'fresh_training', 'single_factor_diff', 'feature_identity',
    'official_score_replay', 'model_replay', 'provenance_complete', 'device_policy',
    'preregistration',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def evidence_gaps(root, results):
    gaps = []
    try:
        runs = completed_runs(root)
        require(results['status'] == 'complete', 'analysis incomplete')
        require(results['seeds'] == list(SEEDS), 'seed set differs from preregistration')
        require(results['best_reference_s'] == BEST_SINGLE_S, 'frozen baseline changed')
        require(results['half_s_target'] == HALF_S_TARGET, '50% threshold changed')
        require(results['bootstrap']['replicates'] == 2000, 'bootstrap count differs')
        require(results['bootstrap']['block_hours'] == 168, 'bootstrap block differs')
        require(results['bootstrap']['seed'] == 20261104, 'bootstrap RNG differs')
        require(results['bootstrap']['period_stratified'] is True, 'unstratified bootstrap')
        require(results['bootstrap']['empirical_timse_quantiles_recomputed'] is True,
                'TIMSE quantiles were not recomputed')
        for receipt in [*results['inputs'], results['observations'], results['scorer']]:
            path = Path(receipt['path'])
            stat = path.stat()
            require(stat.st_size == receipt['bytes'] and stat.st_mtime_ns == receipt['mtime_ns'],
                    f'analysis input changed: {path}')
        for run in runs:
            for period in ('F1', 'F2', 'F3'):
                for stage in ('prior', 'full'):
                    for block in range(3):
                        model = run / f'{period}_models' / stage / f'b{block}.ubj'
                        require(model.is_file() and model.stat().st_size > 0,
                                f'missing trained model: {model}')
        audit = json.loads((root / 'verification.json').read_text())
        require(audit.get('status') == 'pass', 'completion audit did not pass')
        require(bool(audit.get('reviewer')), 'completion audit reviewer missing')
        require(not audit.get('unresolved_required_gaps'), 'required audit gaps remain')
        for name in REQUIRED_CHECKS:
            check = audit.get('checks', {}).get(name, {})
            require(check.get('passed') is True, f'{name}: not verified')
            require(bool(check.get('evidence')), f'{name}: evidence missing')
            for entry in check['evidence']:
                path = Path(entry)
                if not path.is_absolute():
                    path = root / path
                require(path.is_file(), f'{name}: evidence artifact absent: {path}')
    except (KeyError, OSError, ValueError, TypeError) as error:
        gaps.append(str(error))
    return gaps


def performance_failures(results):
    treatment = results['summaries']['treatment']
    control = results['summaries']['control']
    values = treatment['total']['s']['per_seed']
    require(len(values) == 3 and all(math.isfinite(v) and v > 0 for v in values),
            'invalid treatment score population')
    require(len(control['total']['s']['per_seed']) == 3, 'invalid control population')
    failures = []
    if not all(value <= HALF_S_TARGET for value in values):
        failures.append(f'Not all three fresh single-model S scores <= {HALF_S_TARGET}')
    ci = results['paired_reduction_ci95']
    require(len(ci) == 2 and all(math.isfinite(v) for v in ci) and ci[0] <= ci[1],
            'invalid paired confidence interval')
    if ci[0] <= 0:
        failures.append('Paired temporal improvement interval includes zero')
    for metric in ('rmse', 'pooled_mae'):
        if treatment['total'][metric]['mean'] > 1.01 * control['total'][metric]['mean']:
            failures.append(f'{metric} worsens by more than 1%')
    for period in ('f1', 'f2', 'f3'):
        if treatment[period]['s']['mean'] > 1.02 * control[period]['s']['mean']:
            failures.append(f'{period} mean S worsens by more than 2%')
    if not results.get('candidate_gate_passed'):
        failures.append('Preregistered performance-candidate gate failed')
    return failures


def evaluate(root):
    try:
        results = json.loads((root / 'results.json').read_text())
        failures = performance_failures(results)
        gaps = evidence_gaps(root, results)
    except (KeyError, OSError, ValueError, TypeError) as error:
        return {'verdict': 'PARTIAL', 'goal_achieved': False,
                'failures': [], 'gaps': [str(error)], 'exit_code': 2}
    # A definite score failure remains FAIL even if independent audit is pending.
    status = 'FAIL' if failures else ('PARTIAL' if gaps else 'PASS')
    return {'verdict': status, 'goal_achieved': status == 'PASS',
            'failures': failures, 'gaps': gaps, 'exit_code': {'PASS': 0, 'FAIL': 1, 'PARTIAL': 2}[status],
            'fixed_best_single_s': BEST_SINGLE_S, 'required_max_s': HALF_S_TARGET,
            'all_seed_scores': results['summaries']['treatment']['total']['s']['per_seed'],
            'sota_reported_separately': results.get('sota_comparison', {}),
            'workflow_state_modified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = evaluate(args.root)
    write(args.root / 'goal-verdict.json', result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result['exit_code']


if __name__ == '__main__':
    # The unchanged performance-goal command follows an explicit repo-local
    # handoff. --root continues to evaluate this historical study directly.
    import subprocess
    import sys

    handoff = ROOT / 'goal-handoff.json'
    if len(sys.argv) == 1 and handoff.exists():
        active = json.loads(handoff.read_text())
        if active['reference_s'] != BEST_SINGLE_S or active['target_s'] != HALF_S_TARGET:
            raise ValueError('Goal handoff changed the frozen objective')
        evaluator = (ROOT / active['evaluator']).resolve()
        if not evaluator.is_relative_to(ROOT.parent) or evaluator.name != 'evaluate_goal.py':
            raise ValueError('Goal evaluator must remain in the research workspace')
        raise SystemExit(subprocess.run([sys.executable, str(evaluator)], check=False).returncode)
    raise SystemExit(main())
