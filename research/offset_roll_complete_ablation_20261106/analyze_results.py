"""Paired, read-only-to-runs analysis of the preregistered ablation campaign.

Writes report artifacts under ROOT/analysis. Bootstrap resamples 168-hour circular
origin blocks independently within each official period; the same draw is used
for every arm and seed. Labels enter this reporting process only, never fitting.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

SCOPES = ('total', 'f1', 'f2', 'f3')
METRICS = ('s', 'timse', 'rmse', 'trmse')


def scorer_at(root):
    path = root / 'competition.py'
    spec = importlib.util.spec_from_file_location('ablation_competition', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def variant(recipe):
    return str(recipe.get('variant') or re.sub(r'_s\d+$', '', recipe['experiment']))


def seed_of(recipe):
    return int(recipe.get('config', {}).get('seed', recipe.get('seed', 42)))


def validation_gate(out, name, state):
    """Do not consume a trainer's last score before supervisor validation."""
    entry = state.get(name, {})
    if entry.get('state') in {'failed', 'cancelled_by_user'}:
        return False, entry.get('state')
    exit_path = out / 'train.exit'
    if not exit_path.is_file() or exit_path.read_text().strip() != '0':
        return False, 'successful train.exit missing'
    complete_path = out / 'complete.json'
    complete = json.loads(complete_path.read_text()) if complete_path.is_file() else {}
    if complete.get('publish') == 'not_required' or entry.get('state') == 'trained_validated':
        return True, None
    return False, 'supervisor validation pending'


def load_runs(root, assignment, scorer, context, partial):
    runs, missing = [], {}
    statepath = root / 'scheduler-state.json'
    state = json.loads(statepath.read_text()) if statepath.is_file() else {}
    for recipe in assignment['experiments']:
        name = recipe['experiment']
        out = root / 'runs' / name
        try:
            ready, reason = validation_gate(out, name, state)
            paths = [out / 'local_score.json', out / 'predictions.parquet',
                     out / 'paired_no_offset_score.json', out / 'no_offset_predictions.parquet']
            if not ready or not all(p.is_file() for p in paths):
                missing[name] = reason or 'required artifacts missing'
                continue
            arrays, scores = {}, {}
            for label, scorepath, predpath in [('full', paths[0], paths[1]), ('no_offset', paths[2], paths[3])]:
                frame = pd.read_parquet(predpath)
                array = scorer.align_predictions(frame, context)
                if not (np.isfinite(array) & (array > 0)).all():
                    raise ValueError(f'{name}/{label}: invalid or missing forecast')
                actual = scorer.score_competition(frame, context)
                stored = json.loads(scorepath.read_text())
                stored = stored.get('competition', stored)
                for scope in SCOPES:
                    for metric in METRICS:
                        if not np.isclose(actual[scope][metric], stored[scope][metric], rtol=1e-10, atol=1e-8):
                            raise ValueError(f'{name}/{label}: stored score mismatch {scope}/{metric}')
                for scope in SCOPES:
                    selected = np.ones(len(array), dtype=bool) if scope == 'total' else context['period'] == scope.upper()
                    error = array[selected] - context['truth'][selected]
                    finite = np.isfinite(error)
                    actual[scope]['mae'] = float(np.mean(np.abs(error[finite])))
                    actual[scope]['bias'] = float(np.mean(error[finite]))
                arrays[label], scores[label] = array, actual
            runs.append(dict(recipe=recipe, arrays=arrays, scores=scores, out=out))
        except (OSError, ValueError, KeyError, TypeError) as error:
            if not partial:
                raise
            missing[name] = str(error)
    if missing and not partial:
        raise ValueError(f'{len(missing)} runs incomplete; use --partial only for explicitly provisional reporting')
    return runs, missing


def metric_weights(truth, indices):
    """Linear MSE/TIMSE coefficients, recomputing empirical quantiles per draw.

    Duplicate sampled origins retain multiplicity. With identity indices these
    coefficients reproduce competition.score_competition exactly.
    """
    n, leads = truth.shape
    mse = np.zeros((n, leads))
    timse = np.zeros_like(mse)
    for h in range(leads):
        take = indices[np.isfinite(truth[indices, h])]
        y = truth[take, h]
        if len(y) < 2 or np.ptp(y) == 0:
            raise ValueError('Bootstrap draw has undefined TIMSE population')
        order = np.argsort(y, kind='stable')
        sorted_y = y[order]
        q = np.quantile(y, np.arange(101) / 100, method='linear')
        pos = np.searchsorted(sorted_y, q[:-1], side='left')
        increments = np.zeros(len(y))
        np.add.at(increments, pos, np.diff(q) / (len(y) - pos) / (q[-1] - q[0]))
        tail_weights = np.cumsum(increments)
        mse[:, h] = np.bincount(take, minlength=n) / len(y)
        timse[:, h] = np.bincount(take[order], weights=tail_weights, minlength=n)
    return mse, timse


def scores_from_weights(squared_errors, mse_weights, timse_weights):
    rmse = np.sqrt(np.einsum('anh,nh->ah', squared_errors, mse_weights, optimize=True))
    trmse = np.sqrt(np.einsum('anh,nh->ah', squared_errors, timse_weights, optimize=True))
    return dict(s=np.sqrt(rmse * trmse).mean(1), rmse=rmse.mean(1),
                trmse=trmse.mean(1), timse=(trmse ** 2).mean(1))


def circular_draw(context, rng, block_hours):
    pieces = []
    for period in ('F1', 'F2', 'F3'):
        rows = np.flatnonzero(context['period'] == period)
        # Official origin grid is six-hourly. Keep period populations fixed.
        if len(rows) > 1 and not np.all(np.diff(context['origins'][rows].asi8) == 6 * 3600 * 10**9):
            raise ValueError('Expected the official six-hour origin grid')
        length = max(1, block_hours // 6)
        starts = rng.integers(0, len(rows), size=int(np.ceil(len(rows) / length)))
        positions = (starts[:, None] + np.arange(length)) % len(rows)
        pieces.append(rows[positions.ravel()[:len(rows)]])
    return np.concatenate(pieces)


def bootstrap_scores(context, arrays, count, block_hours, seed=20261104):
    errors = np.where(np.isfinite(context['truth'])[None],
                      (arrays - context['truth'][None]) ** 2, 0.)
    exact = scores_from_weights(errors, *metric_weights(context['truth'], np.arange(len(context['origins']))))
    result = {key: np.empty((count, len(arrays))) for key in METRICS}
    rng = np.random.default_rng(seed)
    accepted = attempts = 0
    while accepted < count:
        attempts += 1
        if attempts > max(100, count * 10):
            raise RuntimeError('Too many undefined bootstrap populations')
        try:
            weights = metric_weights(context['truth'], circular_draw(context, rng, block_hours))
        except ValueError:
            continue
        values = scores_from_weights(errors, *weights)
        for key in METRICS:
            result[key][accepted] = values[key]
        accepted += 1
        if accepted % 100 == 0:
            print(json.dumps(dict(stage='analysis_bootstrap', unit=accepted, units=count)), flush=True)
    return exact, result, attempts - accepted


def bh_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    adjusted = np.minimum.accumulate((values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty(len(values))
    result[order] = np.minimum(adjusted, 1.)
    return result.tolist()


def distribution(draws, observed):
    if not len(draws):
        return dict(ci95=None, p_centered_bootstrap=None)
    lo, hi = np.quantile(draws, [.025, .975])
    p = (1 + np.count_nonzero(np.abs(draws - observed) >= abs(observed))) / (len(draws) + 1)
    return dict(ci95=[float(lo), float(hi)], p_centered_bootstrap=float(p))


def diagnostics(truth, base, full):
    valid = np.isfinite(truth)
    residual = truth - base
    change = full - base
    delta = (full - truth) ** 2 - (base - truth) ** 2
    identity = change ** 2 - 2 * change * residual
    masks = {'all': valid, 'below350': valid & (truth < 350),
             '350_to550': valid & (truth >= 350) & (truth < 550),
             '550_plus': valid & (truth >= 550),
             'missed550': valid & (truth >= 550) & (base < 550),
             'predicted550': valid & (truth >= 550) & (base >= 550)}
    result = {}
    for key, mask in masks.items():
        if not mask.any():
            result[key] = {'n': 0}
            continue
        result[key] = dict(n=int(mask.sum()), mse_before=float(np.mean(residual[mask] ** 2)),
            mse_after=float(np.mean((full[mask] - truth[mask]) ** 2)),
            mae_before=float(np.mean(abs(residual[mask]))),
            mae_after=float(np.mean(abs(full[mask] - truth[mask]))),
            bias_before=float(np.mean(-residual[mask])),
            bias_after=float(np.mean(full[mask] - truth[mask])),
            delta_mse=float(delta[mask].mean()), variance_cost=float((change[mask] ** 2).mean()),
            alignment_term=float((-2 * change[mask] * residual[mask]).mean()),
            overall_mse_contribution=float(delta[mask].sum() / valid.sum()),
            identity_max_abs_error=float(abs(delta[mask] - identity[mask]).max()))
    return result


def clipping_diagnostics(run, context):
    result = {}
    for period in ('F1', 'F2', 'F3'):
        path = run['out'] / f'{period}_raw_predictions.npz'
        if not path.exists():
            result[period] = {'status': 'raw artifact missing'}
            continue
        with np.load(path) as z:
            raw, component, preclip = z['raw'], z['component'], z['preclip']
            origins = pd.DatetimeIndex(pd.to_datetime(z['origins'], utc=True))
            selected = np.flatnonzero(context['period'] == period)
            positions = origins.get_indexer(context['origins'][selected])
            if (positions < 0).any():
                raise ValueError('Raw forecast does not cover the official scoring origins')
            truth = context['truth'][selected]
            valid = np.isfinite(truth)
            raw, component, preclip = raw[positions], component[positions], preclip[positions]
            final = run['arrays']['full'][selected]
            result[period] = dict(
                component_changed_pairs=int(np.count_nonzero((raw != component) & valid)),
                final_changed_pairs=int(np.count_nonzero((preclip != final) & valid)),
                component_delta_mse=float(np.mean(((component - truth)**2 - (raw - truth)**2)[valid])),
                final_delta_mse=float(np.mean(((final - truth)**2 - (preclip - truth)**2)[valid])))
    return result


def make_contrasts(runs):
    baselines = {seed_of(r['recipe']): i for i, r in enumerate(runs)
                 if r['recipe'].get('stage') == 'baseline' or variant(r['recipe']) == 'baseline'}
    contrasts = []
    for i, run in enumerate(runs):
        seed = seed_of(run['recipe'])
        if seed not in baselines:
            raise ValueError(f'Missing matched baseline for seed {seed}')
        contrasts.append(dict(name=variant(run['recipe']), kind='versus_baseline', seed=seed,
                              treatment=2 * i, reference=2 * baselines[seed], recipe=run['recipe']))
        contrasts.append(dict(name=variant(run['recipe']), kind='offset_effect', seed=seed,
                              treatment=2 * i, reference=2 * i + 1, recipe=run['recipe']))
    # Eight factorial cells are observed through four trained loss settings and
    # their paired offset-off predictions, or explicit offset-off runs.
    cells = {}
    for i, run in enumerate(runs):
        c = run['recipe']['config']
        if c.get('exclude_features') or c.get('block_size', 24) != 24 or c.get('spectral_hessian_only', False):
            continue
        if any(c.get(k, default) != default for k, default in
               [('offset_scope', 'horizon'), ('offset_window', 336), ('alpha', .25),
                ('clip_component', True), ('clip_final', True)]):
            continue
        seed = seed_of(run['recipe'])
        w, s = int(c.get('tail_weight', 4) == 4), int(c.get('spectral', True))
        cells[(seed, w, s, 0)] = 2 * i + 1
        if c.get('offset', True):
            cells[(seed, w, s, 1)] = 2 * i
    interactions = []
    for seed in sorted(baselines):
        for axes in [(0, 1), (0, 2), (1, 2), (0, 1, 2)]:
            terms = []
            for w in (0, 1):
                for s in (0, 1):
                    for o in (0, 1):
                        bits = (w, s, o)
                        if any(bits[k] != 1 for k in range(3) if k not in axes):
                            continue
                        key = (seed, w, s, o)
                        if key not in cells:
                            terms = []
                            break
                        terms.append((cells[key], (-1) ** sum(1 - bits[k] for k in axes)))
                    else:
                        continue
                    break
                else:
                    continue
                break
            if len(terms) == 2 ** len(axes):
                interactions.append(dict(name=' × '.join(('tail', 'spectral', 'offset')[k] for k in axes),
                                         seed=seed, terms=terms))
    return contrasts, interactions


def narrative(report, runs):
    """Interpret predeclared contrasts without selecting or fitting a new model."""
    baselines = [r for r in runs if r['recipe'].get('stage') == 'baseline' or variant(r['recipe']) == 'baseline']
    base_names = {variant(r['recipe']) for r in baselines}
    rows = report['contrasts']
    comparisons = [r for r in rows if r['kind'] == 'versus_baseline' and r['variant'] not in base_names]
    def supported(row, sign):
        metric = row['metrics']['s']
        ci = metric['ci95']
        if len(row['seeds']) < 3 or ci is None:
            return False
        if metric.get('bh_q', 1.) >= .05:
            return False
        return (ci[0] > 0 and min(metric['per_seed']) > 0 and metric['mean_delta'] >= .10) if sign > 0 else (ci[1] < 0 and max(metric['per_seed']) < 0)
    important = sorted([r for r in comparisons if supported(r, 1)], key=lambda r: -r['metrics']['s']['mean_delta'])
    harmful = sorted([r for r in comparisons if supported(r, -1)], key=lambda r: r['metrics']['s']['mean_delta'])
    uncertain = [r for r in comparisons if not supported(r, 1) and not supported(r, -1)]
    def detail(row):
        m = row['metrics']['s']
        return (f"`{row['variant']}`: 평균 ΔS={m['mean_delta']:+.6f}, "
                f"95% CI={m['ci95']}, seed별 ΔS={m['per_seed']}" +
                (f", BH q={m['bh_q']:.4g}" if 'bh_q' in m else ''))
    lines = ['## 성능 원인 판정', '']
    if len(baselines) == 1:
        lines += ['이번 분석은 seed42의 필수 구성 제거 진단입니다. 원인 후보의 방향·크기를 비교하되 seed 안정성이나 일반적인 성능 우월성은 확정하지 않습니다. 특징·파라미터·상호작용 탐색은 수행하지 않았습니다.', '']
    if report['status'] != 'complete' or report['bootstrap']['replicates'] == 0:
        lines.append('현재는 중간 진단입니다. 미완료 실험 또는 신뢰구간 부재 때문에 최고 성능의 원인을 확정하지 않습니다.')
    elif important:
        lines.append('고정 기준선에서 제거·대체했을 때 성능이 일관되게 악화된 구성이 확인되었습니다. '
                     '가장 큰 조건부 기여는 ' + ', '.join(f"`{r['variant']}`" for r in important[:3]) + ' 비교에서 나타납니다.')
    else:
        lines.append('현재 실험으로는 세 seed와 신뢰구간이 함께 지지하는 기준선 구성의 우위를 확정하지 못했습니다.')
    lines += ['', '아래 ΔS는 변경 모델−기준선이며 양수는 변경 후 악화입니다. '
              'offset 효과 표만 적용 후−적용 전입니다. 기여는 이 기준선과 평가 기간에 조건부이며 서로 더할 수 없습니다.',
              '', '### 기준선 성능', '']
    for run in sorted(baselines, key=lambda r: seed_of(r['recipe'])):
        score = run['scores']['full']
        lines.append(f"- seed {seed_of(run['recipe'])}: Total S={score['total']['s']:.6f}; " +
                     ', '.join(f"{p.upper()}={score[p]['s']:.6f}" for p in ('f1', 'f2', 'f3')) +
                     f"; RMSE={score['total']['rmse']:.6f}, TRMSE={score['total']['trmse']:.6f}.")
    if baselines:
        values = [r['scores']['full']['total']['s'] for r in baselines]
        lines.append(f"- 기준선 seed 평균 S={np.mean(values):.6f}, 최악 S={max(values):.6f}; "
                     f"실행 수={len(values)}. 기존 제출과의 순위 비교는 동일 관측으로 재채점된 경우에만 유효합니다.")
    lines += ['', '### 일관되게 지지되는 구성의 기여', '']
    lines += ['- ' + detail(r) + '. 이 제거·대체는 기준선 성능을 낮췄습니다.' for r in important[:10]] or ['- 세 seed의 방향·95% CI·개별 특징의 BH 기준을 모두 만족한 비교가 아직 없습니다.']
    lines += ['', '### 해로운 구성 후보와 불확실한 중복성', '']
    lines += ['- ' + detail(r) + '. 해당 변경이 개선되어 원래 구성의 유효성을 지지하지 않습니다. 자동 채택하지 않습니다.' for r in harmful[:10]]
    exact_zero = [r for r in uncertain if max(abs(v) for v in r['metrics']['s']['per_seed']) < 1e-10]
    if exact_zero:
        lines.append('- 측정된 ΔS가 0인 비교: ' + ', '.join(f"`{r['variant']}`" for r in exact_zero) +
                     '. 이 표본에서 효과가 관측되지 않았으며 일반적인 불필요성을 입증하지는 않습니다.')
    lines.append(f'- 방향이 seed마다 다르거나 신뢰구간·다중검정 근거가 부족한 비교 {len(uncertain)}개. '
                 'CI가 0을 포함한다는 이유만으로 중복·동등성을 확정하지 않습니다.')
    for row in sorted(uncertain, key=lambda r: -abs(r['metrics']['s']['mean_delta']))[:5]:
        lines.append('- 미확정: ' + detail(row) + '.')
    lines += ['', '### 핵심 손실·offset 상호작용과 기간 집중', '']
    for row in report['factorial_interactions']:
        ci = row['ci95']
        excludes = ci is not None and (ci[0] > 0 or ci[1] < 0)
        consistent = min(row['per_seed']) > 0 or max(row['per_seed']) < 0
        label = '비가산 효과가 지지됨' if excludes and consistent and len(row['seeds']) == 3 else '비가산 효과 미확정'
        lines.append(f"- {row['name']}: 차이의 차이={row['mean_delta']:+.6f}, CI={ci}; {label}. "
                     '나머지 인자는 켠 상태이며 이 값을 독립 기여로 더하지 않습니다.')
    if not report['factorial_interactions']:
        lines.append('- 필요한 핵심 조합이 아직 모두 완료되지 않아 상호작용을 계산하지 못했습니다.')
    offset_rows = [r for r in rows if r['kind'] == 'offset_effect' and r['variant'] in base_names]
    selected = offset_rows + (important[:3] if important else sorted(comparisons, key=lambda r: -abs(r['metrics']['s']['mean_delta']))[:3])
    for row in selected:
        periods = {p: row['period_deltas'][p]['s'] for p in ('f1', 'f2', 'f3')}
        largest = max(periods, key=lambda p: abs(periods[p]))
        lines.append(f"- {row['kind']} / `{row['variant']}`: " +
                     ', '.join(f'{p.upper()} ΔS={v:+.6f}' for p, v in periods.items()) +
                     f'; 절대 변화는 {largest.upper()}에서 가장 큽니다. 기간별 값은 기술통계이며 Total의 가산 분해가 아닙니다.')
    lines += ['', '### offset이 오차를 바꾼 방식', '',
              '실제 예측 변화 k에 대해 ΔMSE = E[k²] − 2E[k·(정답−보정 전 예측)]를 확인합니다. '
              'clipping을 포함한 실제 k를 쓰므로 항등식 자체는 clipping 유무와 무관합니다.']
    for name in ('all', 'below350', '350_to550', '550_plus', 'missed550', 'predicted550'):
        entries = [report['diagnostics'][r['recipe']['experiment']][name] for r in baselines]
        entries = [e for e in entries if e['n']]
        if not entries:
            continue
        mean = lambda key: float(np.mean([e[key] for e in entries]))
        lines.append(f"- {name}: 평균 ΔMSE={mean('delta_mse'):+.6f} = "
                     f"보정 크기 비용 {mean('variance_cost'):+.6f} + 잔차 정렬항 {mean('alignment_term'):+.6f}; "
                     f"MAE {mean('mae_before'):.6f}→{mean('mae_after'):.6f}, "
                     f"bias {mean('bias_before'):+.6f}→{mean('bias_after'):+.6f}.")
    if baselines:
        identities = [report['diagnostics'][r['recipe']['experiment']]['all']['identity_max_abs_error'] for r in baselines]
        lines.append(f'- 항등식 최대 절대 잔차={max(identities):.3g}. 각 행은 동일 seed 집합의 평균이며 구간별 표본 수는 JSON에 기록했습니다.')
        clipping = [v for r in baselines for v in report['clipping'][r['recipe']['experiment']].values() if 'component_changed_pairs' in v]
        if clipping:
            lines.append(f"- 기준선 clipping 활성 pair: 모델 출력 {sum(v['component_changed_pairs'] for v in clipping)}, "
                         f"최종 출력 {sum(v['final_changed_pairs'] for v in clipping)} (seed·기간 합계).")
    lines += ['', '이 해석은 고정 기준선에서의 조건부 기여입니다. 제거 목록은 위의 사전 고정 규칙으로 한 번 확정하며, 최종 조합 결과에 따른 추가 탐색이나 외부 제출은 하지 않습니다.', '']
    return lines


def adjust_families(rows, count):
    """Separate individual-feature, feature-group and remaining hypothesis families."""
    families = defaultdict(list)
    for row in rows:
        if row['kind'] != 'versus_baseline' or row['category'] == 'baseline':
            continue
        family = row['category'] if row['category'] in {'feature', 'group'} else 'other'
        row['multiplicity_family'] = family
        families[family].append(row)
    if count:
        for family in families.values():
            for row, q in zip(family, bh_adjust([r['metrics']['s']['p_centered_bootstrap'] for r in family])):
                row['metrics']['s']['bh_q'] = q


def contribution_supported(row):
    metric = row['metrics']['s']
    ci = metric.get('ci95')
    return (sorted(row['seeds']) == [42, 43, 44]
            and len(metric['per_seed']) == 3 and min(metric['per_seed']) > 0
            and metric['mean_delta'] >= .10 and ci is not None and ci[0] > 0
            and metric.get('bh_q', 1.) < .05)


def classify_rows(rows, complete):
    decisions = {}
    for row in rows:
        if row['kind'] != 'versus_baseline' or row['category'] == 'baseline':
            continue
        supported = contribution_supported(row) if complete else False
        decisions[row['variant']] = dict(
            state=('contribution_confirmed' if supported else 'contribution_unconfirmed') if complete else 'pending',
            category=row['category'], supported=supported,
            metrics=row['metrics']['s'], seeds=row['seeds'],
            meaning='conditional contrast against original A; replacement contrasts are diagnostic only')
    return decisions


def freeze_removal_manifest(root, assignment, report):
    """Freeze the sole post-audit configuration; never revise it from final scores."""
    recipes = assignment['experiments']
    groups = assignment.get('groups', {})
    features = list(dict.fromkeys(n for values in groups.values() for n in values))
    if len(features) != 30:
        raise ValueError('Removal decisions require the registered partition of all 30 baseline features')
    decisions = report['decisions']
    by_variant = {variant(r): r for r in recipes}
    retained, bundles, individual = set(), [], []
    for name, decision in decisions.items():
        if not decision['supported']:
            continue
        recipe = by_variant[name]
        excluded = recipe['config'].get('exclude_features', [])
        if recipe.get('category') == 'group':
            retained.update(excluded)
            bundles.append(dict(contrast=name, features=excluded))
        elif recipe.get('category') == 'feature':
            retained.update(excluded)
            individual.extend(excluded)

    def keep(name):
        if name not in decisions:
            raise ValueError('Required component contrast absent: ' + name)
        return decisions[name]['supported']

    config = dict(exclude_features=[n for n in features if n not in retained],
                  tail_weight=4 if keep('loss_t1_sp1_offset') else 1,
                  spectral=keep('loss_t4_sp0_offset'),
                  offset=keep('loss_t4_sp1_nooffset'),
                  clip_component=keep('clip_component_off'),
                  clip_final=keep('clip_final_off'),
                  spectral_hessian_only=False)
    # Architecture and numeric fitting settings are fixed controls, not tuned by
    # the diagnostic replacement scores. Only optional components are removed.
    result = dict(status='frozen' if retained else 'blocked_no_retained_features',
                  baseline='original A (no B1 persistence blend)', config=config, final_config=config,
                  retained_features=[n for n in features if n in retained],
                  individually_supported_features=sorted(set(individual)),
                  retained_feature_bundles=bundles,
                  component_evidence={k: decisions[k] for k in (
                      'loss_t1_sp1_offset', 'loss_t4_sp0_offset', 'loss_t4_sp1_nooffset',
                      'clip_component_off', 'clip_final_off')},
                  rule='all seeds positive, mean delta >= .10, CI95 lower > 0, family BH q < .05',
                  warning='Unconfirmed contribution is not evidence of zero effect. Final confirmation is descriptive; no score-driven restoration.',
                  final_experiments=[])
    if retained:
        result['final_experiments'] = [dict(
            experiment=f'final_pruned_s{seed}', variant='final_pruned', category='confirmation',
            stage='ablation', seed=seed, family='pooling', config=dict(config, seed=seed), threads=4)
            for seed in (42, 43, 44)]
    path = root / 'analysis' / 'removal_manifest.json'
    path.parent.mkdir(exist_ok=True)
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != result:
            raise ValueError('Refusing to revise the frozen removal manifest; preserve the original audit')
        return previous
    with path.open('x') as stream:
        stream.write(encoded)
    return result


def decision_narrative(report):
    lines = ['## 사전 고정 유지·제거 판정', '']
    ready = report['status'] == 'complete' and report['bootstrap']['replicates'] >= 2000
    if not ready:
        return lines + ['전체 대조와 2,000회 bootstrap 완료 전에는 유지·제거를 확정하지 않습니다.', '']
    lines += ['유지 근거: 3 seed 모두 ΔS>0, 평균 ΔS≥0.10, CI 하한>0, 해당 검정군 BH q<0.05.',
              '특징군의 기여가 확인되면 해당 군을 묶음으로 유지합니다. 개별 기여 확인과는 구분합니다.', '']
    for name, decision in sorted(report['decisions'].items()):
        label = '기여 확인' if decision['supported'] else '기여 미확정'
        lines.append(f'- `{name}`: {label}.')
    if 'removal_manifest' in report:
        manifest = report['removal_manifest']
        lines += ['', f"최종 구성 상태: {manifest['status']}; 유지 특징 {len(manifest['retained_features'])}개.",
                  '제거 목록은 한 번만 고정합니다. 최종 결합 모델의 성능 저하도 보고하며 결과로 반복 복원하지 않습니다.']
    if report.get('final_confirmation'):
        lines += ['', '최종 결합 모델 확인 결과입니다. 원래의 제거 목록과 기준선은 변경하지 않습니다.']
    return lines + ['']


def analyze(root, count=2000, block_hours=168, partial=False, final=False):
    assignment_path = root / 'assignment.global.json'
    if not assignment_path.exists():
        assignment_path = root / 'assignment.json'
    assignment = json.loads(assignment_path.read_text())
    if final:
        manifest = json.loads((root / 'analysis' / 'removal_manifest.json').read_text())
        if manifest['status'] != 'frozen':
            raise ValueError('Final confirmation requires a frozen valid removal manifest')
        assignment = dict(assignment, experiments=[r for r in assignment['experiments'] if r.get('stage') == 'baseline'] + manifest['final_experiments'])
    scorer = scorer_at(root)
    obs = resolve(root, assignment.get('observations', 'observations.parquet'))
    context = scorer.prepare_context(scorer.load_observations(obs))
    runs, missing = load_runs(root, assignment, scorer, context, partial)
    if not runs:
        if not partial:
            raise ValueError('No complete runs to analyze')
        report = dict(status='provisional', completed=0, expected=len(assignment['experiments']),
                      missing=list(missing), missing_reasons=missing, contrasts=[], decisions={},
                      bootstrap=dict(replicates=0, block_hours=block_hours))
        out = root / 'analysis'
        out.mkdir(exist_ok=True)
        (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        (out / 'report.md').write_text('# 전체 ablation 중간 상태\n\n검증 완료 실험이 없어 유지·제거 판정을 보류합니다.\n')
        return report
    arrays = np.stack([r['arrays'][key] for r in runs for key in ('full', 'no_offset')])
    exact, boot, rejected = bootstrap_scores(context, arrays, count, block_hours)
    for i, run in enumerate(runs):
        for j, key in enumerate(('full', 'no_offset')):
            for metric in METRICS:
                if not np.isclose(exact[metric][2 * i + j], run['scores'][key]['total'][metric], rtol=1e-10, atol=1e-8):
                    raise ValueError(f'Bootstrap metric kernel disagrees with official scorer: {metric}')
    contrasts, interactions = make_contrasts(runs)
    grouped = defaultdict(list)
    for contrast in contrasts:
        grouped[(contrast['kind'], contrast['name'])].append(contrast)
    rows = []
    for (kind, name), group in sorted(grouped.items()):
        t, r = [c['treatment'] for c in group], [c['reference'] for c in group]
        item = dict(kind=kind, variant=name, category=group[0]['recipe'].get('category'), seeds=[c['seed'] for c in group], metrics={}, period_deltas={})
        for metric in METRICS:
            values = exact[metric][t] - exact[metric][r]
            draws = (boot[metric][:, t] - boot[metric][:, r]).mean(1)
            item['metrics'][metric] = dict(mean_delta=float(values.mean()),
                std_delta=float(values.std(ddof=1)) if len(values) > 1 else None,
                worst_delta=float(values.max()), per_seed=values.tolist(),
                **distribution(draws, float(values.mean())))
        for scope in SCOPES:
            item['period_deltas'][scope] = {}
            for metric in METRICS:
                values = []
                for c in group:
                    def get(index):
                        return runs[index // 2]['scores'][('full', 'no_offset')[index % 2]][scope][metric]
                    values.append(get(c['treatment']) - get(c['reference']))
                item['period_deltas'][scope][metric] = float(np.mean(values))
        item['single_feature_removal'] = (kind == 'versus_baseline' and
            group[0]['recipe'].get('category') == 'feature' and
            len(group[0]['recipe']['config'].get('exclude_features', [])) == 1)
        rows.append(item)
    adjust_families(rows, count)
    interaction_rows = []
    by_name = defaultdict(list)
    for interaction in interactions:
        by_name[interaction['name']].append(interaction)
    for name, group in by_name.items():
        values = [sum(sign * exact['s'][index] for index, sign in g['terms']) for g in group]
        draws = np.mean([sum(sign * boot['s'][:, index] for index, sign in g['terms']) for g in group], axis=0)
        interaction_rows.append(dict(name=name, seeds=[g['seed'] for g in group], per_seed=values,
                                     mean_delta=float(np.mean(values)), **distribution(draws, float(np.mean(values)))))
    report = dict(status='provisional' if missing else 'complete', completed=len(runs), expected=len(assignment['experiments']),
        missing=list(missing), missing_reasons=missing, scorer_version=scorer.VERSION, observation_file=str(obs),
        observation_identity=dict(bytes=obs.stat().st_size, mtime_ns=obs.stat().st_mtime_ns) if obs.is_file() else {'kind': 'directory'},
        scorer_identity=dict(path=str(root / 'competition.py'), bytes=(root / 'competition.py').stat().st_size),
        bootstrap=dict(replicates=count, block_hours=block_hours, rng_seed=20261104,
            method='paired circular origin blocks stratified by official period; empirical TIMSE quantiles recomputed per draw',
            inference_scope='Total; period deltas are descriptive', rejected_draws=rejected,
            seed_interpretation='same origin resample across seeds; intervals condition on the three fitted seeds'),
        contrasts=rows, factorial_interactions=interaction_rows,
        run_scores={r['recipe']['experiment']: {label: {scope: {metric: score[scope][metric] for metric in (*METRICS, 'mae', 'bias')} for scope in SCOPES} for label, score in r['scores'].items()} for r in runs},
        diagnostics={r['recipe']['experiment']: diagnostics(context['truth'], r['arrays']['no_offset'], r['arrays']['full']) for r in runs},
        clipping={r['recipe']['experiment']: clipping_diagnostics(r, context) for r in runs},
        limitations=['Feature removals estimate conditional effects, not additive causal shares.',
                     'Official-period observations enter only scoring and the preregistered one-pass removal audit; no iterative parameter tuning is performed.',
                     'Block intervals condition on the fitted seeds; single-seed runs do not establish seed stability.',
                     'W is not defined by the copied official scorer; S is reported without relabeling it as W.'])
    report['run_summaries'] = {}
    for name in sorted({variant(r['recipe']) for r in runs}):
        selected = [r for r in runs if variant(r['recipe']) == name]
        report['run_summaries'][name] = {scope: {metric: dict(
            per_seed={str(seed_of(r['recipe'])): float(r['scores']['full'][scope][metric]) for r in selected},
            mean=float(np.mean([r['scores']['full'][scope][metric] for r in selected])),
            std=float(np.std([r['scores']['full'][scope][metric] for r in selected], ddof=1)) if len(selected) > 1 else None,
            worst=float(max(r['scores']['full'][scope][metric] for r in selected)))
            for metric in (*METRICS, 'mae', 'bias')} for scope in SCOPES}
    report['decisions'] = classify_rows(rows, complete=not missing and count >= 2000 and block_hours == 168)
    if not final and report['status'] == 'complete' and count >= 2000 and block_hours == 168:
        report['removal_manifest'] = freeze_removal_manifest(root, assignment, report)
    report['final_confirmation'] = final
    out = root / ('final-analysis' if final else 'analysis')
    out.mkdir(exist_ok=True)
    (out / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    flat = []
    for row in rows:
        flat.append(dict(kind=row['kind'], variant=row['variant'], seeds=','.join(map(str, row['seeds'])),
                         **{f'{k}_delta': v['mean_delta'] for k, v in row['metrics'].items()},
                         s_ci95=json.dumps(row['metrics']['s']['ci95']), s_bh_q=row['metrics']['s'].get('bh_q')))
    pd.DataFrame(flat).to_csv(out / 'contrasts.csv', index=False)
    lines = ['# calib_offset_roll ablation results', '',
        f'Status: **{report["status"]}**; {len(runs)}/{len(assignment["experiments"])} runs complete.', '',
        'All differences are treatment minus reference; positive S means deterioration.']
    lines += decision_narrative(report)
    lines += narrative(report, runs)
    lines += ['', '## 전체 비교표', '',
        '| Contrast | Variant | Seeds | Mean ΔS | Worst ΔS | 95% block CI |',
        '|---|---|---|---:|---:|---|']
    for row in rows:
        s = row['metrics']['s']
        lines.append(f'| {row["kind"]} | {row["variant"]} | {row["seeds"]} | {s["mean_delta"]:.6f} | {s["worst_delta"]:.6f} | {s["ci95"]} |')
    lines += ['', 'Factorial interactions (other factors held on):', '']
    for item in interaction_rows:
        lines.append(f'- {item["name"]}: mean ΔS={item["mean_delta"]:.6f}, CI={item["ci95"]}.')
    lines += ['', 'Interpretation limits:', ''] + ['- ' + x for x in report['limitations']]
    if missing:
        lines += ['', 'Incomplete runs: ' + ', '.join(missing)]
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--block-hours', type=int, default=168)
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--final', action='store_true', help='Compare the once-frozen pruned configuration with A; never revise removal decisions')
    args = parser.parse_args()
    if args.bootstrap < 0 or args.block_hours < 6 or args.block_hours % 6:
        parser.error('bootstrap must be nonnegative; block-hours must be a positive multiple of six')
    report = analyze(args.root.resolve(), args.bootstrap, args.block_hours, args.partial, args.final)
    print(json.dumps({k: report[k] for k in ('status', 'completed', 'expected')}))


if __name__ == '__main__':
    main()
