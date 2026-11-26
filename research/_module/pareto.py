"""Training-only common-descent proposals with exact finite-tree risk gates."""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.special import log_ndtr

from .common import CUTOFFS
from .gbm import load_booster, predict_block
from .mm import TrainingScore, ZeroTrainingRisk
from .relevance import RelevanceCDF, fit_cdf

ALPHAS = (1., .5, .25, .125, .0625, .03125, .015625)


class ParetoRisk(TrainingScore):
    """Frozen official-training TIMSE and actual-fit lognormal tail relevance."""

    def __init__(self, target, original, delta=5., eps=1e-6):
        super().__init__(target)
        original = np.asarray(original, dtype=np.float64)
        if original.shape != self.target.shape or np.any(self.target <= 0):
            raise ValueError('Positive filled targets and matching original labels required')
        self.delta = delta
        self.cdfs = [fit_cdf(column) for column in original.T]
        relevance = np.column_stack([
            np.minimum(-log_ndtr(-(np.log(self.target[:, h])-cdf['log_scale'])/cdf['sigma']),
                       -np.log(eps)) for h, cdf in enumerate(self.cdfs)])
        self.rbar = relevance.mean(axis=0)
        if not np.isfinite(relevance).all() or np.any(self.rbar <= 0):
            raise ValueError('Invalid continuous tail relevance')
        self.relevance = relevance / self.rbar

    def risks(self, prediction):
        error = self.error(prediction)
        mse, timse, score = self.terms(prediction)
        smooth = np.sqrt(error**2 + self.delta**2)-self.delta
        return {'s': float(score.mean()), 'mse': float(mse.mean()), 'mae': float(np.abs(error).mean()),
                    'timse': float(timse.mean()), 'tailmae': float((self.relevance*np.abs(error)).mean()),
                    'smooth_mae': float(smooth.mean()), 'smooth_tailmae': float((self.relevance*smooth).mean())}

    def gradients(self, prediction, include_tail=False):
        error = self.error(prediction)
        n, h = error.shape
        smooth = error/np.sqrt(error**2+self.delta**2)/(n*h)
        gradients = [self.curvature(prediction)*error/h, 2*error/(n*h), smooth,
                     2*self.tail*error/h]
        if include_tail:
            gradients.append(self.relevance*smooth)
        return np.stack(gradients)


def min_norm_simplex(gradients):
    """Enumerate the faces of a small simplex; verify the projection KKT test."""
    flat = np.asarray(gradients, dtype=np.float64).reshape(len(gradients), -1)
    if not np.isfinite(flat).all():
        raise ValueError('Nonfinite gradient')
    gram = flat @ flat.T
    best = None
    for count in range(1, len(flat)+1):
        for face in itertools.combinations(range(len(flat)), count):
            k = gram[np.ix_(face, face)]
            system = np.block([[k, np.ones((count, 1))], [np.ones((1, count)), np.zeros((1, 1))]])
            solution = np.linalg.lstsq(system, np.r_[np.zeros(count), 1.], rcond=None)[0][:count]
            if np.min(solution) < -1e-10 or abs(solution.sum()-1) > 1e-8:
                continue
            weights = np.zeros(len(flat))
            weights[list(face)] = np.maximum(solution, 0)
            weights /= weights.sum()
            value = float(weights @ gram @ weights)
            if best is None or value < best[0]:
                best = value, weights
    if best is None:
        raise ValueError('Simplex solve failed')
    value, weights = best
    dots = gram @ weights
    tolerance = 1e-8*max(float(np.max(np.abs(gram))), np.finfo(float).tiny)
    if np.min(dots-value) < -tolerance:
        raise ValueError('Simplex projection KKT failed')
    return weights, {'norm_squared': value, 'kkt_min': float(np.min(dots-value)), 'tolerance': tolerance}


def gate(candidate, current, initial, include_tail=False):
    names = ['s', 'mse', 'mae', 'timse'] + (['tailmae'] if include_tail else [])
    reasons = []
    for name in names:
        tolerance = 1e-10*max(1., abs(initial[name]))
        if not np.isfinite(candidate[name]):
            reasons.append(name+'_nonfinite')
        elif candidate[name] > min(current[name], initial[name])+tolerance:
            reasons.append(name+'_increase')
    if candidate['s'] >= current['s']-1e-10*max(1., abs(initial['s'])):
        reasons.append('s_no_strict_decrease')
    return reasons


def continue_booster(model, x, xq, loss, params, seed, threads, iterations, mode):
    """Append accepted one-tree surrogates; every retry starts from identical bytes."""
    if mode not in ('s4', 'mgda4', 'mgda4_tail'):
        raise ValueError('Unknown CPC mode')
    initial_rounds = model.num_boosted_rounds()
    original_prefix = json.loads(bytes(model.save_raw('json')))['learner']['gradient_booster']['model']['trees']
    current = model.inplace_predict(x).astype(np.float64).reshape(loss.target.shape)
    initial = loss.risks(current)
    matrix = xgb.DMatrix(x, label=loss.target.astype(np.float32), nthread=threads)
    records, attempts, accepted = [], 0, 0
    stop = 'proposal_budget_exhausted'
    try:
        norms = np.linalg.norm(loss.gradients(current).reshape(4, -1), axis=1)
    except ZeroTrainingRisk:
        norms = np.zeros(4)
    if np.any(norms == 0):
        stop = 'zero_initial_gradient_or_risk'
        iterations = 0
        kappa = None
    else:
        initial_gs = loss.gradients(current)[0]/norms[0]
        kappa = float(np.sqrt(np.mean(loss.error(current)**2))/np.sqrt(np.mean(initial_gs**2)))
    for iteration in range(iterations):
        try:
            normalized = loss.gradients(current)/norms[:, None, None]
        except ZeroTrainingRisk:
            stop = 'zero_risk_lead'
            break
        if mode == 's4':
            weights, qp = np.array([1., 0., 0., 0.]), None
        else:
            weights, qp = min_norm_simplex(normalized)
        direction = np.einsum('j,jnh->nh', weights, normalized)
        if not np.any(direction):
            stop = 'zero_common_direction'
            break
        pseudo = current-kappa*direction
        previous = np.frombuffer(bytes(model.save_raw('ubj')), np.uint8)
        before = loss.risks(current)
        record = {'iteration': iteration, 'seed': seed+iteration, 'lambdas': weights.tolist(), 'qp': qp,
                      'free_direction_dots': np.einsum('jnh,nh->j', normalized, direction).tolist(),
                      'before': before, 'attempts': [], 'accepted': False}

        def objective(prediction, _matrix, pseudo=pseudo):
            gradient = np.asarray(prediction, dtype=np.float64).reshape(pseudo.shape)-pseudo
            if not np.isfinite(gradient).all():
                raise ValueError('Nonfinite surrogate gradient')
            return gradient.astype(np.float32), np.ones(pseudo.shape, np.float32)

        for alpha in ALPHAS:
            attempt_started = time.perf_counter()
            candidate = xgb.train(dict(params, seed=seed+iteration, learning_rate=.03*alpha), matrix,
                                  num_boost_round=1, obj=objective,
                                  xgb_model=load_booster(previous, threads))
            attempts += 1
            prediction = candidate.inplace_predict(x).astype(np.float64).reshape(current.shape)
            risk = loss.risks(prediction)
            reasons = gate(risk, before, initial, mode == 'mgda4_tail')
            record['attempts'].append({'alpha': alpha, 'risks': risk, 'reasons': reasons,
                                           'seconds': time.perf_counter()-attempt_started,
                                           'tree_directional_dots': np.einsum('jnh,nh->j', normalized,
                                                                          prediction-current).tolist()})
            if not reasons:
                model, current = candidate, prediction
                accepted += 1
                record['accepted'] = True
                break
        records.append(record)
        if not record['accepted']:
            stop = 'all_alpha_candidates_rejected'
            break
    buffer = np.frombuffer(bytes(model.save_raw('ubj')), np.uint8)
    final_trees = json.loads(bytes(model.save_raw('json')))['learner']['gradient_booster']['model']['trees']
    if final_trees[:initial_rounds] != original_prefix:
        raise ValueError('Continuation changed original tree prefix')
    forecast = model.inplace_predict(xq)
    tick = time.perf_counter()
    replay = load_booster(buffer, threads)
    if replay.num_boosted_rounds() != initial_rounds+accepted or not np.array_equal(replay.inplace_predict(xq), forecast):
        raise ValueError('CPC checkpoint replay failed')
    final = loss.risks(current)
    names = ['s', 'mse', 'mae', 'timse'] + (['tailmae'] if mode == 'mgda4_tail' else [])
    return forecast, buffer, time.perf_counter()-tick, {'replay_max_abs_error': 0., 'cpc': {
        'mode': mode, 'initial_rounds': initial_rounds, 'final_rounds': initial_rounds+accepted,
        'proposed_iterations': len(records), 'tree_fit_attempts': attempts, 'accepted_trees': accepted,
        'stop_reason': stop, 'initial': initial, 'final': final, 'fixed_norms': norms.tolist(), 'kappa': kappa,
        'cdfs': loss.cdfs, 'rbar': loss.rbar.tolist(), 'delta': loss.delta, 'epsilon': 1e-6,
        'strict_training_pareto': all(final[k] < initial[k]-1e-10*max(1.,abs(initial[k])) for k in names),
        'records': records}}


@dataclass(frozen=True)
class ParetoContinuation:
    constant: tuple
    varying: tuple
    reference_weights: Path = Path(__file__).resolve().parents[1]/'current_offset_roll_direct_s'/'weights'
    mode: str = 'mgda4'

    def __post_init__(self):
        if self.mode not in ('s4', 'mgda4', 'mgda4_tail'):
            raise ValueError('Unknown CPC mode')

    def prepare_period(self, table, period):
        lookup = RelevanceCDF(self.constant, self.varying).prepare_period(table, period)
        selected = table.loc[table[f'fit_{period}'].to_numpy(), 'origin']
        return {'period': period, 'lookup': lookup, 'train_start': str(selected.min()), 'train_end': str(selected.max())}

    def fit_block(self, spec, x, xq, y, seed, threads, rounds, context, smoke):
        from current_offset_roll_direct_s.experiment import SPEC as reference
        if spec.block != 24 or x.shape[1] != 215 or spec.context_name is not None:
            raise ValueError('CPC requires the original 215-input, 24-output layout and private context')
        matches = [e for e in context['lookup'] if e['subset'].startswith('smoke_') == bool(smoke)
                   and e['target'].shape == y.shape and np.array_equal(e['target'], y)]
        if len(matches) != 1:
            raise ValueError('CPC requires one exact actual-fit target context')
        entry = matches[0]
        if spec.feature_names(entry['start'], 24) != reference.feature_names(entry['start'], 24):
            raise ValueError('D900 feature order mismatch')
        provenance_path = Path(__file__).resolve().parents[1]/'result'/'current_offset_roll_direct_s'/'logs'/'provenance.json'
        provenance = json.loads(provenance_path.read_text())
        records = [r for r in provenance['runs'] if r['period'] == context['period'] and r['seed'] == seed]
        if len(records) != 1 or provenance.get('objective') != 'direct_s':
            raise ValueError('Missing unique Direct-S provenance')
        record = records[0]
        stored_names = [name.replace('@h', '_h') for name in record['expanded_features'][str(entry['start'])]]
        if (stored_names != spec.feature_names(entry['start'], 24)
                or record['features']['constant'] != list(self.constant)
                or record['features']['varying'] != list(self.varying)
                or record['training_cutoff'] != CUTOFFS[context['period']]
                or record['train_start'] != context['train_start']
                or record['train_end'] != context['train_end']
                or record['source_objective'] != 'direct_s'):
            raise ValueError('Direct-S source provenance mismatch')
        subset = entry['subset'].removeprefix('smoke_')
        path = Path(self.reference_weights)/context['period']/f's{seed}.npz'
        with np.load(path, allow_pickle=False) as checkpoint:
            buffer = checkpoint[f'{subset}_b{entry["start"]//24}'].copy()
        model = load_booster(buffer, threads)
        config = json.loads(model.save_config())['learner']['learner_model_param']
        if model.num_boosted_rounds() != 900 or model.num_features() != 215 or int(config['num_target']) != 24:
            raise ValueError('Reference must be an ordinary D900/215/24 booster')
        params = dict(spec.params, device='cpu', nthread=threads, num_target=24)
        if params.get('tree_method') != 'hist' or params.get('multi_strategy') != 'multi_output_tree':
            raise ValueError('CPC requires CPU hist multi_output_tree')
        result = continue_booster(model, x, xq, ParetoRisk(y, entry['original']), params, seed, threads,
                                  2 if smoke else 180, self.mode)
        result[3]['cpc'].update(reference=str(path), subset=subset, block=entry['start']//24,
                                smoke=bool(smoke), proposal_budget=2 if smoke else 180,
                                guarantee_scope='raw actual-fit predictions only; not offset/clip/query')
        return result

    predict_block = staticmethod(predict_block)
