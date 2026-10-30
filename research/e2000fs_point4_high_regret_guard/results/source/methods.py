"""Fixed training-only retention objectives and fail-closed acceptance gates."""
import numpy as np

LANES = ('high_anchor', 'teacher_trust', 'high_regret_guard')
KEYS = ('mse', 'mse_lt550', 'mse_ge550')
DESCRIPTIONS = dict(
    high_anchor='gamma15 high training truth else3; exact weighted pseudo-target',
    teacher_trust='gamma3+12exp(-.5((teacher-y)/50)^2); exact weighted pseudo-target',
    high_regret_guard='L=.5e²+1.5(p-t)²+.5Ihigh*max(0,e²-et²)²/10000; mean offset')


def quadratic_targets(lane, y, teacher):
    if lane == 'high_anchor':
        gamma = np.where(y >= 550., 15., 3.)
    elif lane == 'teacher_trust':
        gamma = 3.+12.*np.exp(-.5*((teacher-y)/50.)**2)
    else:
        raise ValueError(lane)
    return (y+gamma*teacher)/(1.+gamma), 1.+gamma


def regret_loss(y, teacher, prediction, high):
    error = prediction-y
    positive = np.maximum(0., error**2-(teacher-y)**2)
    return .5*error**2+1.5*(prediction-teacher)**2+.5*high*positive**2/10000.


def regret_derivatives(y, teacher, prediction, high):
    error = prediction-y
    delta = error**2-(teacher-y)**2
    positive = np.maximum(0., delta)
    gradient = error+3.*(prediction-teacher)+2.*high*error*positive/10000.
    hessian = 4.+2.*high*(positive+2.*error**2*(delta>0))/10000.
    return gradient, hessian


def gate(candidate, reference, improvement=False):
    factors = (.99, .95, 1.) if improvement else (1., 1., 1.)
    return all(candidate.get(k) is not None and reference.get(k) is not None
               and np.isfinite(candidate[k]) and np.isfinite(reference[k])
               and candidate[k] <= reference[k]*factor
               for k, factor in zip(KEYS, factors))


def development_gate(candidate, reference):
    return all(gate(candidate[s], reference[s]) for s in ('dev0', 'dev1', 'dev2')) and gate(candidate['pooled'], reference['pooled'], True)
