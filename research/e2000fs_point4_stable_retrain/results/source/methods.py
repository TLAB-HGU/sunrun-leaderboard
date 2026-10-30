"""Frozen pointweight4 adaptations and fail-closed transfer gates."""
import warnings
import numpy as np

LANES = ('linear_leaf', 'leaf_refit', 'stable_retrain')
KEYS = ('mse', 'mse_lt550', 'mse_ge550')


def scaling(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean, scale = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.)
    return mean, scale


def stable_targets(y, teacher):
    return .25*y + .75*teacher, np.full(len(y), 4.)


def topology(model):
    def node(tree):
        if 'split_index' not in tree:
            return ('leaf', tree['leaf_index'])
        return (tree['split_feature'], tree['threshold'], tree['decision_type'],
                tree['default_left'], tree['missing_type'],
                node(tree['left_child']), node(tree['right_child']))
    return [node(t['tree_structure']) for t in model.dump_model()['tree_info']]


def gate(candidate, reference, improvement=False):
    factors = (.99, .95, 1.) if improvement else (1., 1., 1.)
    return all(candidate.get(k) is not None and reference.get(k) is not None
               and np.isfinite(candidate[k]) and np.isfinite(reference[k])
               and candidate[k] <= reference[k]*factor
               for k, factor in zip(KEYS, factors))


def development_gate(candidate, reference):
    return all(gate(candidate[s], reference[s]) for s in ('dev0', 'dev1', 'dev2')) and gate(candidate['pooled'], reference['pooled'], True)
