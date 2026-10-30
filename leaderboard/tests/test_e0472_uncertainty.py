import numpy as np
import pandas as pd
import pytest

from experiments.e0472_ablation_hpo.uncertainty import paired_block_ci, paired_block_ci_files


def frames(n=56):
    origins = pd.date_range('2025-12-01', periods=n, freq='3h', tz='UTC')
    baseline = pd.DataFrame({'origin': np.repeat(origins, 72), 'h': np.tile(np.arange(1, 73), n),
                             'pred': 410., 'y': 400.})
    candidate = baseline.copy()
    candidate['pred'] = 405.
    return candidate, baseline


def test_known_paired_improvement_deterministic_and_order_invariant():
    c, b = frames()
    result = paired_block_ci(c.sample(frac=1, random_state=9), b)
    assert result['status'] == 'estimated'
    assert result['mse_delta'] == -75
    assert result['ci95'] == [-75, -75]
    assert result['origins_per_block'] == 24 and result['nonoverlapping_blocks'] == 2
    assert result == paired_block_ci(c, b)
    assert result['used_for_gate'] is False


def test_identical_predictions_have_zero_ci():
    _, b = frames()
    result = paired_block_ci(b, b)
    assert result['mse_delta'] == 0 and result['ci95'] == [0, 0]


def test_block_support_is_explicit_not_fake_precision():
    c, b = frames(8)
    result = paired_block_ci(c, b)
    assert result['status'] == 'inconclusive' and result['ci95'] is None
    assert result['mse_delta'] == -75 and result['block_hours_used'] == 24


@pytest.mark.parametrize('change', ['keys', 'mask', 'target', 'duplicate'])
def test_refuses_unpaired_or_inconsistent_inputs(change):
    c, b = frames()
    if change == 'keys':
        c.origin += pd.Timedelta(hours=1)
    elif change == 'mask':
        c.loc[0, 'y'] = np.nan
    elif change == 'target':
        c.loc[0, 'y'] += 1
    else:
        c = pd.concat([c, c.iloc[:1]])
    with pytest.raises(ValueError):
        paired_block_ci(c, b)


def test_missing_observations_weight_pairs_not_origins():
    c, b = frames()
    c.loc[:71, 'y'] = b.loc[:71, 'y'] = np.nan
    c.loc[72:143, 'pred'] = 400
    result = paired_block_ci(c, b)
    assert result['n_observed_pairs'] == 55*72
    assert result['mse_delta'] == pytest.approx((-100-54*75)/55)
    assert result['ci95'][0] <= result['ci95'][1]


def test_explicit_files_and_official_rejection(tmp_path):
    c, b = frames()
    c.to_parquet(tmp_path/'candidate.parquet')
    b.to_parquet(tmp_path/'baseline.parquet')
    assert paired_block_ci_files(tmp_path/'candidate.parquet', tmp_path/'baseline.parquet')['mse_delta'] == -75
    c.origin += pd.Timedelta(days=365)
    b.origin += pd.Timedelta(days=365)
    with pytest.raises(ValueError, match='preofficial'):
        paired_block_ci(c, b)
