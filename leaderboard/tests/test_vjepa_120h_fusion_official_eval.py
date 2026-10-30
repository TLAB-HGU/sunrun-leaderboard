"""Official scoring must fail closed before any truth access or publication."""

import numpy as np
import pandas as pd
import pytest

from experiments.mse20_four_regimes import protocol
from experiments.vjepa_120h_fusion import official_eval
from scorer import scoring


def frozen_run(tmp_path):
    for name in ('predictions.parquet', 'baseline-predictions.parquet'):
        (tmp_path / name).write_bytes(b'prediction-artifact')
    protocol.write_json(tmp_path / 'prediction-validation.json', {'passed': True})
    frozen = {'passed': True, 'official_truth_access': False, 'n_origins': 2833, 'n_rows': 203976,
              'predictions': protocol.file_ref(tmp_path / 'predictions.parquet'),
              'baseline_predictions': protocol.file_ref(tmp_path / 'baseline-predictions.parquet'),
              'validation': protocol.file_ref(tmp_path / 'prediction-validation.json')}
    protocol.write_json(tmp_path / 'predictions-frozen.json', frozen)
    return frozen


def test_scoring_rejects_unfrozen_without_reading_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, 'read_parquet', lambda *a, **kw: pytest.fail('must not read scoring inputs'))
    with pytest.raises(FileNotFoundError):
        official_eval.score_run(tmp_path)


def test_freeze_checks_all_hashes_and_truth_boundary(tmp_path):
    frozen = frozen_run(tmp_path)
    assert official_eval.frozen_predictions(tmp_path)['passed']
    (tmp_path / 'predictions.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='frozen artifact changed'):
        official_eval.frozen_predictions(tmp_path)
    frozen['official_truth_access'] = True
    protocol.write_json(tmp_path / 'predictions-frozen.json', frozen)
    with pytest.raises(ValueError, match='truth boundary'):
        official_eval.frozen_predictions(tmp_path)


def test_failed_core_validation_prevents_scoring(tmp_path, monkeypatch):
    frozen = frozen_run(tmp_path)
    protocol.write_json(tmp_path / 'prediction-validation.json', {'passed': False})
    frozen['validation'] = protocol.file_ref(tmp_path / 'prediction-validation.json')
    protocol.write_json(tmp_path / 'predictions-frozen.json', frozen)
    monkeypatch.setattr(pd, 'read_parquet', lambda *a, **kw: pytest.fail('must not read scoring inputs'))
    with pytest.raises(ValueError, match='core prediction validation failed'):
        official_eval.score_run(tmp_path)


def test_stale_verification_prevents_scoring(tmp_path, monkeypatch):
    frozen_run(tmp_path)
    protocol.write_json(tmp_path / 'verification.json', {'passed': True, 'official_truth_access': False,
                        'predictions_frozen_sha256': 'wrong'})
    monkeypatch.setattr(pd, 'read_parquet', lambda *a, **kw: pytest.fail('must not read scoring inputs'))
    with pytest.raises(ValueError, match='verification freeze differs'):
        official_eval.score_run(tmp_path)


@pytest.mark.parametrize('mse,valid,independent,eligible', [
    (official_eval.BASELINE_MSE, True, True, False),
    (official_eval.BASELINE_MSE - .001, True, True, True),
    (4000., False, True, False), (4000., True, False, False),
    (np.nan, True, True, False), (np.inf, True, True, False),
    (4000., 1, True, False)])
def test_publication_gate_strict_and_all_validation(mse, valid, independent, eligible):
    assert official_eval.publication_gate(mse, valid, independent)['publication_eligible'] is eligible


def test_paired_deltas_use_same_keys_and_observed_mask():
    origins = ['2026-06-01T00:00:00Z', '2026-06-01T01:00:00Z']
    key = pd.DataFrame({'origin_last_input_utc': np.repeat(origins, 72), 'horizon_hours': np.tile(np.arange(1, 73), 2)})
    truth = key.assign(target_kms=400., was_missing=np.tile([0, 1], 72))
    candidate, reference = key.assign(pred_kms=401.), key.assign(pred_kms=403.)
    folds = pd.DataFrame({'origin_last_input_utc': origins, 'regime': ['low_low_low', 'high_high_high']})
    result = official_eval.paired_comparison(candidate, reference, truth, folds, scoring)
    assert result['n_pairs'] == 144
    assert result['mse_delta'] == result['observed_mse_delta'] == -8.
    assert set(result['horizon_mse_delta'].values()) == {-8.}
    assert set(result['regime_mse_delta'].values()) == {-8.}
    with pytest.raises(ValueError, match='paired scoring support mismatch'):
        official_eval.paired_comparison(candidate.iloc[:-1], reference, truth, folds, scoring)
