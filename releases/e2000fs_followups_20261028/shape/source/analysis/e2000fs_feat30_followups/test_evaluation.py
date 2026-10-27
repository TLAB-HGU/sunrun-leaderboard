"""Regression checks for ranking, event semantics and pinned evaluation inputs."""
import json

import numpy as np
import pandas as pd
import pytest

import evaluation as E


def test_peak_ramp_metrics_hand_computed():
    y = np.arange(72, dtype=float)[None, :] + 500
    p = y + 10
    m = E.metrics(y, p)
    assert m['mse'] == 100
    assert m['peak_mae'] == 10
    assert m['peak_bias'] == 10
    assert m['ramp6_mae'] == 0
    assert m['mse_ge550'] == 100
    assert m['bias_ge550'] == 10
    assert m['peak_timing_mae_h'] == 0
    assert m['mse_by_band'] == {'h1-24': 100, 'h25-48': 100, 'h49-72': 100}
    json.dumps(m, allow_nan=False)


def test_declining_ramps_are_clipped_to_zero():
    y = (500 - np.arange(72))[None, :]
    p = (500 - 2 * np.arange(72))[None, :]
    row = E.per_window(y, p).iloc[0]
    assert row.true_ramp6 == row.pred_ramp6 == row.ramp6_ae == 0
    m = E.metrics(y, p)
    assert m['mse_ge550'] is None
    assert m['timing_mae_h'] is None
    assert m['csi'] is None
    json.dumps(m, allow_nan=False)


def test_events_censor_boundaries_and_match_one_to_one():
    y = np.full((1, 72), 400.)
    y[0, [0, 20, 30, 71]] = [600, 650, 660, 700]
    p = np.full_like(y, 400.)
    p[0, 25] = 640
    result = E.metrics(y, p)
    assert result['tp'] == 1
    assert result['fn'] == 1
    assert result['fp'] == 0
    assert result['csi'] == .5
    assert result['timing_mae_h'] == 5


def test_selection_priority_and_zero_parent():
    parent = {'peak_mae': 10., 'ramp6_mae': 20.}
    a = dict(peak_mae=5., ramp6_mae=30., csi=.2, mse=20.)
    b = dict(a, csi=.3, mse=30.)
    c = dict(b, mse=10.)
    assert E.selection_score(a, parent) == 1.
    assert E.selection_key(c, parent) < E.selection_key(b, parent) < E.selection_key(a, parent)
    assert E.selection_key(dict(a, csi=None), parent) > E.selection_key(a, parent)
    assert E.selection_score({'peak_mae': 0., 'ramp6_mae': 0.}, {'peak_mae': 0., 'ramp6_mae': 0.}) == 0


def test_frame_grid_utc_and_validation():
    o = pd.date_range('2026-03-01', periods=2, freq='h', tz='Asia/Seoul')
    p = np.full((2, 72), 400.)
    f = E.frame(o, p)
    assert list(f) == E.KEY + ['pred_kms']
    assert len(f) == 144 and not f.duplicated(E.KEY).any()
    assert f.iloc[0].origin_last_input_utc == '2026-02-28T15:00:00Z'
    assert f.horizon_hours.tolist() == list(range(1, 73)) * 2
    for bad in [o[::-1], [o[0], o[0]], o.tz_localize(None), o + pd.Timedelta(minutes=1)]:
        with pytest.raises(ValueError):
            E.frame(bad, p)
    with pytest.raises(ValueError):
        E.frame(o, p[:, :-1])
    p[0, 0] = np.nan
    with pytest.raises(ValueError):
        E.metrics(p, p)


def test_official_score_uses_local_hash_pinned_tables(tmp_path):
    origins = pd.date_range('2026-06-01', periods=2, freq='h', tz='UTC')
    pred = E.frame(origins, np.full((2, 72), 410.))
    truth = pred.rename(columns={'pred_kms': 'target_kms'}).assign(target_kms=400., was_missing=0)
    folds = pd.DataFrame({E.KEY[0]: origins.strftime('%Y-%m-%dT%H:%M:%SZ'), 'regime': 'low_low_low', 'period': 'F2'})
    naive = E.frame(origins, np.full((2, 72), 420.))
    manifest = {'hashes': {str(E.SCORER): E.digest(E.SCORER)}}
    for name, data in [('truth', truth), ('folds', folds), ('naive', naive)]:
        path = tmp_path / f'{name}.parquet'
        data.to_parquet(path, index=False)
        manifest['hashes'][str(path)] = E.digest(path)
    score = E.official_score(pred, manifest)
    assert score['mse'] == 100.
    assert score['skill_vs_naive'] == .75
    json.dumps(score, allow_nan=False)
    (tmp_path / 'truth.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        E.official_score(pred, manifest)
    (tmp_path / 'truth.parquet').unlink()
    with pytest.raises(FileNotFoundError, match='no download attempted'):
        E.official_score(pred, manifest)
