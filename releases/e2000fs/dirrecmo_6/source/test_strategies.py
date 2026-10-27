import numpy as np
import pandas as pd
import pytest

import data
import evaluate
import strategies as S
from data import H, LAG


def toy(n=160, seed=0):
    rng = np.random.default_rng(seed)
    inp = S.Inputs(rng.normal(size=(n, 5)), rng.normal(size=(n, H, 2)), 400 + 50 * rng.normal(size=(n, LAG)))
    Y = 400 + 50 * rng.normal(size=(n, H))
    return inp, Y


@pytest.fixture(scope='module')
def builder():
    with data.P.relocated_builder(data.P.source_root()) as p:
        yield p, p.A.filled_hourly()


def test_speed_features_lag0_future_poison(builder):
    p, ace = builder
    features, _, _ = data.feature_names()
    origin = pd.Timestamp('2025-09-10T12:00:00Z')
    base = data.retime([origin], p, ace, features)[data.speed_columns(features)]
    poisoned = ace.copy()
    poisoned[poisoned.index > origin] = 5000.
    assert base.equals(data.retime([origin], p, poisoned, features)[data.speed_columns(features)])
    poisoned[origin] = 5000.  # positive control: the origin hour itself is an input now (no 2h lag)
    assert data.retime([origin], p, poisoned, features)['v_lag0'].iloc[0] == 5000.


def test_ace_features_lag0_future_poison(builder, monkeypatch):
    p, ace = builder
    features, _, _ = data.feature_names()
    origin = pd.Timestamp('2025-09-10T12:00:00Z')
    cols = data.ace_columns(features)
    base = data.retime([origin], p, ace, features)[cols]
    hourly = p.HR.hourly

    def poison(after):
        def patched(df, col='timestamp_utc'):
            out = hourly(df, col)
            num = out.select_dtypes('number').columns
            out.loc[out.index > after, num] = 7e3
            return out
        return patched

    monkeypatch.setattr(p.HR, 'hourly', poison(origin))
    assert np.allclose(base, data.retime([origin], p, ace, features)[cols], equal_nan=True)
    monkeypatch.setattr(p.HR, 'hourly', poison(origin - pd.Timedelta(hours=1)))
    assert data.retime([origin], p, ace, features)['ace_speed_kms_last'].iloc[0] == 7e3


def test_past_window_ends_at_origin():
    t = pd.date_range('2025-01-01', periods=100, freq='h', tz='UTC')
    s = pd.Series(np.arange(100.), index=t)
    w = data.past_window(s, t[[50]])
    assert w.shape == (1, LAG) and w[0, -1] == 50 and w[0, 0] == 50 - LAG + 1


def test_prefill_uses_only_earlier_origins():
    C = np.array([[np.nan, 1.], [2., np.nan], [np.nan, np.nan], [5., 6.]], np.float32)
    E = np.full((4, H, 1), np.nan, np.float32)
    E[1, 3, 0] = 9.
    C2, E2 = data.prefill(C, E)
    assert np.isnan(C2[0, 0]) and C2[2, 0] == 2 and C2[1, 1] == 1 and C2[2, 1] == 1
    assert np.isnan(E2[0, 3, 0]) and E2[2, 3, 0] == 9 and E2[3, 3, 0] == 9 and np.isnan(E2[3, 4, 0])


def test_blocks_and_feature_widths():
    inp, Y = toy(4)
    T = S.Block.trajectory(inp, Y)
    assert S.blocks(24) == [(1, 24), (25, 48), (49, 72)]
    with pytest.raises(ValueError):
        S.blocks(7)
    rec = S.RecMO(6, None, None, 't').features(inp, T, 7, 12)
    assert rec.shape[1] == 5 + LAG + 6 * 2 + 1
    assert np.allclose(rec[:, 5:5 + LAG], T[:, 6:6 + LAG])  # lag ends at horizon 6
    assert np.allclose(S.RecMO(1, None, None, 't').features(inp, T, 1, 1)[:, 5:5 + LAG], inp.past)
    assert S.DirRecMO(6, None, None, 't').features(inp, T, 13, 18).shape[1] == 5 + LAG + 12 + 12
    assert S.DirMO(72, None, None, 't').features(inp, T, 1, 72).shape[1] == 5 + LAG + 144


@pytest.mark.parametrize('name', ['Recursive', 'DirRec', 'MIMO', 'RecMO-24', 'DirMO-24',
                                  'DirRecMO-24', 'Rectify', 'Stratify-DirMO-RecMO-24'])
def test_inference_never_reads_unfilled_future(name, monkeypatch):
    inp, Y = toy()
    seen = []
    real = S._predict

    def guarded(model, X, width):
        seen.append(np.isfinite(X).all())
        return real(model, X, width)

    monkeypatch.setattr(S, '_predict', guarded)
    cfg = S.recipe(n_estimators=3, n_jobs=2)
    preds, *_ = S.fit_predict(name, cfg, inp, Y, inp, Y, {'x': inp}, origins=pd.date_range(
        '2025-01-01', periods=len(inp), freq='3h', tz='UTC'))
    assert preds['x'].shape == (len(inp), H) and np.isfinite(preds['x']).all() and all(seen)


def test_prediction_independent_of_training_targets_after_fit():
    inp, Y = toy()
    cfg = S.recipe(n_estimators=3, n_jobs=2)
    m = S.build('RecMO-12', cfg, S.Logger()).fit(inp, Y, inp, Y)
    a = m.predict(inp)
    Y[:] = -1e6  # truth after the origin is not an inference input
    assert np.array_equal(a, m.predict(inp))


def test_purged_folds_gap():
    origins = pd.date_range('2024-01-01', periods=400, freq='3h', tz='UTC')
    for train, held in S.purged_folds(origins):
        lo, hi = origins[held[0]], origins[held[-1]]
        o = origins[train]
        assert len(np.intersect1d(train, held)) == 0
        assert ((o < lo - pd.Timedelta('72h')) | (o > hi + pd.Timedelta('72h'))).all()


def test_internal_split_respects_cutoffs():
    origins = pd.date_range('2026-02-20', '2026-05-28', freq='3h', tz='UTC')
    b = data.Bundle(origins, np.zeros((len(origins), 1)), np.zeros((len(origins), H, 1)),
                    np.zeros((len(origins), LAG)), ['c'], ['e'])
    fit, val = data.internal_split(b, None)
    assert (origins[fit] + pd.Timedelta('72h') < data.TRAIN_END).all()
    assert (origins[val] >= data.VALID_START).all() and (origins[val] <= data.VALID_END).all()
    assert (origins[data.final_split(b)] + pd.Timedelta('72h') < data.CUTOFF).all()


def test_high_mse_hand_computed():
    y = np.array([500., 550., 600., 700.])
    err = np.array([1., -2., 3., 4.])
    r = evaluate.high_mse(err, y)
    assert r['n'] == 3 and r['mse'] == pytest.approx((4 + 9 + 16) / 3) and r['bias'] == pytest.approx(5 / 3)
    r = evaluate.high_mse(err, y, mask=np.array([True, True, False, True]))
    assert r['n'] == 2 and r['mse'] == pytest.approx(10.)


def test_select_and_f1_splits_respect_leaderboard_cutoffs():
    origins = pd.date_range('2025-11-01', '2026-05-28', freq='3h', tz='UTC')
    b = data.Bundle(origins, np.zeros((len(origins), 1)), np.zeros((len(origins), H, 1)),
                    np.zeros((len(origins), LAG)), ['c'], ['e'])
    fit, val = data.select_split(b)
    last = pd.Timedelta('72h')
    assert (origins[fit] + last < data.SELECT_TRAIN_END).all()
    assert (origins[val] >= data.SELECT_VALID_START).all() and (origins[val] + last < data.F1_TRAIN_END).all()
    assert (origins[data.f1_split(b)] + last < data.F1_TRAIN_END).all()
    assert data.F1_TRAIN_END == pd.Timestamp('2026-02-26T00:00:00Z')


def test_combined_grid_matches_leaderboard_folds():
    import json
    from huggingface_hub import hf_hub_download
    import run
    folds = pd.read_parquet(hf_hub_download('tlabtlab/sunrun-lb-store', 'folds.parquet', repo_type='dataset'))
    names = json.loads((data.HERE / 'outputs/cache/names.json').read_text())
    f1 = pd.DatetimeIndex(np.load(data.HERE / 'outputs/cache/f1.npz')['origins'], tz='UTC')
    of = pd.DatetimeIndex(np.load(data.HERE / 'outputs/cache/official.npz')['origins'], tz='UTC')
    assert names['varying'] == list(data.VARYING)
    frame = evaluate.prediction_frame(f1.append(of), np.full((len(f1) + len(of), H), 400.))
    assert evaluate.new_scoring().validate(frame, folds) == []
    assert (folds.set_index('origin_last_input_utc').loc[f1.strftime('%Y-%m-%dT%H:%M:%SZ'), 'period'] == 'F1').all()
