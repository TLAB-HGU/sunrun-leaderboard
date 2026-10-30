import copy
import numpy as np
import pandas as pd
import pytest
from experiments.mse20_ch_dynamics import inputs


def frames(n=1100):
    index = pd.date_range('2020-01-01', periods=n, freq='h', tz='UTC')
    x = np.arange(n)
    ace = pd.DataFrame({'filled_speed_kms': 430+70*np.sin(x/35), 'was_missing': np.zeros(n)}, index=index)
    ch = pd.DataFrame({c: .01+.005*np.sin(x/50+j) for j, c in enumerate(inputs.CH_COLUMNS)}, index=index)
    raw = pd.DataFrame({'ace_density_cm3': 5+np.cos(x/10), 'ace_temperature_k': 1e5+x,
                        'ace_bt_nt': 5+np.sin(x/17), 'ace_speed_kms': ace.filled_speed_kms}, index=index)
    return inputs.rebuild_frames(ace, ch, raw)


def test_explicit_shapes_zero_missing_age():
    f = frames()
    origins = f['speed'].index[[1000, 1001]]
    f['ch'].iloc[1000, 0] = 0
    f['ch'].iloc[999, 1] = np.nan
    f['ch'] = f['ch'].drop(f['ch'].index[998])
    a = inputs.make_inputs(f, origins)
    assert len(inputs.CH_COLUMNS) == 180
    assert sum('_lon' in c and '_lat' in c and c.endswith('_area') for c in inputs.CH_COLUMNS) == 108
    assert a['ch'].shape == (2,120,180) and a['ace'].shape == (2,168,4)
    assert a['transport'].shape == (2,72,75) and a['recurrence'].shape == (2,72,3)
    assert a['ch'][0,-1,0] == 0 and a['ch_mask'][0,-1,0]
    assert not a['ch_mask'][0,-2,1] and not a['ch_slot_valid'][0,-3]
    assert a['ch_last_known_age'][0,0] == 0
    assert a['ace'][0,-1,0] == 0
    assert a['ace'][0,-1,2] > 1e5  # Physics is NOT current-speed centered.


@pytest.mark.parametrize('key', ['ch','base','physics','quality','ace','raw','rec_full'])
def test_unknown_features_denied(key):
    f = frames(); f[key]['future_T_regime'] = 1
    with pytest.raises(ValueError, match='schema|allowlisted'):
        inputs.make_inputs(f, f['speed'].index[[1000]])


@pytest.mark.parametrize('source', ['ace','ch','raw'])
@pytest.mark.parametrize('drop', [False, True])
def test_raw_mutation_before_rebuild_is_causal(source, drop):
    f = frames(); origin = f['speed'].index[[1000]]
    before = inputs.make_inputs(f, origin)
    raw = {k: f[k].copy() for k in ('ace','ch','raw')}
    future = raw[source].index > origin[0]
    assert future.sum() > 72
    if drop: raw[source] = raw[source].loc[~future]
    else: raw[source].loc[future] = 1e7
    rebuilt = inputs.rebuild_frames(**raw)
    after = inputs.make_inputs(rebuilt, origin)
    for key in ('ace','ch','recurrence','transport','anchor','ch_last_known_age'):
        np.testing.assert_array_equal(before[key], after[key])
    # Content, not object identity, controls rebuilding: past changes propagate.
    raw = {k: f[k].copy() for k in ('ace','ch','raw')}
    raw[source].iloc[1000,0] += 100
    changed = inputs.make_inputs(inputs.rebuild_frames(**raw), origin)
    key = {'ace':'ace', 'ch':'ch', 'raw':'ace'}[source]
    assert not np.array_equal(before[key], changed[key], equal_nan=True)


def test_training_pool_cutoff_full_labels_and_stride():
    f = frames(); cutoff=f['speed'].index[950]
    original = inputs.training_origins(f,cutoff,cap=1000,step=3)
    assert (original+pd.Timedelta(hours=72)<=cutoff).all()
    assert ((original.asi8//3_600_000_000_000)%3==0).all()
    changed=copy.deepcopy(f);changed['speed'].loc[changed['speed'].index>cutoff]=1e9
    assert original.equals(inputs.training_origins(changed,cutoff,cap=1000,step=3))
    f['speed'].iloc[900]=np.nan
    pool=inputs.training_origins(f,cutoff,cap=1000)
    assert not ((pool<f['speed'].index[900]) & (pool+pd.Timedelta(hours=72)>=f['speed'].index[900])).any()


def test_pinned_ace_timestamp_column_format():
    f=frames()
    ace=f['ace'].rename_axis('timestamp_utc').reset_index()
    rebuilt=inputs.rebuild_frames(ace,f['ch'],f['raw'])
    a=inputs.make_inputs(f,f['speed'].index[[1000]])
    b=inputs.make_inputs(rebuilt,f['speed'].index[[1000]])
    for key in ('ace','ch','recurrence','transport','anchor'):
        np.testing.assert_array_equal(a[key],b[key])
