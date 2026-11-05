import importlib.util
import json
from pathlib import Path

import direct as d
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

HERE = Path(__file__).parent


def original():
    spec = importlib.util.spec_from_file_location('original_direct',HERE/'source_archive/e2000_direct_s42.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feature_order_original_wide(tmp_path,monkeypatch):
    old = original()
    feature_file = HERE.parents[1]/'leaderboard/experiments/cumulative_lane12_20261024/materialized_v3/features_v3.json'
    assert d.FEATURES == [n for n in json.loads(feature_file.read_text())['features'] if n != 'h']
    assert len(d.feature_names()) == 1530
    rng = np.random.default_rng(17)
    c = rng.normal(size=(3,len(d.CONSTANT))).astype(np.float32)
    e = rng.normal(size=(3,72,len(d.VARYING))).astype(np.float32)
    frame = pd.DataFrame({'origin':np.repeat(pd.date_range('2024-01-01',periods=3,tz='UTC'),72),'h':np.tile(np.arange(1,73),3)})
    values = {n:(e[:,:,d.VARYING.index(n)].ravel() if n in d.VARYING else np.repeat(c[:,d.CONSTANT.index(n)],72)) for n in d.FEATURES}
    frame = pd.concat([frame,pd.DataFrame(values)],axis=1)
    frame.to_parquet(tmp_path/'features.parquet')
    monkeypatch.setattr(old,'DATA',tmp_path)
    _, expected = old.wide('features.parquet',d.FEATURES)
    np.testing.assert_array_equal(d.wide(c,e),expected)


def test_direct_finite_fit_matches_original():
    old = original()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100,9)).astype(np.float32)
    y = rng.normal(450,100,size=100).astype(np.float32)
    actual,_ = d.fit(x,y,3,1)
    expected = old.fit_horizon(x,y,3)
    np.testing.assert_allclose(actual.inplace_predict(x),expected.predict(x),rtol=0,atol=0)


def test_pooled_iterator_quantiles_training_parity():
    rng = np.random.default_rng(18)
    x = rng.integers(0,4,size=(50,7)).astype(np.float32)
    x[0,0] = np.nan
    y = rng.normal(400,100,size=(50,5)).astype(np.float32)
    y[::3,::2] = np.nan
    rows,leads = np.nonzero(np.isfinite(y))
    dense = np.column_stack([x[rows],leads+1]).astype(np.float32)
    matrix = xgb.QuantileDMatrix(dense,label=y[rows,leads],nthread=1,max_bin=256)
    expected = xgb.train(dict(d.PARAMS,nthread=1),matrix,num_boost_round=3)
    actual,history = d.fit(x,y,3,1,pooled=True,batch_size=13)
    np.testing.assert_allclose(actual.inplace_predict(dense),expected.inplace_predict(dense),rtol=0,atol=0)
    assert len(history['train']['rmse']) == 3


def test_iterator_emits_each_observed_cell_in_original_order():
    x = np.arange(12,dtype=np.float32).reshape(4,3)
    y = np.array([[np.nan,2],[3,4],[5,np.nan],[7,8]])
    iterator = d.ObservedCells(x,y,batch_size=2)
    for _ in range(2):
        iterator.reset()
        batches, labels = [],[]
        def accept(data,label,batches=batches,labels=labels):
            batches.append(data.copy()); labels.append(label.copy())
        while iterator.next(accept):
            pass
        rows,leads = np.nonzero(np.isfinite(y))
        np.testing.assert_array_equal(np.vstack(batches),np.column_stack([x[rows],leads+1]))
        np.testing.assert_array_equal(np.concatenate(labels),y[rows,leads])


@pytest.mark.parametrize('pooled',[False,True])
def test_no_observed_targets_rejected(pooled):
    with pytest.raises(ValueError,match='No observed'):
        d.fit(np.ones((3,2)),np.full((3,2) if pooled else (3,),np.nan),2,1,pooled)
