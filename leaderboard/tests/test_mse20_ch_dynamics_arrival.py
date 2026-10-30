import numpy as np
from experiments.mse20_ch_dynamics import arrival, inputs


def test_sign_rotation_delay_units():
    assert np.isclose(arrival.arrival_hour(0,0,450),149597870.7/(3600*450))
    # Same rotating feature seen 24h later has identical issue-relative crossing.
    first=arrival.arrival_hour(24,-20,600,1)
    second=arrival.arrival_hour(0,-20+24*.55,600,1)
    assert np.isclose(first,second)
    assert arrival.arrival_hour(0,20,600,1)<arrival.arrival_hour(0,-20,600,1)
    assert arrival.arrival_hour(0,20,600,-1)>arrival.arrival_hour(0,-20,600,-1)
    assert arrival.specification()['longitude_sign_ambiguous']


def test_lag_normalization_zero_missing_and_latitude():
    ch=np.zeros((2,120,180),dtype=np.float32)
    ch[1]=np.nan
    result,mask=arrival.transport(ch,list(inputs.CH_COLUMNS))
    assert np.all(result[0,:,:72]==0) and mask[0].all()
    assert np.isnan(result[1,:,:72]).all() and not mask[1,:,:72].any()
    assert np.all(result[1,:,72:]==0)
    col=inputs.CH_COLUMNS.index('ch_t035_lon06_lat1_area')
    ch[:]=np.nan;ch[0,:,col]=1
    result,_=arrival.transport(ch,list(inputs.CH_COLUMNS))
    assert np.nanmax(result[0,:,:24])<=1+1e-6  # Never sums 120 snapshots as parcels.
