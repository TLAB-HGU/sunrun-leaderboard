import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('build',Path(__file__).with_name('build.py'))
build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)

class CausalityTest(unittest.TestCase):
    def test_turbulence_delay_and_poison(self):
        m=build.load('aceturb')
        index=pd.date_range('2026-01-01',periods=30,freq='h',tz='UTC')
        raw=pd.DataFrame({'ace_bt_nt':np.arange(30)+2.,'ace_speed_kms':np.arange(30)+300.,'ace_density_cm3':np.arange(30)+3.},index=index)
        origin=index[20];cut=origin-pd.Timedelta(hours=2)
        base=m.build_turb_features(raw).loc[origin]
        future=raw.copy();future.loc[future.index>cut]*=100
        pd.testing.assert_series_equal(base,m.build_turb_features(future).loc[origin])
        past=raw.copy();past.loc[cut]*=2
        self.assertFalse(np.allclose(base,m.build_turb_features(past).loc[origin],equal_nan=True))
        self.assertEqual(m.build_turb_features(raw).index[0],index[0]+pd.Timedelta(hours=2))

class SequenceTest(unittest.TestCase):
    def test_mambats_mask_fill_and_delayed_window(self):
        import sys
        sys.path.insert(0,str(Path(__file__).parent))
        from mambats import prepare,CHANNELS
        idx=pd.date_range('2022-08-22',periods=1000,freq='h',tz='UTC')
        raw=pd.DataFrame({c:np.arange(1000,dtype=float)+1 for _,c in CHANNELS})
        raw.insert(0,'timestamp_utc',idx)
        raw.loc[100:106,CHANNELS[0][1]]=np.nan
        arrays,_=prepare(raw.copy(),idx[-1]+pd.Timedelta(hours=2))
        self.assertTrue(np.isnan(arrays['values'][106,0]))
        self.assertEqual(arrays['values'][105,0],100)
        self.assertEqual(arrays['missing'][105,0],1)
        self.assertEqual(arrays['origin_idx'][-1]-2,len(idx)-1)
        probe=arrays['origin_idx'][0]-2
        changed=raw.copy();changed.loc[changed.index>probe,CHANNELS[0][1]]=1e6
        other,_=prepare(changed,idx[-1]+pd.Timedelta(hours=2))
        np.testing.assert_equal(arrays['values'][probe-719:probe+1],other['values'][probe-719:probe+1])

if __name__=='__main__':unittest.main()
