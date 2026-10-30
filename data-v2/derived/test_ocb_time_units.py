"""Pandas-3 regression: OCB selection compares nanoseconds consistently."""
import importlib.util
from pathlib import Path
import unittest
import pandas as pd
import numpy as np

class OriginUnitsTest(unittest.TestCase):
 def test_two_hour_release_boundary(self):
  path=Path(__file__).parent/'vendor/gong-ocb/gong_ocb.py'
  spec=importlib.util.spec_from_file_location('ocb_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  observation=pd.Timestamp('2026-10-14T06:00Z')
  rows=[{'observation_ns':observation.value,'available_ns':(observation+pd.Timedelta(hours=2)).value}]
  origins=pd.date_range(observation,periods=4,freq='h').as_unit('us')
  anchor=pd.DatetimeIndex([m.L0_ANCHOR_UTC]).as_unit('us')
  expected=360.0*(1.0-np.mod(m.L0_ANCHOR_CR,1.0))
  self.assertAlmostEqual(float(m.anchor_l0(anchor)[0]),expected)
  np.testing.assert_array_equal(m._origin_map_indices(origins,rows),[-1,-1,0,0])

if __name__=='__main__':unittest.main()
