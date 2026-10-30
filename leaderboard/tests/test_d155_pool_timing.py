"""Unit and causality regression checks for D155 arrival diagnostics."""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/d153_shifts/pooling'))
sys.path.insert(0, str(ROOT/'experiments/d153_shifts/pooling_feature'))
import run_pool_safe as pool
import check_pending as check


class PoolTimingTests(unittest.TestCase):
    def test_missing_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            pool.hour(None)

    def test_one_component_crps_matches_gaussian_in_hours(self):
        y=np.array([1.,9.]); mu=np.array([[0.,5.,10.],[3.,7.,12.]])
        s=np.array([2.,3.,4.])
        np.testing.assert_allclose(pool.crps(y,mu,s,np.array([1.,0.,0.])),pool.pl.gauss_crps(y,mu[:,0],s[0]))

    def test_pending_expiration_and_release_boundaries(self):
        origin=1000.
        runs=np.array([[831.,994.],[832.,1003.],[1000.,995.],[1001.,994.],[999.,993.]])
        self.assertEqual(check.pending(runs,origin),-5.)
        self.assertEqual(check.pending(np.vstack([runs,[1000.,994.]]),origin),-6.)

    def test_future_revisions_cannot_change_features(self):
        rows=[dict(sub=0,mc=1,t=0,v=500,eta=40),dict(sub=30,mc=31,t=0,v=900,eta=20)]
        np.testing.assert_array_equal(pool.features(rows,25),pool.features(rows+[dict(sub=50,mc=50,t=99,v=2000,eta=0)],25))

    def test_constant_speed_transit_has_correct_distance_units(self):
        distance=149597870.7-21.5*695700.
        self.assertAlmostEqual(pool.dbm_hours(400),distance/400/3600)


if __name__=='__main__':
    unittest.main()
