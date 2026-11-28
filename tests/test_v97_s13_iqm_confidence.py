"""S13 statistical contract: bootstrap confidence interval targets IQM, not mean."""
import pathlib
import sys
import unittest
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/gpu"))
from v96_s13_rf_ood_sweep import _iqm_stat, _iqm_and_ci95

class TestS13Confidence(unittest.TestCase):
    def test_fractional_iqm(self):
        self.assertAlmostEqual(_iqm_stat(np.array([0.,1.,2.,3.])), 1.5)
        self.assertAlmostEqual(_iqm_stat(np.array([0.,1.,2.])), 1.0)
    def test_iqm_bootstrap_clusters(self):
        x=np.array([0.]*16+[1.]*16+[100.]*16)
        r=_iqm_and_ci95(x,seed_clusters=3,seed=44)
        self.assertAlmostEqual(r["iqm"],17.333333333333332)
        self.assertEqual(r["bootstrap_statistic"],"IQM")
        self.assertEqual(r["bootstrap_unit"],"mission_seed_cluster")
        self.assertEqual(r["ci95"],_iqm_and_ci95(x,seed_clusters=3,seed=44)["ci95"])
        x2=np.tile(np.array([0.]*12+[1.]*4),3)
        r2=_iqm_and_ci95(x2,seed_clusters=3)
        self.assertNotEqual(r2["ci95"],r2["mean_ci95"])
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            _iqm_and_ci95(np.array([1.,float("nan"),2.]),seed_clusters=3)
