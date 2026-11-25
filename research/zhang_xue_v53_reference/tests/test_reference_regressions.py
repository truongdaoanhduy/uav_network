"""Regressions run against original (red) or corrected diagnostic (green).

V52_REFERENCE_DIR=/absolute/original/reference python3 this_file.py
"""
import decimal
import inspect
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np

REFERENCE = Path(os.environ.get("V52_REFERENCE_DIR", str(Path(__file__).resolve().parents[1] / "reference")))
sys.path.insert(0, str(REFERENCE))
import v52_battery_7node_sim as sim


def decimal_ds(p, n):
    with decimal.localcontext() as ctx:
        ctx.prec = 1200
        a, b = decimal.Decimal("0.2") ** p, decimal.Decimal("0.2") ** n
        den = a + b - a*b
        return tuple(float(x / den) for x in (b*(1-a), a*(1-b), a*b))


class TestReferenceRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.energy_samples = []
        original = sim.step_energy_joules
        def observe_energy(v_mps, tx_airtime_s=0., dt_s=1., p=sim.ZengRotaryWingParams()):
            cls.energy_samples.append((float(v_mps), float(tx_airtime_s)))
            return original(v_mps, tx_airtime_s, dt_s, p)
        sim.step_energy_joules = observe_energy
        try:
            cls.episode = sim.run_7node_episode_comparison("full_comm",max_steps=300)
        finally:
            sim.step_energy_joules = original

    def test_ds_remains_normalized_for_horizon_scale_counts(self):
        # Restoring 1-conflict subtraction or the 1e-12 floor fails these cases.
        for p,n in [(17,17),(18,18),(24,24),(30,20),(100,100),(1500,1500),(9000,1000)]:
            with self.subTest(p=p,n=n):
                actual = sim.counts_to_canonical_ds(p,n)
                for value,expected in zip(actual,decimal_ds(p,n)):
                    self.assertAlmostEqual(value,expected,places=12)
                self.assertAlmostEqual(sum(actual),1.,places=12)

    def test_near_total_conflict_is_normalized_without_floor(self):
        eps = 1e-15
        got = sim.canonical_ds_combine((1-eps,0.,eps),(0.,1-eps,eps))
        self.assertAlmostEqual(sum(got),1.,places=12)
        self.assertAlmostEqual(got[0],.5,places=12)
        self.assertAlmostEqual(got[1],.5,places=12)

    def test_total_conflict_is_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError,"conflict"):
            sim.canonical_ds_combine((1.,0.,0.),(0.,1.,0.))

    def test_invalid_bba_and_counts_are_rejected(self):
        for m in [(-.1,0.,1.1),(.2,.2,.2),(float("nan"),0.,1.)]:
            with self.subTest(m=m), self.assertRaises(ValueError):
                sim.canonical_ds_combine(m,(0.,0.,1.))
        with self.assertRaises(ValueError):
            sim.counts_to_canonical_ds(-1,0)

    def test_actual_stationary_motion_uses_hover_energy(self):
        # The canonical route hits buildings. Constant 15m/s inputs hide hover cost.
        self.assertTrue(any(abs(speed)<1e-12 for speed,_ in self.energy_samples))

    def test_tx_energy_is_attributed_to_actual_uav_transmitters(self):
        # At least one step has unequal node TX demands; equal splitting fails.
        per_step = np.asarray(self.energy_samples).reshape(300,sim.ACTOR_N,2)
        self.assertTrue(np.any(np.ptp(per_step[:,:,1],axis=1)>1e-12))

    def test_result_labels_oracle_connectivity_diagnostic(self):
        self.assertEqual(self.episode.get("simulation_semantics"),"oracle_connectivity_diagnostic_no_packet_mac")

    def test_aoi_distinguishes_first_confirmation_from_receiver_age(self):
        self.assertIn("mean_first_tp_confirmation_aoi_s",self.episode)
        self.assertIn("mean_gcs_aoi_over_delivered_cells_s",self.episode)
        self.assertGreater(self.episode["mean_gcs_aoi_over_delivered_cells_s"],
                           self.episode["mean_first_tp_confirmation_aoi_s"])

    def test_fresh_timestamp_delivered_when_counts_saturate(self):
        # Inject one valid monotone saturated snapshot at the real routing boundary.
        source_lines, first_line = inspect.getsourcelines(sim.run_7node_episode_comparison)
        dirty_line = next(first_line+i for i,line in enumerate(source_lines)
                          if "dirty = np.flatnonzero(" in line)
        final = {}
        injected = False
        def trace(frame,event,arg):
            nonlocal injected
            if frame.f_code is not sim.run_7node_episode_comparison.__code__:
                return None
            state=frame.f_locals
            if event=="line" and frame.f_lineno==dirty_line and state["u"]==0 and not injected:
                self.assertEqual(state["nxt"],sim.GCS_ID)
                state["ledger"][0,0,399]=[65535,0]
                state["ledger"][sim.GCS_ID,0,399]=[65535,0]
                state["gen_ts"][0,0,399]=1
                state["gen_ts"][sim.GCS_ID,0,399]=0
                state["gcs_latest_u_s"][399]=0
                injected=True
            if event=="return":
                final["timestamp"]=int(state["gcs_latest_u_s"][399])
            return trace
        sys.settrace(trace)
        try:
            sim.run_7node_episode_comparison("full_comm",max_steps=1)
        finally:
            sys.settrace(None)
        self.assertTrue(injected)
        self.assertEqual(final["timestamp"],1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
