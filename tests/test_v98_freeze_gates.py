"""V98 CPU unit tests verifying S12 continuous-time DES oracle, hidden terminal, and <=500 line policy."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src"), str(ROOT / "scripts/gpu")]

from v94_mac_analytic_validation import solve_finite_retry_bianchi
from v98_simpy_vs_warp_mac_parity import run_cleanroom_continuous_des


class TestV98FreezeGates(unittest.TestCase):
    def test_s12_cleanroom_continuous_des_tracks_bianchi_and_cw_shift(self):
        for n in (2, 4, 6):
            an1 = solve_finite_retry_bianchi(n, inclusive=True, total_attempts=5)
            des1 = run_cleanroom_continuous_des(n, cw_inclusive_upper=1, duration_us=1_500_000, seed=44 + n)
            des0 = run_cleanroom_continuous_des(n, cw_inclusive_upper=0, duration_us=1_500_000, seed=144 + n)
            self.assertGreater(des1["attempts"], 1000)
            self.assertGreater(des0["attempts"], 1000)
            # Continuous-time non-EIFS DES tracks idealized Bianchi within bounded tolerance
            self.assertLess(abs(des1["throughput_des"] - an1["throughput_airtime_frac"]) / an1["throughput_airtime_frac"], 0.16)
        # Hidden terminal increases collision probability dramatically at N=2
        des_clear = run_cleanroom_continuous_des(2, cw_inclusive_upper=1, duration_us=1_000_000, seed=46, hidden=False)
        des_ht = run_cleanroom_continuous_des(2, cw_inclusive_upper=1, duration_us=1_000_000, seed=99, hidden=True)
        self.assertGreater(des_ht["p_des"], des_clear["p_des"] * 2.0)

    def test_v98_new_scripts_comply_with_500_line_policy(self):
        for rel in [
            "scripts/gpu/v98_simpy_vs_warp_mac_parity.py",
            "scripts/gpu/v98_warp_4096x900_endurance_gate.py",
            "scripts/gpu/v98_scenario_ablation_suite.py",
            "tests/test_v98_freeze_gates.py",
        ]:
            nlines = len((ROOT / rel).read_text(encoding="utf-8").splitlines())
            self.assertLessEqual(nlines, 500, f"{rel} has {nlines} lines > 500")


if __name__ == "__main__":
    unittest.main()
