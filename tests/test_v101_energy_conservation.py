"""V101 Energy Conservation & Accounting Audit (Zeng, Xu & Zhang, IEEE TWC 2019).

Verifies exact per-step energy accounting parity between CPU `flight_safety_energy.py`
and the GPU Warp formula in `mission_v80_shield.py` (steady-speed blade/induced/parasite
propulsion + 15 W avionics + linear PA DC airtime power), plus non-negativity clamp and
radio-shutoff contract when battery reaches 0 J.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]

from zhang_xue.mobility.flight_safety_energy import (
    ZengBatteryConfig,
    rf_pa_dc_power_w,
    uav_step_energy_j,
    zeng_propulsion_power_w,
)


def _gpu_shield_energy_formula_j(v_mps: float, air_s: float, dt_s: float = 1.0, tx_power_w: float = 0.1) -> float:
    """Exact mirror of mission_v80_shield.py Phase 3 battery decrement."""
    x = (v_mps * v_mps) / (2.0 * 4.03 * 4.03)
    p_blade = 79.86 * (1.0 + 3.0 * v_mps * v_mps / 14400.0)
    p_ind = 88.63 * math.sqrt(1.0 / (math.sqrt(1.0 + x * x) + x))
    p_par = 0.5 * 0.6 * 1.225 * 0.05 * 0.503 * (v_mps ** 3)
    pa_dc_w = rf_pa_dc_power_w(tx_power_w)
    return (p_blade + p_ind + p_par + 15.0) * dt_s + pa_dc_w * air_s


class TestV101EnergyConservation(unittest.TestCase):
    def test_cpu_gpu_zeng_propulsion_and_rf_energy_parity(self):
        cfg = ZengBatteryConfig()
        for v_mps in (0.0, 2.5, 5.0, 10.0, 15.0):
            for air_s in (0.0, 0.001, 0.015, 0.10):
                e_cpu = uav_step_energy_j(v_mps, air_s, dt_s=1.0, cfg=cfg, tx_power_w=0.1)
                e_gpu = _gpu_shield_energy_formula_j(v_mps, air_s, dt_s=1.0, tx_power_w=0.1)
                self.assertLess(abs(e_cpu - e_gpu), 1.0e-3, f"v={v_mps}, air={air_s}: cpu={e_cpu} gpu={e_gpu}")

    def test_hover_and_cruise_power_bounds_and_900s_budget_headroom(self):
        # At hover (v=0), P_prop = 79.86 + 88.63 = 168.49 W + 15 W avionics = 183.49 W
        p_hover = zeng_propulsion_power_w(0.0) + 15.0
        p_max = zeng_propulsion_power_w(15.0) + 15.0
        self.assertAlmostEqual(p_hover, 183.49, places=2)
        # Over 900s horizon with e_max_j=180,000 J, steady hover uses ~165,141 J < 180,000 J
        self.assertLess(p_hover * 900.0, 180_000.0)
        self.assertLess(p_max * 900.0, 180_000.0)

    def test_rf_tx_pa_share_is_sub_one_percent_justifying_fixed_tx_power(self):
        # Even at 10% duty cycle (90 s TX airtime over 900 s) at P_tx=0.1 W (PA_DC = 1.0 W),
        # RF PA consumes 90 J out of ~165,000 J (< 0.06% of flight energy).
        pa_w = rf_pa_dc_power_w(0.1)
        self.assertAlmostEqual(pa_w, 1.0, places=6)
        rf_900s = pa_w * 90.0
        flight_900s = (zeng_propulsion_power_w(10.0) + 15.0) * 900.0
        self.assertLess(rf_900s / flight_900s, 0.005)

    def test_gpu_shield_source_enforces_nonnegative_battery_and_radio_shutoff(self):
        src = (ROOT / "research/zhang_xue/src/zhang_xue/gpu/kernels/mission_v80_shield.py").read_text(encoding="utf-8")
        self.assertIn("battery[e, u] = wp.max(0.0, battery[e, u] - (p_blade + p_ind + p_par + 15.0) * dt_s - pa_dc_w * air_s)", src)
        self.assertIn("if battery[e, u] <= 0.0:\n                net.radio_alive[e, u] = wp.uint8(0)", src)


if __name__ == "__main__":
    unittest.main()
