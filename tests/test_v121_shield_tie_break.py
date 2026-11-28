"""Fixed enumeration resolves binary32-scale cost ties without relaxing safety."""
from __future__ import annotations

import math
import unittest

import numpy as np

from zhang_xue.communication.common import scale_canonical_geometry
from zhang_xue.learners.actor_adapter import scale_normalized_xy_to_mps
from zhang_xue.mobility.flight_safety_energy import (
    kinematic_safety_shield, trajectory_safety_violations,
)

try:
    import warp as wp
except ImportError:
    wp = None


def _case():
    # Actual CUDA differential: seed44, world2, boundary motion, step5.
    pos = np.asarray([
        [79.88131713867188, 153.84759521484375],
        [250.94869995117188, 158.66018676757812],
        [287.3092346191406, 95.7697525024414],
        [493.6094055175781, 128.8994140625],
        [589.576416015625, 150.5186767578125],
        [661.8389282226562, 175.09925842285156],
    ])
    vel = np.asarray([
        [-3.372892141342163, 14.615867614746094],
        [13.64957332611084, 6.220058441162109],
        [-14.230249404907227, -4.743416786193848],
        [14.536457061767578, -3.7001888751983643],
        [6.525479793548584, 13.506224632263184],
        [-9.387006759643555, 11.699748039245605],
    ])
    controls = np.asarray([
        [-.22485949099063873, .9743912220001221],
        [.9099715948104858, .41467058658599854],
        [-.9486832618713379, -.3162277936935425],
        [.9690971970558167, -.24667926132678986],
        [.4350319802761078, .9004150032997131],
        [-.6258004307746887, .7799832224845886],
    ], dtype=np.float32)
    buildings, _, _ = scale_canonical_geometry(1000., 20)
    return pos, vel, controls, buildings


def _radius_candidate(vel, angle_index):
    # At radius .75*amax*dt=3.75, feasible angle9 precedes angle10.
    angle = angle_index * math.pi / 8.
    return vel[4] + 3.75 * np.asarray([math.cos(angle), math.sin(angle)])


def _cpu(pos, vel, command, buildings):
    safe, _ = kinematic_safety_shield(
        pos, vel, command, np.ones(6, bool), buildings,
        map_size_m=1000., dt_s=1., v_max_mps=15., a_max_mps2=5.,
        uav_safe_distance_m=50.,
    )
    violations = trajectory_safety_violations(
        pos, safe, np.ones(6, bool), buildings,
        map_size_m=1000., dt_s=1., uav_safe_distance_m=50.,
    )
    assert not any(violations.values()), violations
    return safe


if wp is not None:
    from zhang_xue.gpu.kernels.mission_v80_shield import _candidate_v80, _project_uav_v80

    @wp.kernel
    def _project_four(
        pos: wp.array3d(dtype=wp.float32), vel: wp.array3d(dtype=wp.float32),
        candidate: wp.array3d(dtype=wp.float32), safe: wp.array3d(dtype=wp.float32),
        battery: wp.array2d(dtype=wp.float32), buildings: wp.array2d(dtype=wp.float32),
        n_buildings: int, out: wp.array2d(dtype=wp.float32),
    ):
        e = wp.tid()
        result = _project_uav_v80(e, 4, 0, pos, vel, candidate, safe, battery,
                                   buildings, n_buildings, 1., 15., 5., 50.)
        out[e, 0] = result[0]
        out[e, 1] = result[1]


class TestV121ShieldTieBreak(unittest.TestCase):
    def test_observed_state_keeps_first_feasible_near_tie_under_one_ulp_controls(self):
        pos, vel, controls, buildings = _case()
        expected = _radius_candidate(vel, 9)  # enumeration index2+16+9=27
        for direction in (-np.inf, None, np.inf):
            with self.subTest(positive_control_direction=direction):
                action = controls.copy()
                if direction is not None:
                    action[4, 1] = np.nextafter(action[4, 1], np.float32(direction))
                command = scale_normalized_xy_to_mps(action, v_max_mps=15.)
                safe = _cpu(pos, vel, command, buildings)
                np.testing.assert_allclose(safe[4], expected, atol=1.e-8, rtol=0.)

    def test_exact_equal_radius_cost_keeps_first_feasible_enumeration(self):
        pos, vel, _, buildings = _case()
        safe = _cpu(pos, vel, vel.copy(), buildings)
        np.testing.assert_allclose(safe[4], _radius_candidate(vel, 9), atol=1.e-8, rtol=0.)

    def test_meaningful_cost_improvement_still_selects_later_candidate(self):
        pos, vel, controls, buildings = _case()
        controls[4, 1] -= .001  # Cost separation exceeds the binary32 tie band.
        safe = _cpu(pos, vel, scale_normalized_xy_to_mps(controls, v_max_mps=15.), buildings)
        np.testing.assert_allclose(safe[4], _radius_candidate(vel, 10), atol=1.e-8, rtol=0.)

    @unittest.skipIf(wp is None, "NVIDIA Warp unavailable")
    def test_direct_geometry_only_projection_preserves_feasible_nominal_on_cpu_and_cuda(self):
        pos, vel, controls, buildings = _case()
        positions = np.zeros((1, 7, 3), np.float32)
        positions[0, :6, :2] = pos
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            for direction in (-np.inf, None, np.inf):
                with self.subTest(device=device, control_direction=direction):
                    action = controls.copy()
                    if direction is not None:
                        action[4, 1] = np.nextafter(action[4, 1], np.float32(direction))
                    action_wp = wp.array(action[None], dtype=wp.float32, device=device)
                    vel_wp = wp.array(vel[None].astype(np.float32), dtype=wp.float32, device=device)
                    pos_wp = wp.array(positions, dtype=wp.float32, device=device)
                    candidate = wp.zeros((1, 6, 2), dtype=wp.float32, device=device)
                    safe = wp.zeros((1, 6, 2), dtype=wp.float32, device=device)
                    battery = wp.array(np.ones((1, 6), np.float32), dtype=wp.float32, device=device)
                    building_wp = wp.array(np.asarray(buildings, np.float32), dtype=wp.float32, device=device)
                    out = wp.zeros((1, 2), dtype=wp.float32, device=device)
                    wp.launch(_candidate_v80, dim=(1, 6), inputs=[
                        action_wp, vel_wp, candidate, 1., 15., 5.], device=device)
                    wp.copy(safe, candidate)
                    wp.launch(_project_four, dim=1, inputs=[
                        pos_wp, vel_wp, candidate, safe, battery, building_wp, len(buildings), out], device=device)
                    wp.synchronize_device(device)
                    # This wrapper calls _project_uav_v80(check_peers=0), unlike
                    # CPU kinematic_safety_shield's complete three-pass peer shield.
                    # In this prestate the nominal for UAV4 is geometry-feasible,
                    # so the direct independent pass MUST keep that nominal.
                    # The later peer-aware pass picks radius-angle candidate9;
                    # whole-shield parity is tested by v121_shield_differential.py.
                    np.testing.assert_allclose(out.numpy()[0], candidate.numpy()[0, 4], atol=2.e-5, rtol=0.)


if __name__ == "__main__":
    unittest.main()
