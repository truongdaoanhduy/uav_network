"""Pinned native versus device packet engine: custody, MAC timing and energy."""
import importlib
import importlib.util
import random
import unittest

import numpy as np

from uav_network.configuration import Config
from uav_network.environment.env import Action, SwarmSearchEnv
from uav_network.environment.sensing import visible_target_ids


def config(**network):
    c = Config()
    c.world.num_uavs = 2
    c.world.num_targets = 1
    c.world.num_obstacles = 0
    c.world.max_steps = 20
    c.sensor.altitude_m = 20
    c.network.peer_range_m = c.network.gcs_range_m = 160
    c.energy.enabled = False
    for key, value in network.items():
        setattr(c.network, key, value)
    return c


@unittest.skipUnless(importlib.util.find_spec("warp"), "optional warp_packet dependency not installed")
class GPUPacketTests(unittest.TestCase):
    device = "cpu"

    def backend(self):
        try:
            import warp as wp
            state = importlib.import_module("uav_network.environment.gpu.state")
            mac = importlib.import_module("uav_network.communication.gpu.mac")
        except ModuleNotFoundError:
            self.fail("GPU packet backend is missing")
        return wp, state, mac

    def prepare(self, c, positions, targets=None, obstacles=None, seed=44, battery=None):
        wp, state, mac = self.backend()
        s, p = state.allocate_state(c, seeds=[seed], device=self.device, trace=True)
        s.positions.assign(np.asarray([positions], dtype=np.float64))
        targets = targets if targets is not None else [positions[0]]
        obstacles = obstacles if obstacles is not None else []
        s.targets.assign(np.asarray([targets], dtype=np.float64))
        if obstacles:
            s.obstacles.assign(np.asarray([obstacles], dtype=np.float64))
        mask = wp.array([1], dtype=wp.int32, device=self.device)
        wp.launch(mac.reset_network, dim=1, inputs=[s, p, mask], device=self.device)
        if battery is not None:
            s.battery.assign(np.asarray([battery],dtype=np.float64))
        self.sense(c, s, p, mac, wp)
        return wp, s, p, mac

    def sense(self, c, s, p, mac, wp):
        positions, targets, obstacles = s.positions.numpy(), s.targets.numpy(), s.obstacles.numpy()
        visible = np.zeros(s.visible.shape, dtype=np.int32)
        for i in range(p.n):
            visible[0, i, visible_target_ids(positions[0, i], targets[0],
                                            obstacles[0, :p.obstacles_count], c.sensor)] = 1
        s.visible.assign(visible)
        wp.launch(mac.sense_reports, dim=1, inputs=[s, p], device=self.device)

    def step(self, c, s, p, mac, wp, destination, power=0):
        continuous = np.zeros((1, p.n, 3), dtype=np.float32)
        continuous[:, :, 2] = power
        wp.launch(mac.begin_step, dim=1, inputs=[s, p], device=self.device)
        self.sense(c, s, p, mac, wp)
        wp.launch(mac.packet_step, dim=1,
                  inputs=[s, p, wp.array(continuous, dtype=wp.float32, device=self.device),
                          wp.array([destination], dtype=wp.int64, device=self.device)], device=self.device)
        wp.launch(mac.end_step, dim=1, inputs=[s, p], device=self.device)
        self.assertEqual(s.errors.numpy().tolist(), [0])
        return mac.metrics_at(s, p, 0)

    def oracle(self, c, positions, actions, targets=None, obstacles=None, seed=44, battery=None):
        env = SwarmSearchEnv(c)
        try:
            env.reset(seed=seed, options={"positions": positions,
                      "targets": targets if targets is not None else [positions[0]],
                      "obstacles": obstacles if obstacles is not None else []})
            if battery is not None:
                env.battery_j[:] = battery
            for dest, power in actions:
                continuous = np.zeros((c.world.num_uavs, 3))
                continuous[:, 2] = power
                env.step(Action(continuous, np.asarray(dest)))
            events = env.network.simulator.event_bus.since(0)
            trace = [(e["event_type"], e["sim_time_us"], e["data"].get("source"),
                      e["data"].get("destination")) for e in events
                     if e["event_type"] in {"packet_tx_started", "packet_rx_succeeded", "packet_rx_failed"}]
            return (env.network.metrics(), env.reports.known.copy(), env.reports.delivered.copy(),
                    [[(t, r.source_id, r.created_s, r.hops) for t, r in b.items()]
                     for b in env.reports.buffers], trace)
        finally:
            env.close()

    def compare(self, c, positions, actions, targets=None, obstacles=None, seed=44, battery=None):
        native, known, delivered, buffers, trace = self.oracle(c, positions, actions, targets, obstacles, seed, battery)
        wp, s, p, mac = self.prepare(c, positions, targets, obstacles, seed, battery)
        times = []
        for dest, power in actions:
            actual = self.step(c, s, p, mac, wp, dest, power)
            times.extend(mac.transmission_times(s, 0))
        for key in ("generated_packets", "received_packets", "phy_successes", "phy_failures",
                    "collisions", "energy_blocked_transmissions"):
            self.assertEqual(actual[key], native[key], key)
        for key in ("communication_energy_j", "payload_throughput_kbps", "packet_e2e_delay_ms",
                    "report_e2e_delay_s", "report_average_hops"):
            self.assertAlmostEqual(actual[key], native[key], places=10, msg=key)
        self.assertEqual(actual["selection_counts"], native["selection_counts"])
        np.testing.assert_array_equal(s.known.numpy()[0], known)
        np.testing.assert_array_equal(s.delivered.numpy()[0], delivered)
        expected = np.zeros((p.n, p.targets_count), dtype=np.int32)
        for i, reports in enumerate(buffers):
            for target, origin, created, hops in reports:
                expected[i, target] = 1
                self.assertEqual(s.report_origin.numpy()[0, i, target], origin)
                self.assertEqual(s.report_created.numpy()[0, i, target], created)
                self.assertEqual(s.report_hops.numpy()[0, i, target], hops)
        np.testing.assert_array_equal(s.report_present.numpy()[0], expected)
        np.testing.assert_allclose(times, [t[1] for t in trace if t[0] == "packet_tx_started"],
                                   rtol=0, atol=1e-9)
        return wp, s, p, mac

    def test_state_schema_and_masked_reset(self):
        wp, state, mac = self.backend()
        c = config()
        s, p = state.allocate_state(c, 2, seeds=[44, 45], device=self.device)
        self.assertEqual(s.positions.dtype, wp.vec2d)
        self.assertEqual(s.positions.shape, (2, 2))
        self.assertEqual(s.battery.dtype, wp.float64)
        self.assertEqual(s.observation.dtype, wp.float32)
        s.steps.assign(np.array([7, 8], dtype=np.int32))
        wp.launch(mac.reset_network, dim=2,
                  inputs=[s, p, wp.array([1, 0], dtype=wp.int32, device=self.device)], device=self.device)
        self.assertEqual(s.steps.numpy().tolist(), [0, 8])

    def test_python_mt19937_randint_matches_rejection_and_twists(self):
        wp, state, _ = self.backend()
        rng = importlib.import_module("uav_network.communication.gpu.rng")
        s, p = state.allocate_state(config(), seeds=[44], device=self.device)
        bounds = np.array([31, 63, 127, 255, 511] * 300, dtype=np.int32)
        result = rng.draw_backoffs(s, p, bounds, node=0)
        oracle = random.Random(44 + 5)
        np.testing.assert_array_equal(result, [oracle.randint(0, int(n) - 1) for n in bounds])

    def test_direct_delivery_exact_data_and_ack_times(self):
        self.compare(config(), [[500, 100], [600, 100]], [([2, 0], 0)])

    def test_relay_preserves_created_time_and_hops(self):
        self.compare(config(), [[500, 300], [500, 150]], [([1, 0], 0), ([0, 2], 0)])

    def test_all_fragments_required_and_fifo_poll_phase(self):
        self.compare(config(report_bytes=2048, payload_bytes=512), [[500, 100], [600, 100]], [([2, 0], 0)])

    def test_weak_link_retries_and_action_power(self):
        for power in (-1, 1):
            with self.subTest(power=power):
                self.compare(config(sinr_threshold_db=200), [[500, 100], [600, 100]], [([2, 0], power)])

    def test_source_ttl_after_relay(self):
        self.compare(config(report_ttl_s=1.5), [[500, 300], [500, 150]], [([1, 0], 0), ([0, 0], 0)])

    def test_busy_channel_freeze_and_two_senders(self):
        c = config()
        c.world.num_targets = 2
        self.compare(c, [[500, 100], [550, 100]], [([2, 2], 0)],
                     targets=[[500, 100], [550, 100]])

    def test_polygon_nlos_phy_and_retries(self):
        c = config()
        c.world.num_obstacles = 1
        self.compare(c, [[500, 100], [600, 100]], [([2, 0], 0)], obstacles=[[500, 50, 15]])

    def test_queue_cap_fragment_reinjection(self):
        self.compare(config(report_bytes=2048, payload_bytes=512, max_queue_packets=1),
                     [[500, 100], [600, 100]], [([2, 0], 0)] * 4)

    def test_exhausted_energy_attempt_never_delivers(self):
        c = config()
        c.energy.enabled = True
        c.energy.idle_power_w = c.energy.hover_power_w = 0
        c.energy.battery_capacity_j = 1e-6
        self.compare(c, [[500, 100], [600, 100]], [([2, 0], 0)])

    def test_equal_backoff_tie_preserves_collision_and_retry_order(self):
        seed = next(seed for seed in range(500) if
                    random.Random(seed+5).randint(0,30) == random.Random(seed+6).randint(0,30))
        c = config()
        c.world.num_targets = 2
        self.compare(c, [[500,100],[550,100]], [([2,2],0)],
                     targets=[[500,100],[550,100]], seed=seed)

    def test_hidden_sender_airtime_overlap_uses_native_interference(self):
        c = config(cca_threshold_dbm=-30)
        c.world.num_targets = 2
        self.compare(c, [[500,100],[550,100]], [([2,2],0)],
                     targets=[[500,100],[550,100]])

    def test_receiver_fills_after_admission_retains_source_custody(self):
        c = config(buffer_bytes=1024)
        c.world.num_uavs = c.world.num_targets = 3
        self.compare(c, [[500,300],[600,300],[550,150]],
                     [([2,2,0],0),([0,0,3],0),([2,2,0],0),([0,0,3],0)],
                     targets=[[500,300],[600,300],[900,900]])

    def test_oldest_known_report_does_not_starve_forwardable_fifo(self):
        c = config(report_ttl_s=3)
        c.world.num_targets = 2
        c.sensor.altitude_m = 100
        self.compare(c, [[500,300],[500,150]], [([1,2],0),([1,2],0)],
                     targets=[[500,225],[500,390]])

    def test_packet_deadline_before_first_poll_and_rediscovery(self):
        self.compare(config(report_ttl_s=0.0005), [[500,100],[600,100]],
                     [([2,0],0),([2,0],0)])

    def test_active_data_ack_and_wait_processes_cross_step_boundary(self):
        c = config()
        c.world.dt_s = 0.003
        self.compare(c, [[500,100],[600,100]], [([2,0],0),([2,0],0)])

    def test_fractional_queue_poll_keeps_float64_listener_phase(self):
        c = config(queue_poll_us=1000.3)
        c.world.num_targets = 2
        self.compare(c, [[500,100],[550,100]], [([2,2],0)],
                     targets=[[500,100],[550,100]])

    def test_six_uavs_seeded_contention_corpus(self):
        c = config()
        c.world.num_uavs = 6
        c.world.num_targets = 7
        c.network.peer_range_m = c.network.gcs_range_m = 500
        positions = [[350+i*60,100] for i in range(6)]
        for seed in (0,1,44,99,2**40+44):
            with self.subTest(seed=seed):
                self.compare(c, positions, [([6]*6,0),([6]*6,0)],
                             targets=positions+[[950,950]], seed=seed)

    def test_ack_weak_link_duplicates_keep_single_report_delivery(self):
        self.compare(config(ack_uses_data_power=False, sinr_threshold_db=35),
                     [[500,100],[600,100]], [([2,0],1)])

    def test_idle_poll_jump_preserves_next_step_native_poll_phase(self):
        self.compare(config(), [[500,100],[600,100]],
                     [([0,0],0),([0,0],0),([2,0],0)])

    def test_wholly_idle_interval_coalesces_poll_events(self):
        c = config()
        wp, s, p, mac = self.prepare(c, [[500,100],[600,100]])
        self.step(c,s,p,mac,wp,[0,0])
        self.assertLess(int(s.event_sequence.numpy()[0]),100)

    def test_receiver_ack_budget_denial_retains_transferred_custody(self):
        c = config()
        c.energy.enabled = True
        c.energy.idle_power_w = c.energy.hover_power_w = 0
        self.compare(c, [[500,300],[500,150]], [([1,0],0)],
                     battery=[0.02,0.00002])

    def test_exact_data_budget_depletion_still_delivers_emitted_frame(self):
        c = config()
        c.energy.enabled = True
        c.energy.idle_power_w = c.energy.hover_power_w = 0
        cost = (c.network.report_bytes*8+464)/c.network.bit_rate_bps*0.25
        self.compare(c, [[500,100],[600,100]], [([2,0],0)],
                     battery=[cost,c.energy.battery_capacity_j])

    def test_mt_seed_int64_boundary_matches_cpython_unsigned_seed(self):
        wp, state, _ = self.backend()
        rng = importlib.import_module("uav_network.communication.gpu.rng")
        s,p = state.allocate_state(config(),seeds=[2**63-1],device=self.device)
        oracle = random.Random(2**63-1+5)
        np.testing.assert_array_equal(rng.draw_backoffs(s,p,[31]*700),
                                      [oracle.randint(0,30) for _ in range(700)])

    def test_unbounded_retry_setting_rejected_explicitly(self):
        _,state,_ = self.backend()
        with self.assertRaisesRegex(NotImplementedError,"retries"):
            state.params_from_config(config(max_retries=27))

    def test_short_frames_preserve_receiver_fifo_after_phy_retention_prune(self):
        c = config(report_bytes=1,payload_bytes=1,queue_poll_us=1000,
                   sinr_threshold_db=-200,cca_threshold_dbm=100)
        c.world.num_uavs = c.world.num_targets = 6
        c.world.dt_s = 0.01
        c.network.peer_range_m = c.network.gcs_range_m = 500
        positions = [[350+i*60,100] for i in range(6)]
        for seed in (0,44,99):
            with self.subTest(seed=seed):
                self.compare(c,positions,[([6]*6,0)],targets=positions,seed=seed)

    def test_nonbinary_dt_uses_native_previous_clock_after_idle_steps(self):
        c = config()
        c.world.dt_s = 0.7
        wp,s,_,_ = self.compare(c,[[500,100],[600,100]], [([0,0],0)]*17+[([2,0],0)])
        self.assertEqual(float(s.packet_time.numpy()[0]), (17*c.world.dt_s+c.world.dt_s)*1e6)

    def test_event_overflow_is_explicit(self):
        wp, s, p, mac = self.prepare(config(), [[500, 100], [600, 100]])
        p.event_capacity = 1
        wp.launch(mac.reset_network, dim=1,
                  inputs=[s, p, wp.array([1], dtype=wp.int32, device=self.device)], device=self.device)
        self.assertNotEqual(s.errors.numpy()[0], 0)
