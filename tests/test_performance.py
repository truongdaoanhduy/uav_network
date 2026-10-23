"""Real simulation event counts and independent-actor update equivalence."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from uav_network.algorithms import optimization
from uav_network.algorithms.checkpoints import make_learner
from uav_network.configuration import Config
from uav_network.environment.env import Action, SwarmSearchEnv
from uav_network.environment.observations import observation_dim, state_dim


def check_learner_equivalence(device="cpu"):
    """Compare real complete updates to separate backward/Adam per actor."""
    c = Config()
    c.masac.hidden_sizes = c.matd3.hidden_sizes = [32, 32]
    torch.manual_seed(44)
    obs = torch.randn(8, 6, observation_dim(c), device=device)
    obs[:, :, 9] = 1
    obs[:, :, -7:] = 1
    data = {"observation": obs, "next_observation": obs.clone(),
            "state": torch.randn(8, state_dim(c), device=device),
            "next_state": torch.randn(8, state_dim(c), device=device),
            "continuous": torch.zeros(8, 6, 3, device=device),
            "destination": torch.zeros(8, 6, dtype=torch.long, device=device),
            "reward": torch.ones(8, device=device), "terminated": torch.zeros(8, device=device)}

    def sequential(losses, optimizers, groups, clip, check):
        return [optimization.step_optimizer(loss, optimizer, group, clip, check)
                for loss, optimizer, group in zip(losses, optimizers, groups)]

    for algorithm in ("masac", "matd3"):
        c.runtime.algorithm = algorithm
        reference = make_learner(c, observation_dim(c), state_dim(c), device)
        actual = make_learner(c, observation_dim(c), state_dim(c), device)
        actual.load_state_dict(reference.state_dict())
        for step in range(3):
            torch.manual_seed(44 + step)
            with patch(f"uav_network.algorithms.{algorithm}.step_actor_optimizers", sequential):
                reference.update(data, collect_metrics=False)
            torch.manual_seed(44 + step)
            actual.update(data, collect_metrics=False)
            for name, expected in reference.state_dict().items():
                torch.testing.assert_close(actual.state_dict()[name], expected, rtol=0, atol=0)
        if actual.updates != reference.updates:
            raise AssertionError("Learner update counters differ")


class PerformanceTests(unittest.TestCase):
    def test_packet_selection_caches_receiver_buffer_usage(self):
        import inspect
        from uav_network.communication.gpu import mac

        source = inspect.getsource(mac.packet_step)
        selection = source.split("selected = int(-1)", 1)[1].split("if selected >= 0", 1)[0]
        before_targets, target_loop = selection.split("for t in range(p.targets_count):", 1)
        self.assertIn("bytes_used(s,p,e,dest)", before_targets)
        self.assertNotIn("can_receive(s,p,e,dest,t)", target_loop)

    def test_unrequested_fusion_keeps_pytorch_adam_defaults(self):
        c = Config()
        c.masac.hidden_sizes = c.matd3.hidden_sizes = [16, 16]
        for algorithm in ("masac", "matd3"):
            c.runtime.algorithm = algorithm
            for requested in (False, True):
                c.runtime.fused_optimizers = requested
                learner = make_learner(c, observation_dim(c), state_dim(c), "cpu")
                expected = torch.optim.Adam(learner.actors[0].parameters(),
                                            lr=c.algorithm_config.learning_rate).defaults
                for optimizer in [*learner.actor_optimizers, learner.critic_optimizer]:
                    self.assertEqual(optimizer.defaults, expected)

    def test_both_complete_learners_match_separate_actor_updates(self):
        torch.set_num_threads(1)
        check_learner_equivalence()

    def test_idle_native_step_does_not_process_thousands_of_empty_queue_polls(self):
        env = SwarmSearchEnv(Config())
        try:
            env.reset(seed=44)
            clock = env.network.clock
            original_step = clock.step
            events = []

            def counted_step():
                events.append(clock.now)
                return original_step()

            clock.step = counted_step
            env.step(Action(np.zeros((6, 3)), np.zeros(6, dtype=np.int64)))
            self.assertEqual(clock.now, 1_000_000)
            self.assertEqual(env.network.metrics()["generated_packets"], 0)
            self.assertLess(len(events), 256)
        finally:
            env.close()

    def test_coalescing_preserves_packet_events_after_idle_retries_and_relay(self):
        for dt, poll in ((1., 1000.), (.013, 700.), (.013, 700.25)):
            traces = []
            for coalesce in (False, True):
                traces.append(self.packet_trace(coalesce, dt, poll))
            self.assertEqual(traces[0], traces[1])

    def test_active_radio_keeps_original_polling_for_other_idle_nodes(self):
        c = Config()
        c.world.num_targets, c.world.num_obstacles = 1, 0
        c.network.bit_rate_bps = 2000  # A real data frame lasts beyond this step.
        env = SwarmSearchEnv(c)
        try:
            env.reset(seed=44, options={"positions": [[500, 80], [550, 80], [450, 80],
                                                     [500, 130], [550, 130], [450, 130]],
                                       "targets": [[500, 80]], "obstacles": []})
            clock, events = env.network.clock, []
            original_step = clock.step

            def counted_step():
                events.append(clock.now)
                return original_step()

            clock.step = counted_step
            env.step(Action(np.zeros((6, 3)), np.array([6, 0, 0, 0, 0, 0])))
            self.assertGreater(len(events), 5000)
            self.assertEqual(env.network.metrics()["generated_packets"], 1)
            self.assertEqual(env.network.metrics()["received_packets"], 0)
            events.clear()  # The sole frame left the queue, but remains on air.
            env.step(Action(np.zeros((6, 3)), np.zeros(6, dtype=np.int64)))
            self.assertGreater(len(events), 5000)
            self.assertEqual(env.network.metrics()["received_packets"], 0)
        finally:
            env.close()

    @staticmethod
    def packet_trace(coalesce, dt, poll):
        c = Config()
        c.runtime.coalesce_idle_polls = coalesce
        c.world.dt_s, c.network.queue_poll_us = dt, poll
        c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 3, 3, 1
        c.sensor.altitude_m = 20
        c.network.report_bytes, c.network.payload_bytes = 2048, 512
        c.network.buffer_bytes, c.network.report_ttl_s = 4096, 3.5
        env = SwarmSearchEnv(c)
        trace = []
        try:
            env.reset(seed=44, options={"positions": [[450, 200], [550, 200], [500, 100]],
                                       "targets": [[450, 200], [500, 100], [900, 900]],
                                       "obstacles": [[500, 200, 10]]})
            schedule = [[0, 0, 0], [0, 0, 0], [1, 0, 3], [0, 3, 0],
                        [0, 0, 0], [1, 0, 0], [0, 3, 0], [0, 0, 0]]
            for step, destinations in enumerate(schedule):
                continuous = np.zeros((3, 3))
                continuous[:, 2] = -1 if step == 2 else 1
                obs, reward, terminated, truncated, info = env.step(
                    Action(continuous, np.array(destinations)))
                trace.append({"observation": obs.tolist(), "state": env.state().tolist(),
                              "reward": reward, "terminated": terminated, "truncated": truncated,
                              "info": info, "events": env.network.simulator.event_bus.since(0)})
            return json.dumps(trace, sort_keys=True)
        finally:
            env.close()

    def test_grouped_actor_steps_keep_separate_clipping_and_adam_states(self):
        step_many = getattr(optimization, "step_actor_optimizers", None)
        self.assertIsNotNone(step_many, "Missing grouped independent-actor updates")
        initial = [torch.nn.Parameter(torch.tensor([2., -3.])),
                   torch.nn.Parameter(torch.tensor([.01, -.02]))]
        reference = deepcopy(initial)
        actual = deepcopy(initial)
        legacy_optimizers = [torch.optim.Adam([p], lr=.01) for p in reference]
        optimizers = [torch.optim.Adam([p], lr=.01) for p in actual]
        for _ in range(3):
            norms = [optimization.step_optimizer(p.square().sum(), opt, [p], .5)
                     for p, opt in zip(reference, legacy_optimizers)]
            grouped = step_many([p.square().sum() for p in actual], optimizers,
                                [[p] for p in actual], .5)
            torch.testing.assert_close(torch.stack(norms), torch.stack(grouped), rtol=0, atol=0)
            for expected, observed, old, new in zip(reference, actual, legacy_optimizers, optimizers):
                torch.testing.assert_close(expected, observed, rtol=0, atol=0)
                for key in old.state[expected]:
                    torch.testing.assert_close(old.state[expected][key], new.state[observed][key], rtol=0, atol=0)

    def test_bad_actor_gradient_stops_every_actor_adam_before_mutation(self):
        step_many = getattr(optimization, "step_actor_optimizers", None)
        self.assertIsNotNone(step_many, "Missing grouped independent-actor updates")
        good = torch.nn.Parameter(torch.tensor(2.))
        bad = torch.nn.Parameter(torch.tensor(0.))
        optimizers = [torch.optim.Adam([p], lr=.01) for p in (good, bad)]
        with self.assertRaises(RuntimeError):
            step_many([good.square(), bad.sqrt()], optimizers, [[good], [bad]], 10,
                      check_finite=False)
        self.assertEqual(float(good), 2.)
        self.assertEqual(float(bad), 0.)
        self.assertTrue(all(not opt.state for opt in optimizers))


if __name__ == "__main__":
    unittest.main()
