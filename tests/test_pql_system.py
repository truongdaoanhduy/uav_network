"""Rate budgets, immutable snapshots, replay eviction, shutdown and resume."""
from copy import deepcopy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from uav_network.configuration import Config, load_config
from uav_network.environment.env import Action
from test_split_updates import replay_batch, small_learner, optimizers


def pql_config(algorithm="masac", mode="synchronous"):
    c = deepcopy(small_learner(algorithm).config)
    c.runtime.device = c.runtime.replay_device = "cpu"
    c.runtime.num_envs = 4
    c.algorithm_config.batch_size = 4
    c.algorithm_config.replay_capacity = 16
    c.pql.enabled, c.pql.mode = True, mode
    c.pql.collector_device = c.pql.learner_device = c.pql.policy_device = "cpu"
    c.pql.warmup_collection_steps = 2
    c.pql.critic_sample_ratio = 2.5
    c.pql.worker_timeout_s = 10
    c.validate()
    return c


def transition(batch):
    return (batch["observation"], batch["state"], Action(batch["continuous"], batch["destination"]),
            batch["reward"], batch["next_observation"], batch["next_state"], batch["terminated"])


class PQLSystemTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def system_class(self):
        self.assertIsNotNone(importlib.util.find_spec("uav_network.experiments.pql_system"),
                             "Missing PQL worker system")
        from uav_network.experiments.pql_system import PQLSystem
        return PQLSystem

    def test_config_has_validated_opt_in_pql_rates(self):
        c = Config()
        self.assertTrue(hasattr(c, "pql"), "Missing PQL configuration")
        self.assertFalse(c.pql.enabled)
        c.pql.critic_sample_ratio = 0
        with self.assertRaisesRegex(ValueError, "PQL"):
            c.validate()

    def test_fractional_rate_starts_after_warmup_and_preserves_policy_delay(self):
        cls = self.system_class()
        for algorithm in ("masac", "matd3"):
            for mode in ("synchronous", "async"):
                with self.subTest(algorithm=algorithm, mode=mode):
                    engine = cls(pql_config(algorithm, mode))
                    try:
                        batch = replay_batch(engine.critic)
                        engine.submit(transition(batch))
                        engine.flush()
                        self.assertEqual(engine.critic.updates, 0)
                        for _ in range(4):
                            engine.submit(transition(batch))
                        engine.flush()
                        metrics = engine.metrics()
                        self.assertEqual(engine.critic.updates, 10)
                        self.assertEqual(metrics["pql/eligible_collections"], 4)
                        self.assertEqual(metrics["pql/critic_collector_ratio"], 2.5)
                        self.assertEqual(metrics["pql/actor_updates"], 10 if algorithm == "masac" else 5)
                        self.assertEqual(metrics["pql/queued_collections"], 0)
                        self.assertGreater(metrics["replay/overwritten_rows"], 0)
                        self.assertFalse(hasattr(engine.collector, "actor_optimizers"))
                        if mode == "async":
                            self.assertFalse(hasattr(engine.critic, "actor_optimizers"))
                            self.assertFalse(hasattr(engine.policy, "critic_optimizer"))
                    finally:
                        engine.close()

    def test_collection_values_are_frozen_before_caller_reuses_storage(self):
        cls = self.system_class()
        engine = cls(pql_config())
        try:
            batch = replay_batch(engine.critic)
            expected = batch["observation"].clone()
            engine.submit(transition(batch))
            batch["observation"].zero_()
            torch.testing.assert_close(engine.replay.data["observation"][:4], expected)
            snapshot, version = engine.policy_mailbox.read()
            frozen = deepcopy(snapshot["actors"])
            with torch.no_grad():
                next(engine.critic.actors.parameters()).add_(1)
            for key in frozen:
                torch.testing.assert_close(frozen[key], snapshot["actors"][key], rtol=0, atol=0)
        finally:
            engine.close()

    def test_replay_wrap_tracks_real_unsampled_evictions_and_rng(self):
        self.system_class()
        from uav_network.experiments.pql_replay import MeasuredReplay
        learner = small_learner()
        batch = replay_batch(learner)
        replay = MeasuredReplay(4, batch["observation"].shape[1:], learner.state_dim)
        replay.add_batch(*transition(batch))
        replay.add_batch(*transition(batch))
        self.assertEqual(replay.stats()["replay/evicted_unsampled_fraction"], 1)
        self.assertEqual(replay.stats()["replay/overwritten_rows"], 4)
        replay.sample(4)
        saved = replay.state_dict()
        restored = MeasuredReplay(4, batch["observation"].shape[1:], learner.state_dim)
        restored.load_state_dict(saved)
        original_sample, restored_sample = replay.sample(4), restored.sample(4)
        for key, value in original_sample.items():
            torch.testing.assert_close(value, restored_sample[key], rtol=0, atol=0)
        a, b = replay.state_dict(), restored.state_dict()
        self.assertEqual(a["index"], b["index"])
        self.assertEqual(a["collections"], b["collections"])

    def test_async_nonfinite_failure_reaches_caller_without_hanging(self):
        cls = self.system_class()
        engine = cls(pql_config(mode="async"))
        try:
            batch = replay_batch(engine.critic)
            batch["reward"].fill_(float("nan"))
            engine.submit(transition(batch))
            engine.submit(transition(batch))
            with self.assertRaisesRegex(RuntimeError, "PQL.*worker"):
                engine.flush()
        finally:
            engine.close()

    def test_resume_rejects_unearned_credit_inconsistent_history_and_nonfinite_weights(self):
        cls = self.system_class()
        with TemporaryDirectory() as directory:
            c = pql_config()
            engine = cls(c)
            path = Path(directory) / "good.pt"
            try:
                batch = replay_batch(engine.critic)
                for _ in range(3):
                    engine.submit(transition(batch))
                engine.save(path, {"episodes": 1, "global_step": 12, "collection_steps": 3,
                    "next_seed": 48, "records": [{"episode": 1, "seed": 44, "global_step": 4}]})
                original = torch.load(path, weights_only=True)
            finally:
                engine.close()
            for name in ("seed", "history", "collections", "credit", "updates", "model"):
                with self.subTest(invalid=name):
                    bad = deepcopy(original)
                    if name == "seed":
                        bad["progress"]["next_seed"] = 44
                    elif name == "history":
                        bad["progress"]["episodes"] = 3
                    elif name == "collections":
                        bad["training"]["collections"] = 1000
                    elif name == "credit":
                        bad["training"]["credit"] = 900
                    elif name == "updates":
                        bad["updates"] = 0
                    else:
                        next(iter(bad["model"].values())).fill_(float("nan"))
                    corrupt = Path(directory) / (name + ".pt")
                    torch.save(bad, corrupt)
                    with self.assertRaisesRegex(ValueError, "PQL|checkpoint|progress|Checkpoint"):
                        restored = cls(c, checkpoint=corrupt)
                        restored.close()

    def test_legacy_independent_checkpoint_missing_sharing_flags_resumes(self):
        cls = self.system_class()
        with TemporaryDirectory() as directory:
            c = pql_config("masac")
            engine = cls(c)
            path = Path(directory) / "current.pt"
            try:
                engine.save(path, {"episodes": 0, "global_step": 0, "collection_steps": 0,
                                   "next_seed": 44, "records": []})
            finally:
                engine.close()
            saved = torch.load(path, weights_only=True)
            saved["config"]["masac"].pop("share_actor_parameters")
            saved["config"]["masac"].pop("share_critic_parameters")
            legacy = Path(directory) / "legacy.pt"
            torch.save(saved, legacy)
            restored = cls(c, checkpoint=legacy)
            try:
                self.assertFalse(restored.critic.share_actor_parameters)
                self.assertFalse(restored.critic.share_critic_parameters)
            finally:
                restored.close()

    def test_checkpoint_restores_adam_replay_rates_and_rng(self):
        cls = self.system_class()
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                c = pql_config(algorithm)
                engine = cls(c)
                restored = None
                try:
                    batch = replay_batch(engine.critic)
                    for _ in range(3):
                        engine.submit(transition(batch))
                    engine.flush()
                    path = Path(directory) / "state.pt"
                    engine.save(path, {"episodes": 0, "global_step": 12, "collection_steps": 3,
                                      "next_seed": 48, "records": []})
                    restored = cls(c, checkpoint=path)
                    self.assertEqual(restored.critic.updates, engine.critic.updates)
                    self.assertEqual(restored.critic.actor_updates, engine.critic.actor_updates)
                    self.assertTrue(restored.critic.critic_optimizer.state)
                    self.assertEqual(len(restored.replay), len(engine.replay))
                    self.assertEqual(restored.collections, engine.collections)
                    rng = torch.get_rng_state()
                    engine.submit(transition(batch))
                    torch.set_rng_state(rng)
                    restored.submit(transition(batch))
                    for key, value in engine.critic.state_dict().items():
                        torch.testing.assert_close(value, restored.critic.state_dict()[key], rtol=0, atol=0)
                    self.assertEqual(engine.critic.updates, restored.critic.updates)
                finally:
                    engine.close()
                    if restored:
                        restored.close()


if __name__ == "__main__":
    unittest.main()
