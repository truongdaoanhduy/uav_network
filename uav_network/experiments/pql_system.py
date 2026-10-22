"""PQL-inspired three-component training with measured, bounded rate feedback."""
from copy import deepcopy
import math
from queue import Empty, Full, Queue
import random
from threading import Condition, Event, Thread
import time

import numpy as np
import torch

from ..algorithms.checkpoints import load_policy, make_learner, save_policy
from ..algorithms.optimization import compile_models
from ..algorithms.precision import CompileStartupGate
from ..algorithms.replay import resolve_replay_device
from ..environment.observations import observation_dim, state_dim
from .pql_checkpoints import validate_pql_checkpoint
from .pql_replay import MeasuredReplay
from .pql_workers import (CriticWorker, PolicyWorker, SnapshotMailbox, cloned_state, freeze_transition,
                          inference_copy, install_policy, optimizer_states, policy_snapshot,
                          restore_optimizers, synchronize, worker_device)


def resolve_devices(config):
    available = torch.cuda.device_count()
    base = "cuda:0" if config.runtime.device != "cpu" and available else "cpu"
    if config.runtime.device == "cuda" and not available:
        raise RuntimeError("PQL CUDA requested without a CUDA device")
    def selected(value, fallback):
        value = fallback if value == "auto" else "cuda:0" if value == "cuda" else value
        device = torch.device(value)
        if device.type == "cuda" and (not available or device.index >= available):
            raise RuntimeError(f"PQL CUDA device unavailable: {device}")
        return device
    collector = selected(config.pql.collector_device, base)
    learning_default = "cuda:1" if collector.type == "cuda" and available >= 2 else str(collector)
    critic = selected(config.pql.learner_device, learning_default)
    policy = selected(config.pql.policy_device, str(critic))
    return collector, critic, policy


class PQLSystem:
    def __init__(self, config, checkpoint=None):
        self.config = deepcopy(config).validate()
        self.collector_device, self.critic_device, self.policy_device = resolve_devices(config)
        self.replay_device = resolve_replay_device(config.runtime.replay_device, self.critic_device)
        self.asynchronous = config.pql.mode == "async"
        self.condition, self.stop = Condition(), Event()
        self.error = None
        self.threads = []
        self.pending_collections = self.pending_policies = 0
        self.collections = self.processed_collections = self.eligible_collections = 0
        self.credit = 0.0
        self.warmup_origin = 0
        self.collector_wait_s = self.policy_wait_s = self.critic_s = self.actor_s = 0.0
        self.collector_s = 0.0
        self.latest = {}
        self.started = time.perf_counter()
        self.initial_progress = {"episodes": 0, "global_step": 0, "collection_steps": 0,
                                 "next_seed": config.runtime.seed, "records": []}
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True) if checkpoint else None
        self.initial_environment = None
        self.initial_collector_rng = None
        if saved:
            validate_pql_checkpoint(saved, config)
            self.initial_environment = saved["training"].get("environment")
            if self.initial_environment is not None:
                from .environment_checkpoints import validate_environment
                validate_environment(self.initial_environment, config, saved["progress"])
            self.initial_collector_rng = saved["training"]["rng"].get("collector")
            for group in ("world", "sensor", "network", "energy", "reward", config.runtime.algorithm):
                if saved["config"][group] != config.to_dict()[group]:
                    raise ValueError(f"PQL resume {group} differs from checkpoint")
            if (saved["config"]["runtime"]["seed"] != config.runtime.seed
                    or saved["config"]["runtime"]["environment_backend"] != config.runtime.environment_backend
                    or saved["config"]["pql"]["critic_sample_ratio"] != config.pql.critic_sample_ratio):
                raise ValueError("PQL resume seed, environment or rate differs from checkpoint")
            self.critic = load_policy(checkpoint, self.critic_device, config.runtime.algorithm, payload=saved)
            self.critic.config = deepcopy(config)
            self.initial_progress = saved["progress"]
            required = {"episodes", "global_step", "collection_steps", "next_seed", "records"}
            if not required <= self.initial_progress.keys():
                raise ValueError("PQL checkpoint needs verified progress metadata")
        else:
            self.critic = make_learner(config, observation_dim(config), state_dim(config), self.critic_device)
        if self.asynchronous:
            self.policy = make_learner(config, observation_dim(config), state_dim(config), self.policy_device)
            self.policy.load_state_dict(self.critic.state_dict())
            self.policy.updates, self.policy.actor_updates = self.critic.updates, self.critic.actor_updates
        else:
            if self.critic_device != self.policy_device:
                raise ValueError("Synchronous PQL debug requires both learners on the same device")
            self.policy = self.critic
        if saved:
            training = saved["training"]
            restore_optimizers(self.critic, self.policy, training["optimizers"])
            if training.get("precision") is not None:
                self.critic.precision.load_state_dict(training["precision"]["critic"])
                self.policy.precision.load_state_dict(training["precision"]["policy"])
            self.collections = self.processed_collections = training["collections"]
            self.eligible_collections = training["eligible_collections"]
            self.credit, self.warmup_origin = training["credit"], training["warmup_origin"]
        self.replay = MeasuredReplay(config.algorithm_config.replay_capacity,
                                    (config.world.num_uavs, observation_dim(config)), state_dim(config),
                                    config.runtime.seed, device=self.replay_device,
                                    pin_memory=self.replay_device.type == "cpu" and
                                    (self.critic_device.type == "cuda" or self.policy_device.type == "cuda"))
        if saved and saved["training"].get("replay") is not None:
            self.replay.load_state_dict(saved["training"]["replay"])
        elif saved:
            # Explicit small-checkpoint mode restores Adam/counters, then refills replay.
            self.warmup_origin = self.collections
        self.collector = inference_copy(self.policy, self.collector_device)
        self.collector_version = self.policy.actor_updates
        self.policy_mailbox = SnapshotMailbox(policy_snapshot(self.policy), self.policy.actor_updates)
        self.critic_worker = CriticWorker(self.critic, self.replay, self.policy_mailbox)
        self.policy_worker = PolicyWorker(self.policy, self.policy_mailbox)
        self.policy_worker.last_critic_version = self.policy.updates
        self.replay_metrics = self.replay.stats()
        self.collection_queue = Queue(maxsize=config.pql.max_pending_collections)
        self.policy_queue = Queue(maxsize=config.pql.max_pending_policy_updates)
        self.next_metric_update = self.critic.updates + 1
        self.metric_interval = max(1, math.ceil(config.runtime.log_interval_steps /
                                               config.runtime.num_envs * config.pql.critic_sample_ratio))
        if self.asynchronous:
            # Optimizers and live parameters have exactly one writer.
            del self.critic.actor_optimizers
            del self.policy.critic_optimizer
            if self.critic.algorithm == "masac":
                del self.critic.alpha_optimizer
                self.critic.log_alpha_continuous.requires_grad_(False)
                self.critic.log_alpha_discrete.requires_grad_(False)
            self.critic.actors.requires_grad_(False)
            self.policy.critics.requires_grad_(False)
        if config.runtime.compile_models:
            compile_models(self.critic)
            if self.policy is not self.critic:
                compile_models(self.policy)
            if self.asynchronous:
                gate = CompileStartupGate()
                self.critic._compile_startup_gate = self.policy._compile_startup_gate = gate
        self.initial_updates, self.initial_actors = self.critic.updates, self.policy.actor_updates
        self.initial_collections = self.collections
        if saved:
            rng = saved["training"]["rng"]
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng["cuda"]:
                if len(rng["cuda"]) != torch.cuda.device_count():
                    raise ValueError("PQL exact learner RNG resume requires the original CUDA device count")
                torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
            random.setstate(rng["python"])
            numpy_state = rng["numpy"]
            np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                                 *numpy_state[2:]))
        if self.asynchronous:
            for name, queue, operation, device in (
                    ("critic", self.collection_queue, self._collect_and_learn, self.critic_device),
                    ("policy", self.policy_queue, self._policy_learn, self.policy_device)):
                thread = Thread(target=self._run, args=(name, queue, operation, device),
                                name=f"pql-{name}", daemon=True)
                self.threads.append(thread)
                thread.start()

    def _raise_error(self):
        if self.error is not None:
            raise RuntimeError(f"PQL learner worker failed: {type(self.error).__name__}: {self.error}") from self.error
        if self.stop.is_set():
            raise RuntimeError("PQL system is closed")

    def _put(self, queue, item):
        started = time.perf_counter()
        deadline = started + self.config.pql.worker_timeout_s
        while True:
            self._raise_error()
            try:
                queue.put(item, timeout=.05)
                return time.perf_counter() - started
            except Full:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("PQL backpressure wait timed out")

    def _run(self, name, queue, operation, device):
        try:
            with worker_device(device):
                while not self.stop.is_set():
                    try:
                        item = queue.get(timeout=.05)
                    except Empty:
                        continue
                    try:
                        operation(item)
                    finally:
                        with self.condition:
                            if name == "critic":
                                self.pending_collections -= 1
                            else:
                                self.pending_policies -= 1
                            self.condition.notify_all()
                        queue.task_done()
        except BaseException as error:
            with self.condition:
                self.error = error
                self.stop.set()
                self.condition.notify_all()

    def submit(self, transition):
        self._raise_error()
        frozen = freeze_transition(transition)
        self.collections += 1
        if self.asynchronous:
            with self.condition:
                self.pending_collections += 1
            try:
                self.collector_wait_s += self._put(self.collection_queue, frozen)
            except BaseException:
                with self.condition:
                    self.pending_collections -= 1
                self.collections -= 1
                raise
        else:
            self._collect_and_learn(frozen)

    def _collect_and_learn(self, transition):
        self.replay.add_batch(*transition)
        self.processed_collections += 1
        eligible = (self.processed_collections - self.warmup_origin >= self.config.pql.warmup_collection_steps
                    and len(self.replay) >= self.config.algorithm_config.batch_size)
        if eligible:
            self.eligible_collections += 1
            self.credit += self.config.pql.critic_sample_ratio
            updates = math.floor(self.credit + 1e-10)
            self.credit = max(0., self.credit - updates)
            for _ in range(updates):
                begin = time.perf_counter()
                batch = self.replay.sample(self.config.algorithm_config.batch_size)
                checked = self.critic.updates + 1 >= self.next_metric_update
                if self.asynchronous:
                    try:
                        metrics = self.critic_worker.learn(batch, checked)
                    except BaseException:
                        self.failed_batch = {key: value.detach().cpu() for key, value in batch.items()}
                        raise
                    step = self.critic.updates
                    delay = self.config.matd3.policy_delay if self.critic.algorithm == "matd3" else 1
                    if step % delay == 0:
                        snapshot = cloned_state(self.critic.critics)
                        synchronize(self.critic.device)
                        with self.condition:
                            self.pending_policies += 1
                        try:
                            self.policy_wait_s += self._put(self.policy_queue, (step, batch, snapshot))
                        except BaseException:
                            with self.condition:
                                self.pending_policies -= 1
                            raise
                    self.critic_worker.update_targets()
                else:
                    actor_before = self.critic.actor_updates
                    metrics = self.critic.update(batch, checked)
                    if self.critic.actor_updates != actor_before:
                        self.policy_mailbox.publish(policy_snapshot(self.critic), self.critic.actor_updates)
                    self.critic_worker.last_policy_version = self.critic.actor_updates
                    self.policy_worker.last_critic_version = self.critic.updates
                synchronize(self.critic.device)
                with self.condition:
                    self.critic_s += time.perf_counter() - begin
                    if metrics:
                        self.latest.update(metrics)
                if checked:
                    self.next_metric_update = self.critic.updates + self.metric_interval
        synchronize(self.critic.device)
        values = self.replay.stats()
        with self.condition:
            self.replay_metrics = values

    def _policy_learn(self, task):
        begin = time.perf_counter()
        checked = task[0] % self.metric_interval == 0 or self.policy.actor_updates == 0
        metrics = self.policy_worker.learn(task, checked)
        synchronize(self.policy.device)
        with self.condition:
            self.actor_s += time.perf_counter() - begin
            if metrics:
                self.latest.update(metrics)

    def act_batch(self, observation, *, step=0, deterministic=False, return_tensors=False):
        self._raise_error()
        snapshot, version = self.policy_mailbox.read()
        if version != self.collector_version:
            self.collector.actors.load_state_dict(snapshot["actors"])
            synchronize(self.collector.device)
            self.collector_version = version
        return self.collector.act_batch(observation, deterministic, step, return_tensors=return_tensors)

    def record_collection_time(self, elapsed):
        self.collector_s += elapsed

    def flush(self):
        deadline = time.perf_counter() + self.config.pql.worker_timeout_s
        with self.condition:
            while self.pending_collections or self.pending_policies:
                self._raise_error()
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("PQL learner drain timed out")
                self.condition.wait(min(.05, remaining))
        self._raise_error()
        # At a drained boundary no worker mutates models; incorporate the newest policy.
        self.critic_worker.refresh_policy()
        self.critic.actor_updates = self.policy.actor_updates
        synchronize(self.critic.device)

    def metrics(self):
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        new_collections = self.processed_collections - self.initial_collections
        new_critics = self.critic.updates - self.initial_updates
        new_actors = self.policy.actor_updates - self.initial_actors
        with self.condition:
            return {**self.replay_metrics,
                    "pql/eligible_collections": self.eligible_collections,
                    "pql/critic_updates": self.critic.updates, "pql/actor_updates": self.policy.actor_updates,
                    "pql/collector_iterations_per_s": new_collections / elapsed,
                    "pql/transitions_per_s": new_collections * self.config.runtime.num_envs / elapsed,
                    "pql/critic_updates_per_s": new_critics / elapsed,
                    "pql/actor_updates_per_s": new_actors / elapsed,
                    "pql/critic_collector_ratio": self.critic.updates / max(1, self.eligible_collections),
                    "pql/critic_actor_ratio": self.critic.updates / max(1, self.policy.actor_updates),
                    "pql/sample_uses_per_new_transition": self.replay_metrics["replay/sample_uses"] /
                                                        max(1, self.replay_metrics["replay/inserted_rows"]),
                    "pql/queued_collections": self.pending_collections,
                    "pql/queued_policy_updates": self.pending_policies,
                    "pql/collector_wait_seconds": self.collector_wait_s,
                    "pql/critic_wait_for_policy_seconds": self.policy_wait_s,
                    "pql/critic_seconds": self.critic_s, "pql/policy_seconds": self.actor_s,
                    "pql/collector_seconds": self.collector_s,
                    "pql/collector_policy_lag": self.policy.actor_updates - self.collector_version,
                    "pql/critic_policy_lag": self.policy.actor_updates - self.critic_worker.last_policy_version,
                    "pql/policy_critic_lag": self.critic.updates - self.policy_worker.last_critic_version,
                    "pql/ratio_credit": self.credit,
                    "pql/scheduling_deterministic": int(not self.asynchronous)}

    def save(self, path, progress, collector_rng=None, environment=None):
        self.flush()
        numpy_state = np.random.get_state()
        training = {"optimizers": optimizer_states(self.critic, self.policy),
                    "precision": {"critic": self.critic.precision.state_dict(),
                                  "policy": self.policy.precision.state_dict()},
                    "collections": self.collections, "eligible_collections": self.eligible_collections,
                    "credit": self.credit, "warmup_origin": self.warmup_origin,
                    "collector_policy_version": self.collector_version,
                    "critic_policy_version": self.critic_worker.last_policy_version,
                    "policy_critic_version": self.policy_worker.last_critic_version,
                    "replay": self.replay.state_dict() if self.config.pql.checkpoint_replay else None,
                    "rng": {"torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                            "python": random.getstate(), "collector": collector_rng,
                            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])}}
        if environment is not None:
            training["environment"] = environment
        save_policy(self.critic, path, progress, training=training)

    def close(self):
        if self.stop.is_set() and not any(t.is_alive() for t in self.threads):
            return
        try:
            if self.error is None and not self.stop.is_set():
                self.flush()
        finally:
            self.stop.set()
            with self.condition:
                self.condition.notify_all()
            for thread in self.threads:
                thread.join(timeout=self.config.pql.worker_timeout_s)
            if any(thread.is_alive() for thread in self.threads):
                raise RuntimeError("PQL worker shutdown timed out")
