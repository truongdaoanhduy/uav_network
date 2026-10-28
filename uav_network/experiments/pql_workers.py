"""Owned learner stages and immutable snapshots; no optimizer or replay sharing."""
from contextlib import contextmanager
from copy import deepcopy
from threading import Lock

import torch
from torch import nn

from ..environment.env import Action


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.current_stream(device).synchronize()


@contextmanager
def worker_device(device):
    if torch.device(device).type == "cuda":
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.default_stream(device))
            with torch.cuda.stream(stream):
                try:
                    yield
                finally:
                    stream.synchronize()
    else:
        yield


def cloned_state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def policy_snapshot(learner):
    result = {"actors": cloned_state(learner.actors)}
    if learner.algorithm == "matd3":
        result["target_actors"] = cloned_state(learner.target_actors)
    else:
        result["alpha_c"] = learner.log_alpha_continuous.detach().clone()
        result["alpha_d"] = learner.log_alpha_discrete.detach().clone()
    synchronize(learner.device)
    return result


def install_policy(learner, snapshot):
    learner.actors.load_state_dict(snapshot["actors"])
    if learner.algorithm == "matd3":
        learner.target_actors.load_state_dict(snapshot["target_actors"])
    else:
        with torch.no_grad():
            learner.log_alpha_continuous.copy_(snapshot["alpha_c"])
            learner.log_alpha_discrete.copy_(snapshot["alpha_d"])
    # Keep source snapshot storage alive until destination copies finish.
    synchronize(learner.device)


class SnapshotMailbox:
    """One immutable latest value; readers copy it into their own models."""
    def __init__(self, snapshot, version=0):
        self.lock = Lock()
        self.value, self.version = snapshot, version

    def publish(self, snapshot, version):
        with self.lock:
            if version <= self.version:
                raise ValueError("PQL snapshot versions must increase")
            self.value, self.version = snapshot, version

    def read(self):
        with self.lock:
            return self.value, self.version


def inference_copy(learner, device):
    # Allocate only the fields used by the existing act_batch implementation.
    result = type(learner).__new__(type(learner))
    nn.Module.__init__(result)
    result.config = deepcopy(learner.config)
    result.device = torch.device(device)
    result.num_uavs, result.destinations = learner.num_uavs, learner.destinations
    result.share_actor_parameters = learner.share_actor_parameters
    result.share_critic_parameters = learner.share_critic_parameters
    result.actors = deepcopy(learner.actors).to(device).requires_grad_(False)
    return result


def freeze_transition(values):
    observation, state, action, reward, next_obs, next_state, terminated = values
    def freeze(value):
        return value.detach().clone() if isinstance(value, torch.Tensor) else torch.as_tensor(value).clone()
    frozen = (freeze(observation), freeze(state), Action(freeze(action.continuous), freeze(action.destination)),
              freeze(reward), freeze(next_obs), freeze(next_state), freeze(terminated))
    for value in frozen:
        if isinstance(value, torch.Tensor):
            synchronize(value.device)
    return frozen


def optimizer_states(critic, policy):
    result = {"critic": critic.critic_optimizer.state_dict(),
              "actors": [optimizer.state_dict() for optimizer in policy.actor_optimizers]}
    if policy.algorithm == "masac":
        result["alpha"] = policy.alpha_optimizer.state_dict()
    return result


def restore_optimizers(critic, policy, states):
    # Adam's CPU step tensors otherwise retain the entire mmap input checkpoint.
    states = deepcopy(states)
    critic.critic_optimizer.load_state_dict(states["critic"])
    if len(states["actors"]) != len(policy.actor_optimizers):
        raise ValueError("PQL checkpoint actor optimizer count mismatch")
    for optimizer, state in zip(policy.actor_optimizers, states["actors"]):
        optimizer.load_state_dict(state)
    if policy.algorithm == "masac":
        policy.alpha_optimizer.load_state_dict(states["alpha"])


class PolicyWorker:
    def __init__(self, learner, mailbox):
        self.learner, self.mailbox = learner, mailbox
        self.last_critic_version = 0

    def learn(self, task, collect_metrics):
        step, batch, critics = task
        self.learner.critics.load_state_dict(critics)
        self.learner.updates = self.last_critic_version = step
        metrics = self.learner.update_actor(batch, collect_metrics)
        if self.learner.algorithm == "matd3":
            self.learner.update_targets(actor=True, critic=False)
        self.learner.critics.requires_grad_(False)
        self.mailbox.publish(policy_snapshot(self.learner), self.learner.actor_updates)
        return metrics


class CriticWorker:
    def __init__(self, learner, replay, mailbox):
        self.learner, self.replay, self.mailbox = learner, replay, mailbox
        self.last_policy_version = learner.actor_updates

    def refresh_policy(self):
        snapshot, version = self.mailbox.read()
        if version != self.last_policy_version:
            install_policy(self.learner, snapshot)
            self.last_policy_version = version

    def learn(self, batch, collect_metrics):
        self.refresh_policy()
        return self.learner.update_critic(batch, collect_metrics)

    def update_targets(self):
        if self.learner.algorithm == "masac":
            self.learner.update_targets()
        elif self.learner.updates % self.learner.config.matd3.policy_delay == 0:
            self.learner.update_targets(actor=False, critic=True)
