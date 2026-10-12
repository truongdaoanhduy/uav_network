"""Gymnasium process collection: one live UavNetSim per spawned worker."""
from copy import deepcopy
from functools import partial
import os

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
import numpy as np

from .env import Action, SwarmSearchEnv
from .observations import observation_dim, state_dim


class GymMissionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config):
        self.config = deepcopy(config)
        self.mission = SwarmSearchEnv(self.config)
        n = config.world.num_uavs
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, (n, observation_dim(config)), np.float32),
            "state": spaces.Box(-np.inf, np.inf, (state_dim(config),), np.float32)})
        self.action_space = spaces.Dict({
            "continuous": spaces.Box(-1., 1., (n, 3), np.float32),
            "destination": spaces.MultiDiscrete(np.full(n, n + 1))})

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Gymnasium removes reset_mask and forwards an empty options dictionary.
        observation, _ = self.mission.reset(seed=seed, options=options or None)
        return {"observation": observation, "state": self.mission.state()}, {
            "worker_pid": os.getpid(), "episode_seed": self.mission.seed}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.mission.step(
            Action(action["continuous"], action["destination"]))
        # Send large nested metrics only at episode boundaries, not every step.
        info = {**info, "worker_pid": os.getpid()} if terminated or truncated else {}
        return {"observation": observation, "state": self.mission.state()}, reward, terminated, truncated, info

    def close(self):
        self.mission.close()


def make_vector(config):
    factories = [partial(GymMissionEnv, config) for _ in range(config.runtime.num_envs)]
    if len(factories) == 1:
        return SyncVectorEnv(factories, autoreset_mode=AutoresetMode.DISABLED)
    # spawn never inherits initialized CUDA or UavNetSim module globals.
    return AsyncVectorEnv(factories, context="spawn", shared_memory=config.runtime.shared_memory,
                          copy=True, autoreset_mode=AutoresetMode.DISABLED)


def random_actions(observation, rng, destinations):
    shape = observation.shape[:-1]
    continuous = rng.uniform(-1, 1, (*shape, 3)).astype(np.float32)
    mask = observation[..., -destinations:] > .5
    mask[..., 0] = True
    # Independent uniform choice over valid destinations, including silent.
    random_scores = np.where(mask, rng.random(mask.shape), -1)
    destination = random_scores.argmax(-1).astype(np.int64)
    active = observation[..., 9] > .5
    continuous[..., :2] *= active[..., None]
    continuous[..., 2] *= active & (destination != 0)
    return Action(continuous, destination)


def unpack_info(infos, index):
    result = {}
    for name, values in infos.items():
        if name.startswith("_") or (f"_{name}" in infos and not infos[f"_{name}"][index]):
            continue
        if isinstance(values, dict):
            result[name] = unpack_info(values, index)
        else:
            value = values[index]
            result[name] = value.tolist() if isinstance(value, np.ndarray) else value.item() if isinstance(value, np.generic) else value
    return result
