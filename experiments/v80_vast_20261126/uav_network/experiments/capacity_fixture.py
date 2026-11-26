"""Small real CPU transitions repeated solely for a disclosed capacity probe."""
import numpy as np
import torch

from uav_network.algorithms.core.action_types import Action
from zhang_xue.environment.core import ZhangXue7NodePacketEnv
from zhang_xue.learners.actor_adapter import (
    encode_actor_input_scalable, make_critic_state_scalable, step_from_policy_action,
)


def real_transition_fixture(scenario, *, seed, steps=4):
    env = ZhangXue7NodePacketEnv(scenario, seed=seed)
    observation, _, mask = env.reset(seed=seed)
    values = {name: [] for name in ("observation", "next_observation", "state", "next_state",
                                   "continuous", "destination", "reward", "terminated")}
    for _ in range(steps):
        encoded = encode_actor_input_scalable(observation, mask)
        action = Action(np.zeros((env.actor_n, 2), dtype=np.float32),
                        np.zeros(env.actor_n, dtype=np.int64))
        next_observation, _, next_mask, reward, terminated, truncated, _ = step_from_policy_action(
            env, action, mask, power_control="fixed")
        next_encoded = encode_actor_input_scalable(next_observation, next_mask)
        transition = dict(observation=encoded, state=make_critic_state_scalable(encoded),
                          next_observation=next_encoded, next_state=make_critic_state_scalable(next_encoded),
                          continuous=action.continuous, destination=action.destination,
                          reward=reward, terminated=float(terminated))
        for name, value in transition.items():
            values[name].append(value)
        observation, mask = next_observation, next_mask
        if terminated or truncated:
            observation, _, mask = env.reset(seed=seed)
    return {name: np.asarray(value, dtype=np.int64 if name == "destination" else np.float32)
            for name, value in values.items()}


def fill_repeated_fixture(replay, fixture, *, chunk_size):
    """Commit every replay slot; no uninitialized or fabricated random tensors."""
    source = {name: torch.as_tensor(value, device=replay.device)
              for name, value in fixture.items()}
    for start in range(0, replay.capacity, chunk_size):
        count = min(chunk_size, replay.capacity - start)
        indices = torch.arange(start, start + count, device=replay.device) % len(fixture["reward"])
        batch = {name: value[indices] for name, value in source.items()}
        replay.add_batch(batch["observation"], batch["state"],
                         Action(batch["continuous"], batch["destination"]), batch["reward"],
                         batch["next_observation"], batch["next_state"], batch["terminated"])
