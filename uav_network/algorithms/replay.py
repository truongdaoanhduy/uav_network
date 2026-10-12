"""Preallocated ring buffer containing joint transitions and true termination."""
import numpy as np
import torch


def transition_shapes(observation_shape, state_dim):
    return {"observation": observation_shape, "next_observation": observation_shape,
            "state": (state_dim,), "next_state": (state_dim,),
            "continuous": (observation_shape[0], 3),
            "destination": (observation_shape[0],), "reward": (), "terminated": ()}


class ReplayBuffer:
    def __init__(self, capacity, observation_shape, state_dim, seed=44):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.index = self.size = 0
        shapes = transition_shapes(observation_shape, state_dim)
        self.data = {name: np.empty((capacity, *shape), dtype=np.int64 if name == "destination" else np.float32)
                     for name, shape in shapes.items()}

    def __len__(self):
        return self.size

    def add(self, observation, state, action, reward, next_observation, next_state, terminated):
        values = dict(observation=observation, state=state, continuous=action.continuous,
                      destination=action.destination, reward=reward,
                      next_observation=next_observation, next_state=next_state,
                      terminated=float(terminated))
        for name, value in values.items():
            self.data[name][self.index] = value
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if batch_size > self.size:
            raise ValueError("Not enough transitions for this batch")
        indices = self.rng.choice(self.size, batch_size, replace=False)
        return {name: array[indices].copy() for name, array in self.data.items()}


class TensorReplayBuffer:
    """Batched ring storage on CPU or CUDA; uniform samples with replacement."""
    def __init__(self, capacity, observation_shape, state_dim, seed=44, device="cpu", pin_memory=False):
        self.capacity = capacity
        self.device = torch.device(device)
        self.pin_memory = pin_memory and self.device.type == "cpu"
        self.index = self.size = 0
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.data = {name: torch.empty((capacity, *shape), device=self.device,
                                     dtype=torch.long if name == "destination" else torch.float32,
                                     pin_memory=self.pin_memory)
                     for name, shape in transition_shapes(observation_shape, state_dim).items()}

    @property
    def allocated_bytes(self):
        return sum(value.numel() * value.element_size() for value in self.data.values())

    def __len__(self):
        return self.size

    def add_batch(self, observation, state, action, reward, next_observation, next_state, terminated):
        values = dict(observation=observation, state=state, continuous=action.continuous,
                      destination=action.destination, reward=reward,
                      next_observation=next_observation, next_state=next_state, terminated=terminated)
        count = len(reward)
        if count > self.capacity:
            raise ValueError("Replay capacity must accommodate a complete collection batch")
        indices = (torch.arange(count, device=self.device) + self.index) % self.capacity
        for name, value in values.items():
            tensor = torch.as_tensor(value, dtype=self.data[name].dtype, device=self.device)
            self.data[name][indices] = tensor
        self.index = (self.index + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size):
        if batch_size > self.size:
            raise ValueError("Not enough transitions for this batch")
        indices = torch.randint(self.size, (batch_size,), device=self.device, generator=self.rng)
        result = {name: array[indices] for name, array in self.data.items()}
        if self.pin_memory:
            result = {name: value.pin_memory() for name, value in result.items()}
        return result
