"""Preallocated ring buffer containing joint transitions and true termination."""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, observation_shape, state_dim, seed=44):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.index = self.size = 0
        shapes = {"observation": observation_shape, "next_observation": observation_shape,
                  "state": (state_dim,), "next_state": (state_dim,),
                  "continuous": (observation_shape[0], 3),
                  "destination": (observation_shape[0],), "reward": (), "terminated": ()}
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
