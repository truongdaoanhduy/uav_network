"""Hybrid replay uses the same uniform global ring and CPU RNG across both tiers."""
import torch
from .replay import transition_shapes


def validate_hybrid_state(saved, capacity, observation_shape, state_dim, cpu_fraction, hot_device_type=None, continuous_dim=3):
    if (not isinstance(saved, dict) or saved.get("layout") != "hybrid"
            or saved.get("capacity") != capacity
            or saved.get("cpu_capacity") != round(capacity * cpu_fraction)
            or type(saved.get("size")) is not int or not 0 <= saved["size"] <= capacity
            or type(saved.get("index")) is not int or not 0 <= saved["index"] < capacity):
        raise ValueError("Hybrid replay checkpoint capacity/layout/cursor mismatch")
    if hot_device_type is not None and saved.get("hot_device_type") != hot_device_type:
        raise ValueError("Hybrid replay hot-device type mismatch")
    counters = ("collections", "inserted", "sample_uses", "overwritten", "unsampled", "age_sum",
                "age_max", "age_count", "measured_evictions", "known_rows")
    if any(type(saved.get(name)) is not int or saved[name] < 0 for name in counters):
        raise ValueError("Hybrid replay checkpoint measurement counters are invalid")
    if (saved["size"] != min(capacity, saved["inserted"])
            or saved["index"] != saved["inserted"] % capacity
            or saved["known_rows"] > saved["size"]
            or saved["age_count"] > saved["sample_uses"]):
        raise ValueError("Hybrid replay checkpoint cursor/counters are inconsistent")
    try:
        torch.Generator(device="cpu").set_state(saved["rng"].cpu())
    except (KeyError, RuntimeError, TypeError, AttributeError) as error:
        raise ValueError("Hybrid replay RNG checkpoint is invalid") from error
    shapes = transition_shapes(observation_shape, state_dim, continuous_dim)
    sizes = {"cpu_data": min(saved["size"], saved["cpu_capacity"]),
             "hot_data": max(0, saved["size"] - saved["cpu_capacity"])}
    for group, size in sizes.items():
        data = saved.get(group)
        if not isinstance(data, dict) or data.keys() != shapes.keys():
            raise ValueError("Hybrid replay checkpoint transition fields mismatch")
        for name, shape in shapes.items():
            value = data[name]
            dtype = torch.int64 if name == "destination" else torch.float32
            if (not isinstance(value, torch.Tensor) or value.shape != (size, *shape)
                    or value.dtype != dtype):
                raise ValueError("Hybrid replay checkpoint transition shape/dtype mismatch")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Hybrid replay checkpoint transitions must be finite")
    for name, dtype in (("born", torch.int64), ("seen", torch.bool)):
        value = saved.get(name)
        if (not isinstance(value, torch.Tensor) or value.shape != (saved["size"],)
                or value.dtype != dtype):
            raise ValueError("Hybrid replay checkpoint measurement shape/dtype mismatch")


class HybridMeasuredReplay:
    """CPU + hot-device replay with one global ring/RNG."""
    def __init__(self, capacity, observation_shape, state_dim_value, seed=44, hot_device="cuda:1", cpu_fraction=0.5, continuous_dim=3):
        if capacity < 2 or not 0 < cpu_fraction < 1:
            raise ValueError("Hybrid replay requires capacity >= 2 and 0 < cpu_fraction < 1")
        self.capacity = capacity
        self.continuous_dim = continuous_dim
        self.observation_shape, self.state_dim = observation_shape, state_dim_value
        self.cpu_capacity = round(capacity * cpu_fraction)
        self.hot_capacity = capacity - self.cpu_capacity
        if not self.cpu_capacity or not self.hot_capacity:
            raise ValueError("Hybrid split must allocate at least one row per device")
        self.hot_device = torch.device(hot_device)
        self.device = self.hot_device
        self.index = self.size = 0
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        shapes = transition_shapes(observation_shape, state_dim_value, continuous_dim)
        self.cpu_data = {name: torch.empty((self.cpu_capacity, *shape), device="cpu",
                                           dtype=torch.long if name == "destination" else torch.float32)
                         for name, shape in shapes.items()}
        self.hot_data = {name: torch.empty((self.hot_capacity, *shape), device=self.hot_device,
                                           dtype=torch.long if name == "destination" else torch.float32)
                         for name, shape in shapes.items()}
        self.born = torch.full((capacity,), -1, dtype=torch.int64)
        self.seen = torch.zeros(capacity, dtype=torch.bool)
        self.collections = self.inserted = self.sample_uses = 0
        self.overwritten = self.unsampled = self.age_sum = self.age_max = 0
        self.age_count = self.measured_evictions = self.known_rows = 0
    def __len__(self):
        return self.size
    @property
    def cpu_allocated_bytes(self):
        data = sum(value.numel() * value.element_size() for value in self.cpu_data.values())
        return data + self.born.numel() * self.born.element_size() + self.seen.numel() * self.seen.element_size()
    @property
    def hot_allocated_bytes(self):
        return sum(value.numel() * value.element_size() for value in self.hot_data.values())
    @property
    def allocated_bytes(self):
        return self.cpu_allocated_bytes + self.hot_allocated_bytes
    @staticmethod
    def _rows(value, positions, device, dtype):
        tensor = torch.as_tensor(value)
        selected = tensor.index_select(0, positions.to(tensor.device))
        return selected.to(device=device, dtype=dtype)
    def add_batch(self, observation, state, action, reward, next_observation, next_state, terminated):
        values = dict(observation=observation, state=state, continuous=action.continuous,
                      destination=action.destination, reward=reward, next_observation=next_observation,
                      next_state=next_state, terminated=terminated)
        count = len(reward)
        if count < 1 or count > self.capacity:
            raise ValueError("Replay insert must contain 1..capacity joint transitions")
        indices = (torch.arange(count) + self.index) % self.capacity
        occupied = (indices < self.size) if self.size < self.capacity else torch.ones_like(indices, dtype=torch.bool)
        known = self.born[indices] >= 0
        self.overwritten += int(occupied.sum())
        self.measured_evictions += int((occupied & known).sum())
        self.unsampled += int((occupied & known & ~self.seen[indices]).sum())
        self.known_rows += int((~known).sum())
        cpu_mask = indices < self.cpu_capacity
        hot_mask = ~cpu_mask
        cpu_positions = torch.nonzero(cpu_mask, as_tuple=False).flatten()
        hot_positions = torch.nonzero(hot_mask, as_tuple=False).flatten()
        cpu_indices = indices[cpu_mask]
        hot_indices = indices[hot_mask] - self.cpu_capacity
        for name, value in values.items():
            if cpu_positions.numel():
                rows = self._rows(value, cpu_positions, "cpu", self.cpu_data[name].dtype)
                self.cpu_data[name].index_copy_(0, cpu_indices, rows)
            if hot_positions.numel():
                rows = self._rows(value, hot_positions, self.hot_device, self.hot_data[name].dtype)
                self.hot_data[name].index_copy_(0, hot_indices.to(self.hot_device), rows)
        self.index = (self.index + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.collections += 1
        self.inserted += count
        self.born[indices] = self.collections
        self.seen[indices] = False
    def sample(self, batch_size):
        if batch_size < 1 or batch_size > self.size:
            raise ValueError("Not enough transitions for this batch")
        indices = torch.randint(self.size, (batch_size,), generator=self.rng)
        self.seen[indices.unique()] = True
        born = self.born[indices]
        known = born >= 0
        age = torch.where(known, self.collections - born, 0)
        self.age_sum += int(age.sum())
        self.age_max = max(self.age_max, int(age.max()))
        self.age_count += int(known.sum())
        self.sample_uses += batch_size
        cpu_mask = indices < self.cpu_capacity
        hot_mask = ~cpu_mask
        cpu_positions = torch.nonzero(cpu_mask, as_tuple=False).flatten()
        hot_positions = torch.nonzero(hot_mask, as_tuple=False).flatten()
        cpu_indices = indices[cpu_mask]
        hot_indices = indices[hot_mask] - self.cpu_capacity
        result = {}
        for name in self.cpu_data:
            shape = self.cpu_data[name].shape[1:]
            out = torch.empty((batch_size, *shape), device=self.hot_device, dtype=self.cpu_data[name].dtype)
            if cpu_positions.numel():
                rows = self.cpu_data[name].index_select(0, cpu_indices)
                if self.hot_device.type == "cuda":
                    rows = rows.pin_memory().to(self.hot_device, non_blocking=True)
                else:
                    rows = rows.to(self.hot_device)
                out.index_copy_(0, cpu_positions.to(self.hot_device), rows)
            if hot_positions.numel():
                rows = self.hot_data[name].index_select(0, hot_indices.to(self.hot_device))
                out.index_copy_(0, hot_positions.to(self.hot_device), rows)
            result[name] = out
        return result
    def stats(self):
        evictions, ages = self.measured_evictions, self.age_count
        return {"replay/size": self.size, "replay/bytes": self.allocated_bytes,
                "replay/cpu_bytes": self.cpu_allocated_bytes, "replay/hot_bytes": self.hot_allocated_bytes,
                "replay/hybrid": 1, "replay/inserted_rows": self.inserted,
                "replay/sample_uses": self.sample_uses, "replay/overwritten_rows": self.overwritten,
                "replay/evicted_unsampled_rows": self.unsampled, "replay/eviction_measured_rows": evictions,
                "replay/evicted_unsampled_fraction": self.unsampled / evictions if evictions else None,
                "replay/sample_age_measured_uses": ages, "replay/sample_age_mean_collections": self.age_sum / ages if ages else None,
                "replay/sample_age_max_collections": self.age_max if ages else None, "replay/measurement_unknown_rows": self.size - self.known_rows,
                "replay/policy_age_available": 0}
    def measurement_state(self):
        return {"collections": self.collections, "inserted": self.inserted, "sample_uses": self.sample_uses,
                "overwritten": self.overwritten, "unsampled": self.unsampled, "age_sum": self.age_sum,
                "age_max": self.age_max, "age_count": self.age_count, "measured_evictions": self.measured_evictions, "known_rows": self.known_rows,
                "born": self.born[:self.size], "seen": self.seen[:self.size]}

    def state_dict(self, *, copy=False):
        cpu_size = self.cpu_capacity if self.size >= self.cpu_capacity else self.size
        hot_size = max(0, self.size - self.cpu_capacity)
        def compact(value, size):
            populated = value[:size]
            if copy:
                return populated.detach().cpu().clone()
            return populated if size == len(value) else populated.clone()
        return {"layout": "hybrid", "capacity": self.capacity, "cpu_capacity": self.cpu_capacity,
                "hot_device_type": self.hot_device.type, "index": self.index, "size": self.size,
                "rng": self.rng.get_state(),
                **{name: value.clone() if copy and isinstance(value, torch.Tensor) else value
                   for name, value in self.measurement_state().items()},
                "cpu_data": {name: compact(value, cpu_size) for name, value in self.cpu_data.items()},
                "hot_data": {name: compact(value, hot_size) for name, value in self.hot_data.items()}}

    def load_state_dict(self, saved):
        validate_hybrid_state(saved, self.capacity, self.observation_shape, self.state_dim,
                              self.cpu_capacity / self.capacity, self.hot_device.type, self.continuous_dim)
        if (saved.get("layout") != "hybrid" or saved["capacity"] != self.capacity
                or saved["cpu_capacity"] != self.cpu_capacity or not 0 <= saved["size"] <= self.capacity
                or not 0 <= saved["index"] < self.capacity):
            raise ValueError("Hybrid replay checkpoint capacity/layout/cursor mismatch")
        if saved.get("hot_device_type") != self.hot_device.type:
            raise ValueError("Hybrid replay hot-device type mismatch")
        cpu_size = self.cpu_capacity if saved["size"] >= self.cpu_capacity else saved["size"]
        hot_size = max(0, saved["size"] - self.cpu_capacity)
        if saved["cpu_data"].keys() != self.cpu_data.keys() or saved["hot_data"].keys() != self.hot_data.keys():
            raise ValueError("Hybrid replay checkpoint transition fields mismatch")
        for name in self.cpu_data:
            if saved["cpu_data"][name].shape != self.cpu_data[name][:cpu_size].shape:
                raise ValueError("Hybrid CPU replay checkpoint shape mismatch")
            if saved["hot_data"][name].shape != self.hot_data[name][:hot_size].shape:
                raise ValueError("Hybrid hot replay checkpoint shape mismatch")
        self.index, self.size = saved["index"], saved["size"]
        self.collections, self.inserted, self.sample_uses = saved["collections"], saved["inserted"], saved["sample_uses"]
        for name in ("overwritten", "unsampled", "age_sum", "age_max", "age_count", "measured_evictions", "known_rows"):
            setattr(self, name, int(saved[name]))
        self.born[:self.size].copy_(saved["born"])
        self.seen[:self.size].copy_(saved["seen"])
        for name in self.cpu_data:
            self.cpu_data[name][:cpu_size].copy_(saved["cpu_data"][name])
            self.hot_data[name][:hot_size].copy_(saved["hot_data"][name].to(self.hot_device))
        self.rng.set_state(saved["rng"].cpu())
