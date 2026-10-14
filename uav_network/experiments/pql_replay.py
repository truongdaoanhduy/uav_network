"""Single-writer joint replay with measured sampling age and unsampled evictions."""
import torch

from ..algorithms.replay import TensorReplayBuffer


class MeasuredReplay(TensorReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.born = torch.zeros(self.capacity, device=self.device, dtype=torch.int64)
        self.seen = torch.zeros(self.capacity, device=self.device, dtype=torch.bool)
        self.collections = self.inserted = self.sample_uses = 0
        self.overwritten = torch.zeros((), device=self.device, dtype=torch.int64)
        self.unsampled = self.overwritten.clone()
        self.age_sum = self.overwritten.clone()
        self.age_max = self.overwritten.clone()

    @property
    def allocated_bytes(self):
        return super().allocated_bytes + self.born.numel() * 8 + self.seen.numel()

    def add_batch(self, *values):
        count = len(values[3])
        if count < 1 or count > self.capacity:
            raise ValueError("Replay insert must contain 1..capacity joint transitions")
        indices = (torch.arange(count, device=self.device) + self.index) % self.capacity
        occupied = (indices < self.size) if self.size < self.capacity else torch.ones_like(indices, dtype=torch.bool)
        self.overwritten += occupied.sum()
        self.unsampled += (occupied & ~self.seen[indices]).sum()
        super().add_batch(*values)
        self.collections += 1
        self.inserted += count
        self.born[indices] = self.collections
        self.seen[indices] = False

    def sample(self, batch_size):
        if batch_size < 1 or batch_size > self.size:
            raise ValueError("Not enough transitions for this batch")
        indices = torch.randint(self.size, (batch_size,), device=self.device, generator=self.rng)
        # Unique indices avoid concurrent duplicate writes on CUDA.
        self.seen[indices.unique()] = True
        age = self.collections - self.born[indices]
        self.age_sum += age.sum()
        self.age_max = torch.maximum(self.age_max, age.max())
        self.sample_uses += batch_size
        return {name: array[indices] for name, array in self.data.items()}

    def stats(self):
        overwritten, unsampled, age_sum, age_max = torch.stack(
            [self.overwritten, self.unsampled, self.age_sum, self.age_max]).cpu().tolist()
        return {"replay/size": self.size, "replay/bytes": self.allocated_bytes,
                "replay/inserted_rows": self.inserted, "replay/sample_uses": self.sample_uses,
                "replay/overwritten_rows": overwritten, "replay/evicted_unsampled_rows": unsampled,
                "replay/evicted_unsampled_fraction": unsampled / max(1, overwritten),
                "replay/sample_age_mean_collections": age_sum / max(1, self.sample_uses),
                "replay/sample_age_max_collections": age_max}

    def state_dict(self):
        return {"capacity": self.capacity, "index": self.index, "size": self.size,
                "collections": self.collections, "inserted": self.inserted, "sample_uses": self.sample_uses,
                "rng": self.rng.get_state(), "born": self.born[:self.size], "seen": self.seen[:self.size],
                "overwritten": self.overwritten, "unsampled": self.unsampled,
                "age_sum": self.age_sum, "age_max": self.age_max,
                "data": {name: value[:self.size] for name, value in self.data.items()}}

    def load_state_dict(self, saved):
        if saved["capacity"] != self.capacity or not 0 <= saved["size"] <= self.capacity:
            raise ValueError("PQL replay checkpoint capacity/size mismatch")
        if not 0 <= saved["index"] < self.capacity:
            raise ValueError("Invalid PQL replay cursor")
        self.index, self.size = saved["index"], saved["size"]
        for name in ("collections", "inserted", "sample_uses"):
            setattr(self, name, saved[name])
        for name in ("born", "seen"):
            getattr(self, name)[:self.size].copy_(saved[name])
        for name in ("overwritten", "unsampled", "age_sum", "age_max"):
            getattr(self, name).copy_(saved[name])
        for name, value in self.data.items():
            if saved["data"][name].shape != value[:self.size].shape:
                raise ValueError("PQL replay checkpoint transition shape mismatch")
            value[:self.size].copy_(saved["data"][name])
        self.rng.set_state(saved["rng"].cpu())
