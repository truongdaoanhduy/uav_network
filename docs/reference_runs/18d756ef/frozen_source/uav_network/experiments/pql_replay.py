"""Single-writer replay; collection age is measured, policy age is unavailable."""
import os
from math import prod
import torch

from ..algorithms.replay import TensorReplayBuffer


class MeasuredReplay(TensorReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.born = torch.full((self.capacity,), -1, device=self.device, dtype=torch.int64)
        self.seen = torch.zeros(self.capacity, device=self.device, dtype=torch.bool)
        self.collections = self.inserted = self.sample_uses = 0
        self.overwritten = torch.zeros((), device=self.device, dtype=torch.int64)
        for name in ("unsampled", "age_sum", "age_max", "age_count", "measured_evictions", "known_rows"):
            setattr(self, name, self.overwritten.clone())

    @property
    def allocated_bytes(self):
        return super().allocated_bytes + self.born.numel() * 8 + self.seen.numel()

    def add_batch(self, *values):
        count = len(values[3])
        if count < 1 or count > self.capacity:
            raise ValueError("Replay insert must contain 1..capacity joint transitions")
        indices = (torch.arange(count, device=self.device) + self.index) % self.capacity
        occupied = (indices < self.size) if self.size < self.capacity else torch.ones_like(indices, dtype=torch.bool)
        known = self.born[indices] >= 0
        self.overwritten += occupied.sum()
        self.measured_evictions += (occupied & known).sum()
        self.unsampled += (occupied & known & ~self.seen[indices]).sum()
        self.known_rows += (~known).sum()
        super().add_batch(*values)
        self.collections += 1
        self.inserted += count
        self.born[indices] = self.collections
        self.seen[indices] = False

    def sample(self, batch_size):
        if batch_size < 1 or batch_size > self.size:
            raise ValueError("Not enough transitions for this batch")
        indices = torch.randint(self.size, (batch_size,), device=self.device, generator=self.rng)
        self.seen[indices.unique()] = True
        born = self.born[indices]
        known = born >= 0
        age = torch.where(known, self.collections - born, 0)
        self.age_sum += age.sum()
        self.age_max = torch.maximum(self.age_max, age.max())
        self.age_count += known.sum()
        self.sample_uses += batch_size
        result = {name: array[indices] for name, array in self.data.items()}
        if self.pin_memory:
            result = {name: value.pin_memory() for name, value in result.items()}
        return result

    def stats(self):
        overwritten, unsampled, age_sum, age_max, ages, evictions, known = torch.stack(
            [self.overwritten, self.unsampled, self.age_sum, self.age_max, self.age_count,
             self.measured_evictions, self.known_rows]).cpu().tolist()
        return {"replay/size": self.size, "replay/bytes": self.allocated_bytes,
                "replay/inserted_rows": self.inserted, "replay/sample_uses": self.sample_uses,
                "replay/overwritten_rows": overwritten, "replay/evicted_unsampled_rows": unsampled,
                "replay/eviction_measured_rows": evictions,
                "replay/evicted_unsampled_fraction": unsampled / evictions if evictions else None,
                "replay/sample_age_measured_uses": ages,
                "replay/sample_age_mean_collections": age_sum / ages if ages else None,
                "replay/sample_age_max_collections": age_max if ages else None,
                "replay/measurement_unknown_rows": self.size - known,
                "replay/policy_age_available": 0}

    def measurement_state(self):
        result = {name: getattr(self, name) for name in (
            "collections", "inserted", "sample_uses", "overwritten", "unsampled", "age_sum",
            "age_max", "age_count", "measured_evictions", "known_rows")}
        result.update(born=self.born[:self.size], seen=self.seen[:self.size])
        return result

    def load_measurement_state(self, saved):
        for name in ("collections", "inserted", "sample_uses"):
            setattr(self, name, saved[name])
        for name in ("born", "seen"):
            getattr(self, name)[:self.size].copy_(saved[name])
        for name in ("overwritten", "unsampled", "age_sum", "age_max"):
            getattr(self, name).copy_(saved[name])
        # Original PQL checkpoints measured all their rows; old BASE has no metadata.
        for name, fallback in (("age_count", saved["sample_uses"]),
                               ("measured_evictions", saved["overwritten"]), ("known_rows", self.size)):
            getattr(self, name).copy_(torch.as_tensor(saved.get(name, fallback), device=self.device))

    def state_dict(self):
        def compact(value):
            result = value.detach().cpu()
            if result.untyped_storage().nbytes() != result.numel() * result.element_size():
                result = result.clone()
            return result

        measurement = self.measurement_state()
        for name in ("born", "seen"):
            measurement[name] = compact(measurement[name])
        return {"capacity": self.capacity, "index": self.index, "size": self.size,
                "device_type": self.device.type, "rng": self.rng.get_state(), **measurement,
                # save() drains workers and serializes synchronously. Borrow full
                # CPU storage, compact partial views, and avoid extra CUDA replay copies.
                "data": {name: compact(value[:self.size]) for name, value in self.data.items()}}

    def load_state_dict(self, saved):
        if saved["capacity"] != self.capacity or not 0 <= saved["size"] <= self.capacity:
            raise ValueError("PQL replay checkpoint capacity/size mismatch")
        if not 0 <= saved["index"] < self.capacity:
            raise ValueError("Invalid PQL replay cursor")
        if saved.get("device_type", self.device.type) != self.device.type:
            raise ValueError("PQL replay RNG resume requires the original device type")
        restored_rng = torch.Generator(device=self.device)
        try:
            restored_rng.set_state(saved["rng"].cpu())
        except (RuntimeError, TypeError, AttributeError) as error:
            # Legacy checkpoints lack device_type; validate their RNG before copying rows.
            raise ValueError("PQL replay RNG state is incompatible with this device") from error
        if saved["data"].keys() != self.data.keys():
            raise ValueError("PQL replay checkpoint transition fields mismatch")
        for name, value in self.data.items():
            incoming = saved["data"][name]
            if incoming.shape != value[:saved["size"]].shape or incoming.dtype != value.dtype:
                raise ValueError("PQL replay checkpoint transition shape/dtype mismatch")
        for name in ("born", "seen"):
            value = getattr(self, name)
            if saved[name].shape != value[:saved["size"]].shape or saved[name].dtype != value.dtype:
                raise ValueError("PQL replay checkpoint measurement shape/dtype mismatch")
        self.index, self.size = saved["index"], saved["size"]
        self.load_measurement_state(saved)
        for name, value in self.data.items():
            value[:self.size].copy_(saved["data"][name])
        self.rng = restored_rng


def make_replay(config, observation_shape, state_dim, learner_device, *, measured=True):
    """Construct the explicitly configured layout; default storage stays unchanged."""
    from ..algorithms.replay import resolve_replay_device
    settings = config.algorithm_config
    if config.runtime.replay_cpu_fraction:
        from ..algorithms.hybrid_replay import HybridMeasuredReplay
        from ..algorithms.replay import transition_shapes
        hot_device = torch.device(learner_device)
        target = os.environ.get("UAV_HYBRID_HOT_GPU", "").strip()
        if target:
            # Explicit opt-in for BASE learner on GPU1 with hybrid replay hot
            # tier on GPU0, where observed free VRAM is greater. No change to
            # the default MASAC/MATD3 placement or any PQL placement.
            if config.pql.enabled:
                raise ValueError("UAV_HYBRID_HOT_GPU is BASE-only; PQL device split is unsupported")
            if target not in {"cuda:0", "cuda:1"}:
                raise ValueError("UAV_HYBRID_HOT_GPU must be cuda:0 or cuda:1")
            hot_device = torch.device(target)
            if not torch.cuda.is_available() or hot_device.index >= torch.cuda.device_count():
                raise RuntimeError(f"Replay hot GPU {target} is not available")
            if hot_device == torch.device(learner_device):
                raise RuntimeError("Replay hot GPU must differ from the busy learner GPU")
            # Check true driver free bytes AFTER Warp vector initialization.
            # PyTorch allocated/reserved alone cannot account for Warp or
            # concurrent evaluation. Keep a physical reserve for later peaks.
            shapes = transition_shapes(observation_shape, state_dim, config.continuous_dim)
            hot_count = settings.replay_capacity - round(settings.replay_capacity * config.runtime.replay_cpu_fraction)
            bytes_per_row = sum(
                (8 if key == "destination" else 4) *
                prod(shape)
                for key, shape in shapes.items()
            )
            required_bytes = hot_count * bytes_per_row
            reserve_gib = float(os.environ.get("UAV_HYBRID_MIN_FREE_GIB", "3.0"))
            if not 1.0 <= reserve_gib <= 10.0:
                raise ValueError("UAV_HYBRID_MIN_FREE_GIB must be 1..10 GiB")
            free_bytes, total_bytes = torch.cuda.mem_get_info(hot_device)
            if free_bytes < required_bytes + reserve_gib * 2**30:
                raise MemoryError(
                    f"Not enough free VRAM on {hot_device} for hybrid hot replay: "
                    f"free={free_bytes/2**30:.2f} GiB, "
                    f"hot_storage={required_bytes/2**30:.2f} GiB, "
                    f"reserve={reserve_gib:.2f} GiB; stop before allocating to avoid OOM"
                )
            print(
                f"HYBRID_REPLAY_GPU_SPLIT learner={learner_device} hot={hot_device} "
                f"hot_storage={required_bytes/2**30:.2f}GiB "
                f"preallocation_free={free_bytes/2**30:.2f}GiB "
                f"total={total_bytes/2**30:.2f}GiB "
                f"reserved_headroom={reserve_gib:.2f}GiB", flush=True
            )
        return HybridMeasuredReplay(settings.replay_capacity, observation_shape, state_dim,
                                    config.runtime.seed, hot_device=hot_device,
                                    cpu_fraction=config.runtime.replay_cpu_fraction, continuous_dim=config.continuous_dim)
    device = resolve_replay_device(config.runtime.replay_device, learner_device)
    cls = MeasuredReplay if measured else TensorReplayBuffer
    return cls(settings.replay_capacity, observation_shape, state_dim, config.runtime.seed, device,
               pin_memory=device.type == "cpu" and torch.device(learner_device).type == "cuda", continuous_dim=config.continuous_dim)
