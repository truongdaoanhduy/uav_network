"""Bounded full-capacity CPU replay memory diagnostic; never a quality evaluation."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from threading import Event, Thread
import time
import traceback

import numpy as np
import torch

from ..configuration import load_config
from ..environment.env import Action
from ..environment.observations import observation_dim, state_dim
from ..algorithms.replay import transition_shapes
from .checkpoint_store import file_sha256
from .environment_checkpoints import initialize_environment, snapshot_environment
from .pql_system import PQLSystem
from .training import seed_runtime

class HybridMeasuredReplay:
    """Probe-only 50/50 CPU + hot-device replay with one global ring/RNG."""
    def __init__(self, capacity, observation_shape, state_dim_value, seed=44, hot_device="cuda:1"):
        if capacity < 2 or capacity % 2:
            raise ValueError("Hybrid probe replay requires an even capacity >= 2")
        self.capacity = capacity
        self.cpu_capacity = capacity // 2
        self.hot_capacity = capacity - self.cpu_capacity
        self.hot_device = torch.device(hot_device)
        self.device = self.hot_device
        self.index = self.size = 0
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        shapes = transition_shapes(observation_shape, state_dim_value)
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

    def state_dict(self):
        cpu_size = self.cpu_capacity if self.size >= self.cpu_capacity else self.size
        hot_size = max(0, self.size - self.cpu_capacity)
        full = self.size == self.capacity
        compact = lambda value, size: value if full else value[:size].clone()
        return {"layout": "hybrid50", "capacity": self.capacity, "cpu_capacity": self.cpu_capacity,
                "hot_device_type": self.hot_device.type, "index": self.index, "size": self.size,
                "rng": self.rng.get_state(), **self.measurement_state(),
                "cpu_data": {name: compact(value, cpu_size) for name, value in self.cpu_data.items()},
                "hot_data": {name: compact(value, hot_size) for name, value in self.hot_data.items()}}

    def load_state_dict(self, saved):
        if (saved.get("layout") != "hybrid50" or saved["capacity"] != self.capacity
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

def probe_config():
    c = load_config("configs/comparison-20k-steps1800-scenario-v3.yaml")
    c.runtime.algorithm, c.runtime.num_envs = "masac", 2048
    c.runtime.device, c.runtime.replay_device = "cuda", "cpu"
    c.runtime.amp_mode, c.runtime.compile_models = "fp16", True
    c.runtime.fused_optimizers = c.runtime.vectorize_critics = True
    c.runtime.torch_threads = 1
    c.tracking.enabled = False
    c.masac.replay_capacity, c.masac.batch_size = 1_000_000, 4096
    c.pql.enabled, c.pql.mode, c.pql.checkpoint_replay = True, "async", True
    c.pql.collector_device = "cuda:0"
    c.pql.learner_device = c.pql.policy_device = "cuda:1"
    c.pql.warmup_collection_steps, c.pql.critic_sample_ratio = 2, 16
    c.pql.worker_timeout_s = 1200
    return c.validate()

def fill_replay(replay, values):
    """Repeat one real transition batch for MEMORY testing, including the tail."""
    observation, state, action, reward, nxt, next_state, terminated = values
    cpu = [v.detach().cpu() for v in (
        observation, state, action.continuous, action.destination,
        reward, nxt, next_state, terminated)]
    inserted = 0
    while len(replay) < replay.capacity:
        count = min(len(cpu[4]), replay.capacity - len(replay))
        replay.add_batch(cpu[0][:count], cpu[1][:count],
                         Action(cpu[2][:count], cpu[3][:count]), cpu[4][:count],
                         cpu[5][:count], cpu[6][:count], cpu[7][:count])
        inserted += count
    return inserted

def _host_memory():
    status = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/self/status").read_text().splitlines()
              if line.startswith(("VmRSS:", "VmHWM:"))}
    available = next(int(line.split()[1]) * 1024
                     for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    limit = current = None
    for base, limit_name, current_name in (
            (Path("/sys/fs/cgroup"), "memory.max", "memory.current"),
            (Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes", "memory.usage_in_bytes")):
        if (base / limit_name).is_file() and (base / current_name).is_file():
            raw = (base / limit_name).read_text().strip()
            if raw.isdigit():
                limit, current = int(raw), int((base / current_name).read_text())
                break
    effective = min(available, max(0, limit - current)) if limit is not None else available
    return {"host_rss_bytes": status["VmRSS"], "host_hwm_bytes": status["VmHWM"],
            "system_available_bytes": available, "effective_available_bytes": effective,
            "cgroup_limit_bytes": limit, "cgroup_current_bytes": current}

def _require_two_t4():
    message = "CPU replay probe requires exactly two T4 CUDA devices"
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(message)
    devices = [torch.cuda.get_device_properties(i) for i in range(2)]
    if any("t4" not in device.name.lower() for device in devices):
        raise RuntimeError(message)
    return [{"device": i, "name": device.name, "total_bytes": device.total_memory,
             "compute_capability": [device.major, device.minor]}
            for i, device in enumerate(devices)]

def _gpu_memory():
    return [{"device": i, "name": torch.cuda.get_device_name(i),
             "total_bytes": torch.cuda.get_device_properties(i).total_memory,
             "allocated_bytes": torch.cuda.memory_allocated(i),
             "peak_allocated_bytes": torch.cuda.max_memory_allocated(i),
             "peak_reserved_bytes": torch.cuda.max_memory_reserved(i)}
            for i in range(torch.cuda.device_count())]

def _digest(batch):
    digest = hashlib.sha256()
    for name, value in sorted(batch.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def run_probe(output, report=None, layout="cpu"):
    hardware = _require_two_t4()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c = probe_config()
    # Bound compiler subprocess RAM as well as Torch operator threads.
    from torch._inductor import config as inductor_config
    from ..environment.gpu.vector import WarpVectorEnv
    inductor_config.compile_threads = 1
    seed_runtime(c)
    for i in range(2):
        torch.cuda.reset_peak_memory_stats(i)
    result = {"schema": 1, "status": "running", "scope": "memory_and_execution_only",
              "data": "one real 2048-world transition batch repeated to fill replay; not training-quality evidence",
              "config": c.to_dict(), "commit": os.getenv("PROBE_COMMIT"), "layout": layout,
              "hardware": hardware, "compiler_threads": 1, "events": []}
    stop = Event()
    peaks = {"rss_peak_bytes": 0, "effective_available_min_bytes": _host_memory()["effective_available_bytes"],
             "cgroup_current_peak_bytes": 0}

    def sample_memory():
        while not stop.is_set():
            m = _host_memory()
            peaks["rss_peak_bytes"] = max(peaks["rss_peak_bytes"], m["host_rss_bytes"], m["host_hwm_bytes"])
            peaks["effective_available_min_bytes"] = min(peaks["effective_available_min_bytes"], m["effective_available_bytes"])
            peaks["cgroup_current_peak_bytes"] = max(peaks["cgroup_current_peak_bytes"], m["cgroup_current_bytes"] or 0)
            stop.wait(.1)

    sampler = Thread(target=sample_memory, daemon=True)
    sampler.start()
    started = time.perf_counter()
    engine = vector = None

    def event(phase, **values):
        row = {"phase": phase, "elapsed_s": time.perf_counter() - started,
               **_host_memory(), "gpu": _gpu_memory(), **values}
        result["events"].append(row)
        result["host_peaks"] = dict(peaks)
        temporary = output / "cpu-replay-result.json.tmp"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(output / "cpu-replay-result.json")
        print("CPU_REPLAY", json.dumps(row), flush=True)
        if report:
            report(row)

    try:
        event("initializing")
        replay = (HybridMeasuredReplay(c.masac.replay_capacity,
                                       (c.world.num_uavs, observation_dim(c)), state_dim(c),
                                       seed=c.runtime.seed, hot_device="cuda:1")
                  if layout == "hybrid" else None)
        engine = PQLSystem(c, replay=replay)
        vector = WarpVectorEnv(c, device=engine.collector_device)
        obs, returns, lengths, seeds, next_seed = initialize_environment(vector, engine.initial_progress, None)
        rng = np.random.default_rng(c.runtime.seed + 1)
        real_collections = 0

        def collect():
            nonlocal obs, real_collections
            before = time.perf_counter()
            action = (vector.random_actions(obs["observation"]) if real_collections == 0
                      else engine.act_batch(obs["observation"], step=real_collections * c.runtime.num_envs,
                                            return_tensors=True))
            nxt, reward, terminated, truncated, _ = vector.step(
                {"continuous": action.continuous, "destination": action.destination})
            torch.cuda.synchronize(engine.collector_device)
            if bool((terminated | truncated).any()):
                raise RuntimeError("Unexpected terminal world in the short memory probe")
            values = (obs["observation"], obs["state"], action, reward,
                      nxt["observation"], nxt["state"], terminated)
            returns.add_(reward.to(torch.float64))
            lengths.add_(1)
            obs = nxt
            real_collections += 1
            return values, time.perf_counter() - before

        first, collector_s = collect()
        engine.submit(first)
        engine.flush()
        event("first_real_collection", collector_s=collector_s, updates=engine.critic.updates)
        before = time.perf_counter()
        inserted = fill_replay(engine.replay, first)
        assert len(engine.replay) == 1_000_000
        if layout == "hybrid":
            assert engine.replay.cpu_capacity == engine.replay.hot_capacity == 500_000
            assert engine.replay.hot_device == engine.critic_device
        else:
            assert engine.replay.device.type == "cpu"
            assert all(not value.is_pinned() for value in engine.replay.data.values())
        event("replay_full", duplicated_inserted_rows=inserted, replay_rows=len(engine.replay),
              replay_bytes=engine.replay.allocated_bytes,
              replay_cpu_bytes=getattr(engine.replay, "cpu_allocated_bytes", engine.replay.allocated_bytes),
              replay_hot_bytes=getattr(engine.replay, "hot_allocated_bytes", 0),
              fill_s=time.perf_counter() - before)
        before = time.perf_counter()
        sample = engine.replay.sample(c.masac.batch_size)
        sample_s = time.perf_counter() - before
        if layout == "hybrid":
            assert all(value.device == engine.critic_device for value in sample.values())
        else:
            assert all(value.is_pinned() for value in sample.values())
        from ..algorithms.optimization import tensor_batch
        before = time.perf_counter()
        moved = tensor_batch(sample, engine.critic_device)
        torch.cuda.synchronize(engine.critic_device)
        event("batch_transfer", sample_s=sample_s, transfer_s=time.perf_counter() - before,
              batch_bytes=sum(v.numel() * v.element_size() for v in sample.values()))
        del sample, moved
        for cycle in range(3):
            values, collector_s = collect()
            before, old = time.perf_counter(), engine.critic.updates
            engine.submit(values)
            engine.flush()
            learner_s = time.perf_counter() - before
            assert engine.critic.updates - old == 16
            assert all(math.isfinite(float(v)) for v in engine.latest.values())
            event("learning", cycle=cycle + 1, collector_s=collector_s, learner_s=learner_s,
                  updates=engine.critic.updates, actor_updates=engine.policy.actor_updates,
                  replay_rows=len(engine.replay), metrics=engine.latest)
        progress = {"episodes": 0, "global_step": real_collections * c.runtime.num_envs,
                    "collection_steps": real_collections, "next_seed": next_seed, "records": []}
        checkpoint = output / "memory-probe-resume.pt"
        event("before_save", disk_free_bytes=shutil.disk_usage(output).free)
        before = time.perf_counter()
        environment = snapshot_environment(vector, returns, lengths, seeds, next_seed)
        engine.save(checkpoint, progress,
                    collector_rng={"numpy": rng.bit_generator.state,
                                   "torch": vector.action_generator.get_state().cpu()},
                    environment=environment)
        save_s = time.perf_counter() - before
        del environment
        checkpoint_hash = file_sha256(checkpoint)
        expected_sample = _digest(engine.replay.sample(c.masac.batch_size))
        updates_before_resume = engine.critic.updates
        event("saved", save_s=save_s, checkpoint_bytes=checkpoint.stat().st_size,
              checkpoint_sha256=checkpoint_hash)
        engine.close()
        engine = None
        gc.collect()
        torch.cuda.empty_cache()
        event("before_restore")
        before = time.perf_counter()
        replay = (HybridMeasuredReplay(c.masac.replay_capacity,
                                       (c.world.num_uavs, observation_dim(c)), state_dim(c),
                                       seed=c.runtime.seed, hot_device="cuda:1")
                  if layout == "hybrid" else None)
        engine = PQLSystem(c, checkpoint=checkpoint, replay=replay)
        obs, returns, lengths, seeds, next_seed = initialize_environment(
            vector, engine.initial_progress, engine.initial_environment)
        engine.initial_environment = None
        collector_rng = engine.initial_collector_rng
        rng.bit_generator.state = collector_rng["numpy"]
        vector.action_generator.set_state(collector_rng["torch"].cpu())
        engine.initial_collector_rng = collector_rng = None
        assert _digest(engine.replay.sample(c.masac.batch_size)) == expected_sample
        assert engine.critic.updates == updates_before_resume
        assert len(engine.replay) == 1_000_000
        event("restored", restore_s=time.perf_counter() - before, rng_next_sample_matches=True,
              updates=engine.critic.updates, replay_rows=len(engine.replay),
              environment_restored=True)
        values, collector_s = collect()
        before = time.perf_counter()
        engine.submit(values)
        engine.flush()
        assert engine.critic.updates == updates_before_resume + 16
        assert all(math.isfinite(float(v)) for v in engine.latest.values())
        assert all(bool(torch.isfinite(p).all()) for p in engine.critic.parameters())
        assert file_sha256(checkpoint) == checkpoint_hash
        event("resumed_learning", learner_s=time.perf_counter() - before,
              collector_s=collector_s, updates=engine.critic.updates,
              checkpoint_file_unchanged=True, metrics=engine.latest)
        # Diagnostic replay is duplicated data, not a production checkpoint.
        # Remove it after verified restore so Kaggle exports only small evidence.
        checkpoint.unlink()
        result.update(status="passed", real_collections=real_collections,
                      real_joint_transitions=real_collections * c.runtime.num_envs,
                      duplicated_replay=True, checkpoint_removed_after_verified_restore=True)
        event("complete")
        return result
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        (output / "cpu-replay-traceback.log").write_text(traceback.format_exc())
        event("failed")
        raise
    finally:
        stop.set()
        sampler.join(timeout=2)
        if engine is not None:
            engine.close()
        if vector is not None:
            vector.close()

def record_probe_summary(run, result):
    run.summary.update({"probe_passed": result["status"] == "passed",
                        "probe_duplicated_replay": True,
                        "episode_budget_complete": False})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/kaggle/working")
    parser.add_argument("--layout", choices=["cpu", "hybrid"], default="cpu")
    args = parser.parse_args()
    import wandb
    run = wandb.init(entity=os.environ["WANDB_ENTITY"], project=os.environ["WANDB_PROJECT"],
                     id=os.environ["WANDB_RUN_ID"],
                     name=("DIAG-HYBRID-REPLAY-1M-MASAC-PQL" if args.layout == "hybrid"
                           else "DIAG-CPU-REPLAY-1M-MASAC-PQL"),
                     job_type="memory-diagnostic", config={"probe_scope": "full_buffer_memory_execution",
                     "probe_layout": args.layout, "duplicated_replay": True, **probe_config().to_dict()})
    def report(row):
        metrics = {"probe/" + k: v for k, v in row.items() if type(v) in (int, float)}
        for gpu in row["gpu"]:
            metrics.update({f"probe/gpu{gpu['device']}/" + k: v for k, v in gpu.items()
                            if type(v) in (int, float)})
        run.log({**metrics, "probe/phase": row["phase"]})
    code = 1
    try:
        result = run_probe(args.output, report=report, layout=args.layout)
        record_probe_summary(run, result)
        code = 0
    finally:
        run.finish(exit_code=code)

if __name__ == "__main__":
    main()
