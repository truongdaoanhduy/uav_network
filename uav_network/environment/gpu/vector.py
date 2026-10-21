"""Warp worlds sharing Torch storage; no simulator process or per-step array transport."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from .state import allocate_state, params_from_config, seed_rows
from .motion import motion_step
from .world import reset_geometry
from .validation import validate_actions
from .geometry import sense
from .observations import observe, centralized
from ...communication.gpu.mac import (reset_network, begin_step, sense_reports, masked_sense_reports,
                                      packet_step, packet_step_gen2, end_step)
from ...communication.gpu.metrics import world_info
from ..env import Action
from ..observations import observation_dim, state_dim


class WarpVectorEnv:
    num_cpu_workers = 0
    environment_backend = "warp_packet"

    def __init__(self, config, device=None, packet_scheduler="gen1"):
        if packet_scheduler not in {"gen1", "gen2"}:
            raise ValueError(f"Unknown packet scheduler: {packet_scheduler}")
        self.packet_scheduler = packet_scheduler
        self._packet_step_kernel = packet_step if packet_scheduler == "gen1" else packet_step_gen2
        self.config = config.validate()
        self.num_envs = config.runtime.num_envs
        selected = config.runtime.device if device is None else device
        if selected == "auto":
            selected = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device("cuda:0" if selected == "cuda" else selected)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Warp environment requires a cpu or cuda device")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA environment requested without a CUDA device")
        self.wp_device = str(self.device)
        wp.init()
        self._torch_stream = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        self.stream = wp.stream_from_torch(self._torch_stream) if self._torch_stream is not None else None
        scope = wp.ScopedStream(self.stream) if self.stream is not None else nullcontext()
        with scope:
            self.s, self.p = allocate_state(config, self.num_envs, device=self.wp_device)
            self._buffers = [
                {"observation":wp.to_torch(self.s.observation),"state":wp.to_torch(self.s.state)},
                {"observation":torch.empty((self.num_envs,config.world.num_uavs,observation_dim(config)),
                                           device=self.device,dtype=torch.float32),
                 "state":torch.empty((self.num_envs,state_dim(config)),device=self.device,dtype=torch.float32)}]
        self._buffer = 0
        self._seeds = np.arange(config.runtime.seed, config.runtime.seed+self.num_envs,dtype=np.int64)
        self._waiting = self._closed = False
        self._ready = False
        self._keepalive = []
        self.action_generator = torch.Generator(device=self.device).manual_seed(config.runtime.seed+1)

    @property
    def allocated_bytes(self):
        arrays = [value for value in vars(self.s).values() if isinstance(value,wp.array)]
        unique = {int(a.ptr):a for a in arrays if a.size}
        total = sum(a.size * wp.types.type_size_in_bytes(a.dtype) for a in unique.values())
        # Buffer1 is Torch-owned; buffer0 is already counted through the state.
        return total + sum(t.numel()*t.element_size() for t in self._buffers[1].values())

    def _bind_stream(self):
        if self.device.type == "cuda":
            current = torch.cuda.current_stream(self.device)
            if current != self._torch_stream:
                current.wait_stream(self._torch_stream)
                self._torch_stream = current
                self.stream = wp.stream_from_torch(current)

    def _launch(self, kernel, dim, *args):
        wp.launch(kernel,dim=dim,inputs=list(args),device=self.wp_device,stream=self.stream)

    def _check_errors(self):
        error = wp.to_torch(self.s.errors)
        if bool(error.any()):
            bad = [(int(i),int(error[i])) for i in torch.nonzero(error).flatten().cpu().tolist()]
            raise RuntimeError(f"Warp world capacity/configuration error: {bad[:8]}")

    def _publish(self, buffer):
        self.s.observation = wp.from_torch(buffer["observation"])
        self.s.state = wp.from_torch(buffer["state"])
        self._launch(observe,(self.num_envs,self.p.n),self.s,self.p)
        self._launch(centralized,self.num_envs,self.s,self.p)
        self._check_errors()
        return buffer

    def reset(self, *, seed=None, options=None):
        if self._closed or self._waiting:
            raise RuntimeError("Cannot reset a closed/waiting device vector")
        self._bind_stream()
        mask = np.ones(self.num_envs,dtype=bool) if options is None else np.asarray(options.get("reset_mask"),dtype=bool)
        if mask.shape != (self.num_envs,):
            raise ValueError("reset_mask must have one flag per world")
        if seed is None:
            supplied = self._seeds.tolist()
        elif isinstance(seed,(int,np.integer)):
            supplied = list(range(int(seed),int(seed)+self.num_envs))
        else:
            supplied = list(seed)
            if len(supplied) != self.num_envs:
                raise ValueError("seed must contain one entry per world")
        for i in np.flatnonzero(mask):
            if supplied[i] is not None:
                self._seeds[i] = int(supplied[i])
        selected = np.flatnonzero(mask)
        if not self._ready and len(selected) != self.num_envs:
            raise ValueError("The first reset must initialize all worlds")
        if not len(selected):
            return self._buffers[self._buffer], {"environment_backend":"warp_packet","num_cpu_env_workers":0}
        indices = torch.as_tensor(selected,device=self.device,dtype=torch.int64)
        wp.to_torch(self.s.seeds).index_copy_(0,indices,torch.as_tensor(self._seeds[selected],device=self.device))
        rows = seed_rows(self._seeds[selected]).view(np.int64)
        wp.to_torch(self.s.pcg_state).view(torch.int64).index_copy_(0,indices,torch.as_tensor(rows,device=self.device))
        mask_tensor = torch.as_tensor(mask,dtype=torch.int32,device=self.device)
        wp_mask = wp.from_torch(mask_tensor)
        self._launch(reset_network,self.num_envs,self.s,self.p,wp_mask)
        self._launch(reset_geometry,self.num_envs,self.s,self.p,wp_mask)
        self._check_errors()
        self._launch(sense,(self.num_envs,self.p.n,self.p.targets_count),self.s,self.p)
        self._launch(masked_sense_reports,self.num_envs,self.s,self.p,wp_mask)
        observation = self._publish(self._buffers[self._buffer])
        self._ready = True
        return observation, {"environment_backend":"warp_packet","num_cpu_env_workers":0}

    def _snapshot_config(self):
        config = self.config.to_dict()
        return {key: deepcopy(config[key]) for key in
                ("world", "sensor", "network", "energy", "reward")} | {
                    "num_envs": self.num_envs, "backend": self.environment_backend}

    def snapshot_state(self):
        """Copy every simulator array at a completed collection boundary."""
        if self._closed or self._waiting or not self._ready:
            raise RuntimeError("Cannot snapshot a closed, waiting, or uninitialized vector")
        self._bind_stream()
        wp.synchronize_device(self.wp_device)
        return {"version": 1, "config": self._snapshot_config(),
                "device_type": self.device.type,
                "arrays": {name: wp.to_torch(value).detach().to(device="cpu", copy=True)
                           for name, value in vars(self.s).items() if isinstance(value, wp.array)},
                "seeds": torch.as_tensor(self._seeds.copy()),
                "action_generator": self.action_generator.get_state().cpu().clone()}

    def restore_state(self, snapshot):
        """Validate the complete payload before changing any live world."""
        if self._closed or self._waiting:
            raise RuntimeError("Cannot restore a closed or waiting vector")
        if (not isinstance(snapshot, dict) or snapshot.get("version") != 1
                or snapshot.get("config") != self._snapshot_config()
                or snapshot.get("device_type") != self.device.type):
            raise ValueError("Incompatible Warp snapshot version/config/device")
        arrays = {name: wp.to_torch(value) for name, value in vars(self.s).items()
                  if isinstance(value, wp.array)}
        incoming = snapshot.get("arrays")
        if not isinstance(incoming, dict) or incoming.keys() != arrays.keys():
            raise ValueError("Warp snapshot array fields mismatch")
        for name, target in arrays.items():
            value = incoming[name]
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.shape != target.shape or value.dtype != target.dtype
                    or value.layout != torch.strided):
                raise ValueError(f"Warp snapshot array shape/dtype mismatch: {name}")
        seeds = snapshot.get("seeds")
        if (not isinstance(seeds, torch.Tensor) or seeds.shape != (self.num_envs,)
                or seeds.dtype != torch.int64 or seeds.device.type != "cpu"
                or not torch.equal(seeds, incoming["seeds"])):
            raise ValueError("Warp snapshot active seeds mismatch")
        # Validate generator state on a disposable generator, before any copies.
        generator = torch.Generator(device=self.device)
        try:
            generator.set_state(snapshot["action_generator"])
        except (KeyError, TypeError, RuntimeError) as error:
            raise ValueError("Invalid Warp snapshot action RNG") from error
        self._bind_stream()
        wp.synchronize_device(self.wp_device)
        for name, target in arrays.items():
            target.copy_(incoming[name])
        self._seeds = seeds.numpy().copy()
        self.action_generator.set_state(generator.get_state())
        self._ready = True
        # The current buffer already owns s.observation/state. No publish kernels
        # run here: even observation arrays are restored byte for byte.
        wp.synchronize_device(self.wp_device)
        return self._buffers[self._buffer]

    def random_actions(self, observations):
        shape = observations.shape[:-1]
        continuous = torch.rand((*shape,3),device=self.device,generator=self.action_generator)*2-1
        scores = torch.rand((*shape,self.p.n+1),device=self.device,generator=self.action_generator)
        mask = observations[..., -(self.p.n+1):] > .5
        mask[...,0] = True
        scores.masked_fill_(~mask,-1)
        destination = scores.argmax(-1)
        active = observations[...,9] > .5
        continuous[..., :2] *= active[...,None]
        continuous[..., 2] *= active & (destination != 0)
        return Action(continuous,destination)

    def step_async(self, actions):
        if self._closed or self._waiting or not self._ready:
            raise RuntimeError("Device vector is closed, waiting, or not reset")
        self._bind_stream()
        continuous = torch.as_tensor(actions["continuous"],device=self.device,dtype=torch.float32).contiguous()
        destination = torch.as_tensor(actions["destination"],device=self.device)
        integer_types = (torch.int8,torch.int16,torch.int32,torch.int64,
                         torch.uint8,torch.uint16,torch.uint32,torch.uint64)
        if destination.dtype not in integer_types:
            raise ValueError("Destination actions require an integer dtype")
        destination = destination.to(dtype=torch.int64).contiguous()
        if continuous.shape != (self.num_envs,self.p.n,3) or destination.shape != (self.num_envs,self.p.n):
            raise ValueError("Invalid batched action shape")
        # Native policy validation is performed by checked kernels; no host action copy.
        wc, wd = wp.from_torch(continuous), wp.from_torch(destination)
        self._keepalive = [continuous,destination,wc,wd]
        self._launch(validate_actions,self.num_envs,self.s,self.p,wc,wd)
        self._check_errors()
        self._buffer = 1-self._buffer
        next_buffer = self._buffers[self._buffer]
        self.s.observation = wp.from_torch(next_buffer["observation"])
        self.s.state = wp.from_torch(next_buffer["state"])
        self._launch(begin_step,self.num_envs,self.s,self.p)
        self._launch(motion_step,self.num_envs,self.s,self.p,wc)
        self._launch(sense,(self.num_envs,self.p.n,self.p.targets_count),self.s,self.p)
        self._launch(sense_reports,self.num_envs,self.s,self.p)
        self._launch(self._packet_step_kernel,self.num_envs,self.s,self.p,wc,wd)
        self._launch(end_step,self.num_envs,self.s,self.p)
        self._launch(observe,(self.num_envs,self.p.n),self.s,self.p)
        self._launch(centralized,self.num_envs,self.s,self.p)
        self._waiting = True

    def step_wait(self):
        if not self._waiting:
            raise RuntimeError("No device step is waiting")
        self._bind_stream()
        self._check_errors()
        self._waiting = False
        done = wp.to_torch(self.s.terminated).bool()
        infos = {"device_episode_infos":{}}
        if bool(done.any()):
            indices = torch.nonzero(done).flatten().cpu().tolist()
            infos["device_episode_infos"] = {int(i):world_info(self.s,self.p,int(i)) for i in indices}
        self._keepalive.clear()
        return (self._buffers[self._buffer],wp.to_torch(self.s.reward),done,
                torch.zeros_like(done),infos)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        if not self._closed:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._keepalive.clear()
            self._closed = True


class WarpMissionEnv:
    """Single CPU world for checkpoint evaluation/rendering of the same backend."""
    def __init__(self, config):
        self.config = config
        selected = deepcopy(config)
        selected.runtime.num_envs = 1
        self.vector = WarpVectorEnv(selected,device="cpu")
        self.reports = SimpleNamespace()
        self._communication_trace = False
        self._communication_events = []

    def enable_communication_trace(self):
        """Opt in on this CPU visualization world, never on the training vector."""
        if self._communication_trace:
            return
        s, p = self.vector.s, self.vector.p
        capacity = params_from_config(self.config).trace_capacity
        for name, dtype in (("trace_time", wp.float64), ("trace_source", wp.int32),
                            ("trace_destination", wp.int32), ("trace_kind", wp.int32)):
            setattr(s, name, wp.zeros((1,capacity), dtype=dtype, device="cpu"))
        s.trace_report = wp.zeros((1,capacity,9), dtype=wp.float64, device="cpu")
        s.trace_count.zero_()
        p.trace_capacity, p.trace_enabled = capacity, 1
        self._communication_trace = True

    def drain_communication_events(self):
        events, self._communication_events = self._communication_events, []
        return events

    def _collect_communication_events(self):
        s = self.vector.s
        count = int(s.trace_count.numpy()[0])
        kinds = s.trace_kind.numpy()[0,:count]
        # Read every mission step so the next begin_step cannot discard hops
        # between rendered frames. Training never enters this host-only path.
        reports = s.trace_report.numpy()[0,:count]
        times = s.trace_time.numpy()[0,:count]
        sources = s.trace_source.numpy()[0,:count]
        destinations = s.trace_destination.numpy()[0,:count]
        for i in np.flatnonzero(kinds == 2):
            report = reports[i]
            self._communication_events.append({
                "event": "report_received", "time_s": float(times[i]/1e6),
                "source": int(sources[i]), "destination": int(destinations[i]),
                "target": int(report[0]), "origin": int(report[1]),
                "created_s": float(report[2]), "hops": int(report[3]),
                "power_w": float(report[4]), "power_scope": "last_fragment",
                "source_xy": report[5:7].tolist(), "destination_xy": report[7:9].tolist()})

    def _refresh(self):
        s = self.vector.s
        self.positions = s.positions.numpy()[0]
        self.velocities = s.velocities.numpy()[0]
        self.targets = s.targets.numpy()[0]
        self.obstacles = s.obstacles.numpy()[0,:self.config.world.num_obstacles]
        self.gcs = s.gcs.numpy()[0]
        self.battery_j = s.battery.numpy()[0]
        self.battery_fraction = self.battery_j/self.config.energy.battery_capacity_j
        self.steps = int(s.steps.numpy()[0])
        self.reports.known = s.known.numpy()[0].astype(bool)
        self.reports.delivered = s.delivered.numpy()[0].astype(bool)

    def reset(self, seed=None, options=None):
        if options is not None:
            raise ValueError("CPU Warp visualization resets from an episode seed")
        self._communication_events = []
        obs,_ = self.vector.reset(seed=seed)
        self._refresh()
        return obs["observation"][0].numpy(),self.info()

    def step(self, action):
        result = self.vector.step({"continuous":np.asarray(action.continuous)[None],
                                   "destination":np.asarray(action.destination)[None]})
        self._refresh()
        if self._communication_trace:
            self._collect_communication_events()
        obs,reward,done,truncated,_ = result
        return obs["observation"][0].numpy(),float(reward[0]),bool(done[0]),bool(truncated[0]),self.info()

    def info(self):
        return world_info(self.vector.s,self.vector.p,0)

    def close(self):
        self.vector.close()
