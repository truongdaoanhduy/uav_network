"""Warp worlds sharing Torch storage; no simulator process or per-step array transport."""
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from .state import allocate_state, seed_rows
from .motion import motion_step
from .world import reset_geometry
from .validation import validate_actions
from .geometry import sense
from .observations import observe, centralized
from ...communication.gpu.mac import reset_network, begin_step, sense_reports, masked_sense_reports, packet_step, end_step
from ...communication.gpu.metrics import world_info
from ..env import Action
from ..observations import observation_dim, state_dim


class WarpVectorEnv:
    num_cpu_workers = 0
    environment_backend = "warp_packet"

    def __init__(self, config, device=None):
        self.config = config.validate()
        self.num_envs = config.runtime.num_envs
        selected = config.runtime.device if device is None else device
        if selected == "auto":
            selected = "cuda" if torch.cuda.is_available() else "cpu"
        if selected == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA environment requested without a CUDA device")
        self.device = torch.device("cuda:0" if selected == "cuda" else "cpu")
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
        unique = {int(a.ptr):a for a in arrays}
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
        self._launch(packet_step,self.num_envs,self.s,self.p,wc,wd)
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
        obs,_ = self.vector.reset(seed=seed)
        self._refresh()
        return obs["observation"][0].numpy(),self.info()

    def step(self, action):
        result = self.vector.step({"continuous":np.asarray(action.continuous)[None],
                                   "destination":np.asarray(action.destination)[None]})
        self._refresh()
        obs,reward,done,truncated,_ = result
        return obs["observation"][0].numpy(),float(reward[0]),bool(done[0]),bool(truncated[0]),self.info()

    def info(self):
        return world_info(self.vector.s,self.vector.p,0)

    def close(self):
        self.vector.close()
