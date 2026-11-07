"""Audit observations without GPU training, changing runtime state only."""
from copy import deepcopy
from pathlib import Path
import datetime
import json
import numpy as np
import torch
import warp as wp
from uav_network.configuration import load_config
from uav_network.environment.env import SwarmSearchEnv
from uav_network.environment.observations import local_observations, centralized_state, observation_dim, state_dim
from uav_network.environment.gpu.vector import WarpVectorEnv
from uav_network.environment.gpu.observations import observe, centralized
from uav_network.environment.bayesian import sensor_profile
from uav_network.algorithms.masac import MASAC
from uav_network.algorithms.matd3 import MATD3

torch.set_num_threads(1)
torch.manual_seed(44)
profile = "configs/comparison-independent-35t25o-h512-xyz-sensing.yaml"
c = load_config(profile, ["runtime.device=cpu", "runtime.num_envs=1"])
models = [cls(c, observation_dim(c), state_dim(c)) for cls in (MASAC, MATD3)]
dimensions = []
for targets, obstacles in ((20, 15), (20, 25), (35, 15), (35, 25)):
    d = deepcopy(c)
    d.world.num_targets = targets
    d.world.num_obstacles = obstacles
    dimensions.append(dict(targets=targets, obstacles=obstacles,
                           actor_input=observation_dim(d), state=state_dim(d),
                           critic_input=state_dim(d) + 66))
rows = []
for backend in ("native", "warp_cpu"):
    env = SwarmSearchEnv(c) if backend == "native" else WarpVectorEnv(c, device="cpu")
    def observations():
        if backend == "native":
            return local_observations(env).copy(), centralized_state(env).copy()
        wp.launch(observe, dim=(1, 6), inputs=[env.s, env.p], device="cpu")
        wp.launch(centralized, dim=1, inputs=[env.s, env.p], device="cpu")
        wp.synchronize()
        return env.s.observation.numpy()[0].copy(), env.s.state.numpy()[0].copy()
    try:
        for case in ("hidden_target", "hidden_obstacle", "other_agent_known",
                     "unannounced_delivery", "own_known_positive_control",
                     "visible_obstacle_positive_control"):
            env.reset(seed=44)
            before_obs, before_state = observations()
            if backend == "native":
                targets = env.targets.copy()
                obstacles = env.obstacles.copy()
                positions = env.positions[:, :2].copy()
            else:
                targets = env.s.targets.numpy()[0].copy()
                obstacles = env.s.obstacles.numpy()[0].copy()
                positions = env.s.positions.numpy()[0].copy()
            if case == "hidden_target":
                # Target0 is not in any initial UAV's camera footprint.
                assert np.min(np.linalg.norm(positions - targets[0], axis=1)) > 150
                targets[0, 0] += .25
                if backend == "native":
                    env.targets[:] = targets
                else:
                    env.s.targets.assign(targets[None])
            elif case in ("hidden_obstacle", "visible_obstacle_positive_control"):
                distances = np.linalg.norm(obstacles[:, None, :2] - positions[None], axis=2)
                eligible = np.flatnonzero((distances > 250 + obstacles[:, 2, None]).all(axis=1)
                                         & (obstacles[:, 1] > 350))
                assert len(eligible)
                index = int(eligible[0])
                if case == "hidden_obstacle":
                    obstacles[index, 0] += .25
                else:
                    obstacles[index, :2] = positions[0] + [30, 0]
                    obstacles[index, 2] = 5
                if backend == "native":
                    env.obstacles[:] = obstacles
                else:
                    env.s.obstacles.assign(obstacles[None])
            elif case in ("other_agent_known", "own_known_positive_control"):
                observer = 1 if case == "other_agent_known" else 0
                if backend == "native":
                    assert not env.reports.known[observer, 0]
                    env.reports.known[observer, 0] = True
                else:
                    known = env.s.known.numpy()
                    assert not known[0, observer, 0]
                    known[0, observer, 0] = 1
                    env.s.known.assign(known)
            else:
                if backend == "native":
                    env.reports.delivered[0] = True
                else:
                    delivered = env.s.delivered.numpy()
                    delivered[0, 0] = 1
                    env.s.delivered.assign(delivered)
            after_obs, after_state = observations()
            unchanged = np.array_equal(before_obs[0], after_obs[0])
            critic_changed = not np.array_equal(before_state, after_state)
            positive = case.endswith("positive_control")
            assert unchanged != positive, (backend, case)
            assert critic_changed, (backend, case)
            if case in ("hidden_target", "hidden_obstacle"):
                np.testing.assert_array_equal(before_obs, after_obs)
            policies_unchanged = {}
            if not positive:
                for model in models:
                    first = model.act_batch(before_obs[None], deterministic=True, return_tensors=True)
                    second = model.act_batch(after_obs[None], deterministic=True, return_tensors=True)
                    same = (torch.equal(first.continuous[:, 0], second.continuous[:, 0])
                            and torch.equal(first.destination[:, 0], second.destination[:, 0]))
                    assert same
                    policies_unchanged[model.algorithm] = same
            rows.append(dict(backend=backend, case=case, observer_actor_input_unchanged=unchanged,
                             critic_state_changed=critic_changed,
                             policies_unchanged=policies_unchanged))
    finally:
        env.close()
result = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              profile=profile, dimensions=dimensions, probes=rows,
              probe_count=len(rows), passed=True, source_modified=False,
              kaggle_jobs_submitted=0,
              scope="Current observation builder and deterministic raw actor outputs; not a proof for every possible rollout",
              safety_caveat="CBF is a separate privileged geometry filter; it is not fed as actor state")
Path("outputs/decpomdp-audit-20261107").mkdir(parents=True, exist_ok=True)
Path("outputs/decpomdp-audit-20261107/observation-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
