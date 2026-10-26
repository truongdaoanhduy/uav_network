"""Opt-in real2T4 proof with the unchanged primary scientific budgets."""
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from uav_network.configuration import load_config
from uav_network.environment.world import generate_scenario
from uav_network.experiments.training import train

ENABLED = (os.environ.get("UAV_CUDA_ROBUST_TRAINING") == "1" and
           torch.cuda.is_available() and torch.cuda.device_count() == 2)
pytestmark = pytest.mark.skipif(not ENABLED, reason="opt-in real2T4 robust production checks")


def release():
    gc.collect()
    for i in range(2):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()


def test_real_cuda_primary_seed_schedule():
    import warp as wp
    from uav_network.environment.gpu.vector import WarpVectorEnv
    c = load_config("configs/comparison-20k-hybrid75-1kb.yaml")
    seeds = list(range(44, 44 + 4096 + 20000))
    seeds += list(range(2000000044, 2000000074))
    seeds += list(range(100000, 150000))
    env = WarpVectorEnv(c, device="cuda:0")
    fallbacks = retries = checked = 0
    try:
        for start in range(0, len(seeds), 4096):
            selected = seeds[start:start + 4096]
            padded = selected + [44] * (4096 - len(selected))
            env.reset(seed=padded)
            assert not bool(wp.to_torch(env.s.errors).any())
            obstacles = wp.to_torch(env.s.obstacles)[:len(selected)].cpu().numpy()
            expanded = obstacles[:, :, 2] + c.world.obstacle_clearance_m
            assert np.all(obstacles[:, :, :2] > expanded[:, :, None])
            assert np.all(obstacles[:, :, :2] < c.world.map_size_m - expanded[:, :, None])
            delta = obstacles[:, :, None, :2] - obstacles[:, None, :, :2]
            distances = np.linalg.norm(delta, axis=-1)
            distances[:, np.arange(25), np.arange(25)] = np.inf
            assert np.all(distances > expanded[:, :, None] + expanded[:, None, :])
            fallbacks += int(wp.to_torch(env.s.generation_fallback)[:len(selected)].sum())
            retries += int((wp.to_torch(env.s.generation_attempts)[:len(selected)] > 1).sum())
            # Independent native comparison at both ends of each batch.
            for index in {0, len(selected) - 1}:
                scene = generate_scenario(c, np.random.default_rng(selected[index]))
                for field in ("positions", "targets", "obstacles", "gcs"):
                    np.testing.assert_allclose(
                        wp.to_torch(getattr(env.s, field))[index].cpu().numpy(),
                        getattr(scene, field), rtol=0, atol=2e-10)
            checked += len(selected)
        print(json.dumps({"seed_schedule_checked": checked, "fallbacks": fallbacks,
                          "retries": retries, "generation_version": 3}), flush=True)
        assert checked == 74126
    finally:
        env.close()
        release()


@pytest.mark.parametrize("pql", [False, True])
def test_fullgpu_masac_updates_and_exact_resume_keep_logical_budgets(pql):
    with TemporaryDirectory(prefix="robust-fullgpu-") as directory:
        c = load_config("configs/comparison-20k-hybrid75-1kb.yaml")
        c.runtime.algorithm = "masac"
        c.pql.enabled = pql
        c.runtime.replay_device = "cuda"
        c.runtime.replay_cpu_fraction = 0
        c.runtime.max_train_seconds = 90
        c.runtime.output_dir = str(Path(directory) / "first")
        c.tracking.enabled = False
        assert (c.runtime.num_envs, c.masac.batch_size, c.masac.replay_capacity,
                c.masac.learning_starts, c.masac.updates_per_step,
                c.runtime.episodes, c.world.max_steps) == (
                    4096, 8192, 1000000, 131072, 8, 20000, 1800)
        for i in range(2):
            torch.cuda.reset_peak_memory_stats(i)
        try:
            first = train(c)
            assert first["new_updates"] >= 8
            assert first["episode_budget_complete"] is False
            assert first["stop_reason"] == "time_budget"
            assert first["environment_state_saved"] is True
            peak = [{"gpu": i, "total": torch.cuda.get_device_properties(i).total_memory,
                     "allocated_peak": torch.cuda.max_memory_allocated(i),
                     "reserved_peak": torch.cuda.max_memory_reserved(i)} for i in range(2)]
            checkpoint_bytes = Path(first["resume_checkpoint"]).stat().st_size
            release()
            c.runtime.max_train_seconds = 1e-9
            c.runtime.output_dir = str(Path(directory) / "second")
            c.runtime.warm_start_checkpoint = first["resume_checkpoint"]
            resumed = train(c)
            assert resumed["state_restore"] == "training_state_exact_environment"
            assert resumed["optimizer_replay_restored"] is True
            assert resumed["initial_global_step"] == first["global_step"]
            assert resumed["global_step"] > first["global_step"]
            assert resumed["updates"] > first["updates"]
            print(json.dumps({"fullgpu_masac": "pql" if pql else "base",
                              "first_updates": first["updates"],
                              "resume_updates": resumed["updates"],
                              "first_step": first["global_step"],
                              "resume_step": resumed["global_step"],
                              "checkpoint_bytes": checkpoint_bytes, "peak": peak}), flush=True)
        finally:
            release()
