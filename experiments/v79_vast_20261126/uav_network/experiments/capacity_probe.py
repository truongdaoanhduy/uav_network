"""Exact-size radio/replay/optimizer capacity check, explicitly not full GPU RL."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import traceback

import numpy as np
import torch

from uav_network.algorithms.core.replay import TensorReplayBuffer
from uav_network.algorithms.matd3.learner import MATD3
from uav_network.algorithms.masac.learner import MASAC
from uav_network.experiments.capacity_fixture import real_transition_fixture, fill_repeated_fixture
from uav_network.experiments.radio_config import radio_config_manifest, scenario_protocol_config
from uav_network.experiments.resource_logging import ResourceJournal, write_json
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67


class CudaRequired(RuntimeError):
    pass


def classify_failure(error, phase):
    if isinstance(error, CudaRequired):
        return "BLOCKED_CUDA_REQUIRED"
    if isinstance(error, (MemoryError, torch.OutOfMemoryError)):
        return "OOM"
    if (phase == "warp_radio_allocate_reset" and isinstance(error, RuntimeError) and
            re.fullmatch(r"Failed to allocate [1-9][0-9]* bytes on device 'cuda:[0-9]+'", str(error))):
        return "OOM"
    return "FAIL"


def _provenance(scenario_path):
    from scripts.maintenance.source_inventory import source_paths
    root = Path(__file__).resolve().parents[2]
    tree = hashlib.sha256()
    for relative in source_paths(root):
        file = root / relative
        tree.update(relative.encode() + b"\0" + hashlib.sha256(file.read_bytes()).digest())
    revision, dirty = os.environ.get("UAV_SOURCE_COMMIT"), None
    if (root / ".git").exists():
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    return {"scenario_path": str(Path(scenario_path).resolve()),
            "scenario_file_sha256": hashlib.sha256(Path(scenario_path).read_bytes()).hexdigest(),
            "source_tree_sha256": tree.hexdigest(), "git_head": revision, "git_dirty": dirty,
            "source_bundle_sha256": os.environ.get("UAV_SOURCE_BUNDLE_SHA256")}


def _preflight(mode, devices):
    for name in devices:
        device = torch.device(name)
        if mode == "capacity" and device.type != "cuda":
            raise CudaRequired("Exact target capacity mode requires explicit CUDA devices; no CPU fallback")
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Only explicit cpu/cuda devices are supported")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise CudaRequired("CUDA is unavailable; requested sizes were not reduced")
            index = torch.cuda.current_device() if device.index is None else device.index
            if index >= torch.cuda.device_count():
                raise CudaRequired(f"Requested CUDA device {index} is unavailable")
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)


def _finite_metrics(metrics):
    values = {name: float(value) for name, value in metrics.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("Non-finite learner metrics")
    return values


def run_probe(config, *, scenario_path, output_dir, mode="capacity", algorithm="matd3",
              network_device="cuda:0", learner_device="cuda:1", seed=44,
              functional_envs=4, functional_batch=16, functional_hidden=64,
              scenario_overrides=(), warp_q_cap=32, allow_dynamic_mac_timing=False,
              require_queue_parity=False):
    config.validate()
    if mode not in ("functional", "capacity") or algorithm not in ("matd3", "masac"):
        raise ValueError("Expected functional/capacity mode and matd3/masac algorithm")
    if not (type(functional_envs) is int and 1 <= functional_envs <= 32 and
            type(functional_batch) is int and 1 <= functional_batch <= 256 and
            type(functional_hidden) is int and 16 <= functional_hidden <= 512):
        raise ValueError("Functional sizes must be envs 1..32, batch 1..256, hidden 16..512")
    output = Path(output_dir)
    journal = ResourceJournal(output, devices=[network_device, learner_device])
    effective = deepcopy(config)
    effective.runtime.algorithm = algorithm
    requested = {"num_envs": config.runtime.num_envs,
                 "batch_size": getattr(config, algorithm).batch_size}
    if mode == "functional":
        effective.runtime.num_envs = functional_envs
        hp = getattr(effective, algorithm)
        hp.batch_size = functional_batch
        hp.replay_capacity = max(32, functional_batch * 2)
        hp.hidden_sizes = [functional_hidden, functional_hidden]
    effective.validate()
    hp = getattr(effective, algorithm)
    result = {"status": "RUNNING", "mode": mode, "algorithm": algorithm, "seed": seed,
              "requested": requested,
              "effective": {"num_envs": effective.runtime.num_envs, "batch_size": hp.batch_size,
                            "replay_capacity": hp.replay_capacity, "hidden_sizes": hp.hidden_sizes},
              "placement": {"warp_radio": network_device, "learner": learner_device,
                            "replay": learner_device, "real_transition_fixture": "cpu"},
              "full_gpu_rl_training": False,
              "limitations": ["Warp subsystem is radio only; CPU flight/sensing/reward are not vectorized",
                              "Four sequential real CPU transitions repeated to commit replay memory",
                              "Two optimizer updates are a capacity check, not a trained policy",
                              "Easy radio fixture does not establish MAC timing or upstream parity",
                              "q_cap/mutable MAC use explicit opt-in and need GPU runtime validation"],
              "warp_packet_workload_executed": False, "updates": 0, "actor_updates": 0}
    write_json(output / "requested_config.json", config.to_dict())
    write_json(output / "effective_config.json", effective.to_dict())
    write_json(output / "result.json", result)
    try:
        with journal.phase("hardware_preflight"):
            _preflight(mode, [network_device, learner_device])
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        with journal.phase("scenario_and_real_cpu_fixture"):
            scenario = load_scenario(scenario_path, overrides=tuple(scenario_overrides))
            radio_config = radio_config_manifest(
                scenario, warp_q_cap=warp_q_cap,
                allow_dynamic_mac_timing=allow_dynamic_mac_timing,
                require_queue_parity=require_queue_parity,
            )
            result["radio_config"] = radio_config
            write_json(output / "radio_config.json", radio_config)
            write_json(output / "scenario_config.json", asdict(scenario))
            result["provenance"] = _provenance(scenario_path)
            write_json(output / "provenance.json", result["provenance"])
            if (scenario.num_uavs != effective.world.num_uavs or
                    scenario.comm_action_mode != "compact_3" or
                    scenario.enable_tx_power_control or
                    effective.network.destination_mode != "compact_3" or
                    effective.continuous_dim != 2 or effective.network.power_control != "fixed"):
                raise ValueError("This probe requires matching fleet, compact_3 and fixed-power 2D actions")
            fixture = real_transition_fixture(scenario, seed=seed)
            n, obs_dim = fixture["observation"].shape[1:]
            state_dim = fixture["state"].shape[-1]
            result["fixture"] = {"real_cpu_steps": len(fixture["reward"]),
                                 "repeated_for_capacity": True, "parallel_cpu_envs": 1}
            result["effective"].update(observation_shape=[n, obs_dim], state_dim=state_dim,
                                       grid_w=scenario.grid_w)
        envs = effective.runtime.num_envs
        with journal.phase("warp_radio_allocate_reset", num_envs=envs):
            runtime_network_kwargs = dict(radio_config["network_kwargs"])
            if allow_dynamic_mac_timing:
                # JSON manifest stores only primitives; this validated object must
                # reach SimDimsV67 on the *actual constructor path*.
                runtime_network_kwargs["protocol"] = scenario_protocol_config(scenario)
            net = WarpCompressedRadioNetworkV67(
                envs, num_uavs=n, grid_w=scenario.grid_w,
                **runtime_network_kwargs, device=network_device, seed=seed,
            )
            result["network_backing_bytes"] = net.active_soa_bytes_per_environment(
                n, scenario.grid_w, q_cap=radio_config["comparability"]["warp_queue_capacity"]) * envs
        with journal.phase("warp_packet_workload", num_envs=envs):
            from scripts.gpu.packet_workload import run_packet_workload, summarize_snapshot
            result["warp_packet_summary"] = summarize_snapshot(run_packet_workload(
                net, step_kwargs=radio_config["phy_step_kwargs"]))
            result["warp_packet_workload_executed"] = True
            result["warp_network_step_calls"] = 3
        with journal.phase("learner_allocate"):
            learner = (MATD3 if algorithm == "matd3" else MASAC)(
                effective, observation_dim=obs_dim, state_dim=state_dim, device=learner_device)
            before = [value.detach().cpu().clone() for value in learner.actors.parameters()]
        with journal.phase("replay_allocate", capacity=hp.replay_capacity):
            replay = TensorReplayBuffer(hp.replay_capacity, (n, obs_dim), state_dim,
                                        device=learner_device, continuous_dim=2, seed=seed)
            result["replay_allocated_bytes"] = replay.allocated_bytes
        with journal.phase("replay_fill_repeated_real_fixture", capacity=hp.replay_capacity):
            fill_repeated_fixture(replay, fixture, chunk_size=min(envs, replay.capacity))
            result["replay_size"] = len(replay)
        with journal.phase("actor_inference", batch_size=envs):
            indices = np.arange(envs) % len(fixture["reward"])
            observations = torch.as_tensor(fixture["observation"][indices], device=learner_device)
            action = learner.act_batch(observations, deterministic=True, return_tensors=True)
            if not torch.isfinite(action.continuous).all():
                raise FloatingPointError("Non-finite inference actions")
            result["inference_shapes"] = {"continuous": list(action.continuous.shape),
                                          "destination": list(action.destination.shape)}
            del observations, action
        result["update_metrics"] = []
        for index in range(1, 3):
            with journal.phase(f"replay_sample_{index}", batch_size=hp.batch_size):
                batch = replay.sample(hp.batch_size)
                result["sample_shapes"] = {name: list(value.shape) for name, value in batch.items()}
            with journal.phase(f"learner_update_{index}", batch_size=hp.batch_size):
                result["update_metrics"].append(_finite_metrics(learner.update(batch)))
                result["updates"], result["actor_updates"] = learner.updates, learner.actor_updates
            del batch
        with journal.phase("verify_actor_change"):
            delta = sum(float((value.detach().cpu() - old).square().sum())
                        for value, old in zip(learner.actors.parameters(), before)) ** .5
            if not math.isfinite(delta) or delta <= 0:
                raise FloatingPointError("No finite actor parameter change after delayed update")
            result["actor_parameter_delta_l2"] = delta
        result["status"] = ("FUNCTIONAL_PASS_NOT_FULL_RL" if mode == "functional"
                            else "CAPACITY_PROBE_PASS_NOT_FULL_RL")
    except Exception as error:
        result.update(failure_phase=journal.current_phase, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        result["status"] = classify_failure(error, journal.current_phase)
    write_json(output / "result.json", result)
    journal.record("probe_result", journal.current_phase, status=result["status"])
    return result
