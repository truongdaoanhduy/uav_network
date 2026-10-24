"""Bounded CUDA preflight; its transitions never enter production checkpoints."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import torch
from ..configuration import load_config


def _counter_total(counter):
    return sum(counter.values())

def check_execution(config, output):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Execution preflight requires a usable CUDA device")
    device_count = torch.cuda.device_count()
    from ..algorithms.checkpoints import make_learner
    from ..algorithms.optimization import compile_models
    from ..experiments.pql_replay import make_replay
    from ..environment.observations import observation_dim, state_dim
    from ..environment.vector import make_vector
    from ..experiments.pql_system import PQLSystem
    from ..experiments.pql_workers import inference_copy
    from ..experiments.training import seed_runtime, refresh_base_collector, resolve_base_devices
    from torch._dynamo.utils import counters
    from torch._inductor import metrics as inductor_metrics

    cfg = deepcopy(config)
    cfg.tracking.enabled = False
    cfg.runtime.warm_start_checkpoint = "none"
    production_warmup = cfg.pql.warmup_collection_steps
    cfg.pql.warmup_collection_steps = 3  # Diagnostic only; trainer is a fresh process.
    output = Path(output)
    started = time.monotonic()
    report = dict(status="running", scope="execution preflight; zero production episodes",
        algorithm=cfg.runtime.algorithm, pql=cfg.pql.enabled, num_envs=cfg.runtime.num_envs,
        batch_size=cfg.algorithm_config.batch_size, replay_capacity=cfg.algorithm_config.replay_capacity,
        diagnostic_pql_warmup=3, production_pql_warmup=production_warmup,
        amp_mode=cfg.runtime.amp_mode, compile=cfg.runtime.compile_models,
        fused=cfg.runtime.fused_optimizers)
    def persist():
        report["seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2, allow_nan=False))
    engine = vector = None
    graphs_before = counters["stats"]["unique_graphs"]
    graph_breaks_before = _counter_total(counters["graph_break"])
    frames_total_before = counters["frames"]["total"]
    frames_ok_before = counters["frames"]["ok"]
    persist()
    try:
        seed_runtime(cfg)
        torch.cuda.init()
        for i in range(device_count):
            torch.cuda.reset_peak_memory_stats(i)
        if cfg.pql.enabled:
            engine = PQLSystem(cfg)
            learner, policy, collector, replay = engine.critic, engine.policy, engine.collector, engine.replay
            collector_device = engine.collector_device
        else:
            collector_device, learner_device, _ = resolve_base_devices(cfg)
        vector = make_vector(cfg, device=collector_device)
        obs, _ = vector.reset(seed=cfg.runtime.seed)
        if not cfg.pql.enabled:
            learner = make_learner(cfg, observation_dim(cfg), state_dim(cfg), device=learner_device)
            policy = learner
            collector = inference_copy(learner, collector_device)
            replay = make_replay(cfg, (cfg.world.num_uavs, observation_dim(cfg)),
                                 state_dim(cfg), learner.device)
            if cfg.runtime.compile_models:
                compile_models(learner)
        optimizers = [learner.critic_optimizer, *policy.actor_optimizers]
        if cfg.runtime.algorithm == "masac":
            optimizers.append(policy.alpha_optimizer)
        report["all_adam_groups_fused"] = all(group.get("fused") is True
            for optimizer in optimizers for group in optimizer.param_groups)
        report["replay_bytes"] = replay.allocated_bytes
        report["replay_layout"] = "hybrid" if hasattr(replay, "cpu_data") else replay.device.type
        report["replay_cpu_bytes"] = getattr(replay, "cpu_allocated_bytes",
            replay.allocated_bytes if replay.device.type == "cpu" else 0)
        report["replay_hot_bytes"] = getattr(replay, "hot_allocated_bytes",
            replay.allocated_bytes if replay.device.type == "cuda" else 0)
        report["devices"] = dict(world=str(vector.device), collector=str(collector.device),
                                 learner=str(learner.device), replay=str(replay.device))
        collector.act_batch(obs["observation"], return_tensors=True)
        next_seed = cfg.runtime.seed + cfg.runtime.num_envs
        latest = {}
        for collection in range(6):
            action = (vector.random_actions(obs["observation"]) if collection < 2 else
                (engine.act_batch(obs["observation"], return_tensors=True) if engine else
                 collector.act_batch(obs["observation"], return_tensors=True)))
            following, reward, terminated, truncated, infos = vector.step(
                dict(continuous=action.continuous, destination=action.destination))
            values = (obs["observation"], obs["state"], action, reward,
                      following["observation"], following["state"], terminated)
            if engine:
                engine.submit(values)
            else:
                replay.add_batch(*values)
                if collection >= 2:
                    with torch.cuda.device(learner.device):
                        for _ in range(cfg.algorithm_config.updates_per_step):
                            latest = learner.update(replay.sample(cfg.algorithm_config.batch_size))
                    refresh_base_collector(collector, learner)
            obs = following
            indices = list(infos["device_episode_infos"])
            if indices:
                seeds = [None] * cfg.runtime.num_envs
                for index in indices:
                    seeds[index] = next_seed
                    next_seed += 1
                obs, _ = vector.reset(seed=seeds,
                    options={"reset_mask": (terminated | truncated).cpu().numpy()})
            report["collections"] = collection + 1
            persist()
        if engine:
            engine.flush()
            latest = dict(engine.latest)
        for i in range(device_count):
            torch.cuda.synchronize(i)
        report.update(critic_updates=learner.updates, actor_updates=policy.actor_updates,
            precision=dict(critic=learner.precision.state_dict(), policy=policy.precision.state_dict()),
            amp_autocast_fp16=latest.get("amp/autocast_fp16", 0),
            compiled_unique_graphs=counters["stats"]["unique_graphs"] - graphs_before,
            inductor_generated_kernels=inductor_metrics.generated_kernel_count,
            dynamo_graph_breaks=_counter_total(counters["graph_break"]) - graph_breaks_before,
            dynamo_frames_total=counters["frames"]["total"] - frames_total_before,
            dynamo_frames_ok=counters["frames"]["ok"] - frames_ok_before)
        report["dynamo_frames_fallback"] = report["dynamo_frames_total"] - report["dynamo_frames_ok"]
        report["memory"] = [dict(device=i, name=torch.cuda.get_device_name(i),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(i),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(i),
            total_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(device_count)]
        if not all(torch.isfinite(p).all() for owned in (learner, policy) for p in owned.parameters()):
            raise FloatingPointError("Non-finite production parameters")
        if cfg.runtime.fused_optimizers and not report["all_adam_groups_fused"]:
            raise RuntimeError("Fused Adam is not active")
        if cfg.runtime.compile_models and report["compiled_unique_graphs"] <= 0:
            raise RuntimeError("No actual compiled production graph executed")
        if engine and not all(t.is_alive() for t in engine.threads):
            raise RuntimeError("Both PQL workers must remain alive")
        if cfg.runtime.amp_mode == "fp16":
            groups = [(learner, "critic"), (policy, "actor")]
            if cfg.runtime.algorithm == "masac":
                groups.append((policy, "alpha"))
            if report["amp_autocast_fp16"] != 1 or any(
                    owned.precision.counts[group]["applied"] <= 0 for owned, group in groups):
                raise RuntimeError("AMP must execute and apply every owned optimizer group")
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if engine:
                engine.close()
            if vector:
                vector.close()
        finally:
            persist()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--algorithm", choices=("masac", "matd3"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, [*args.overrides, f"runtime.algorithm={args.algorithm}"])
    result = check_execution(config, args.output)
    print(json.dumps({key: result[key] for key in
        ("status", "algorithm", "pql", "critic_updates", "actor_updates", "compiled_unique_graphs")}), flush=True)

if __name__ == "__main__":
    main()
