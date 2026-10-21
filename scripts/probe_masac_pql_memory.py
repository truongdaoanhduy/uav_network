"""Bounded actual async MASAC PQL probe; scientific algorithms remain unchanged.

Reuses the BASE probe's AMP arithmetic, compiler and physical GPU sampler.
One critic thread owns the critic scaler; one policy thread owns actor/alpha
scalers. The single process-wide helper patch is installed before submission
and removed only after both workers have stopped.
"""
from contextlib import ExitStack, contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
from threading import Lock, current_thread
import time
import traceback
from types import SimpleNamespace

import torch

from scripts import probe_masac_memory as base


class CompileStartupGate:
    """Prevent process-global FX tracing from overlapping initial worker updates."""
    def __init__(self):
        self.lock, self.warmed = Lock(), set()

    @contextmanager
    def enter(self, group):
        if len(self.warmed) < 2:
            with self.lock:
                yield True
                self.warmed.add(group)
        else:
            yield False


class PQLProbeExecution:
    """Apply autocast inside each real worker, with disjoint scaler ownership."""
    def __init__(self, engine, compiler_evidence, compile_enabled=True, device_type="cuda"):
        self.engine, self.device_type = engine, device_type
        self.dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
        self.amp = base.ProbeAMP(engine.critic, device_type=device_type)
        actors = [*engine.critic.actors, *engine.policy.actors]
        self.compiler = base.ProbeCompiler(SimpleNamespace(actors=actors),
                                          compiler_evidence, enabled=compile_enabled)
        self.rows, self.restores = [], []
        self.lock, self.stack = Lock(), ExitStack()
        self.startup_gate = CompileStartupGate()

    def _wrap(self, learner, method, group):
        original = getattr(learner, method)
        self.restores.append((learner, method, original))
        def update(batch, collect_metrics=True):
            started = time.perf_counter()
            guard = self.startup_gate.enter(group) if self.compiler.enabled else nullcontext(False)
            with guard as serialized:
                entered = time.perf_counter()
                events = None
                if self.device_type == "cuda":
                    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    events[0].record()
                # Each stage reuses actors under no_grad and with gradients;
                # do not cache detached casts across those two forwards.
                with torch.autocast(self.device_type, dtype=self.dtype, cache_enabled=False):
                    enabled = torch.is_autocast_enabled(self.device_type)
                    metrics = original(batch, collect_metrics)
                if events:
                    events[1].record()
            row = {"group": group, "thread": current_thread().name,
                   "autocast_enabled": enabled, "autocast_dtype": str(self.dtype),
                   "compile_startup_serialized": serialized,
                   "compile_startup_wait_seconds": entered - started,
                   "host_seconds": time.perf_counter() - started,
                   "completed_at": time.perf_counter(), "metrics": base.json_safe(metrics),
                   "_events": events}
            with self.lock:
                self.rows.append(row)
            return metrics
        setattr(learner, method, update)

    def __enter__(self):
        try:
            self.stack.enter_context(self.compiler)
            self.stack.enter_context(self.amp)
            self._wrap(self.engine.critic, "update_critic", "critic")
            self._wrap(self.engine.policy, "update_actor", "policy")
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        # Worker shutdown precedes restoring globals, including after worker errors.
        try:
            self.engine.close()
        finally:
            if all(not thread.is_alive() for thread in self.engine.threads):
                for learner, method, original in reversed(self.restores):
                    setattr(learner, method, original)
                self.stack.close()
                self.compiler.evidence["startup_warmed_worker_groups"] = sorted(self.startup_gate.warmed)
                self.compiler.evidence["startup_serialized_updates"] = sum(
                    row["compile_startup_serialized"] for row in self.rows)

    def evidence(self):
        evidence = self.amp.evidence()
        evidence["scope"] = "async PQL probe; independent critic and policy scaler owners"
        return evidence

    def timings(self):
        rows = []
        for row in self.rows:
            clean = {key: value for key, value in row.items() if key != "_events"}
            if row["_events"]:
                clean["cuda_stream_seconds"] = row["_events"][0].elapsed_time(row["_events"][1]) / 1000
            rows.append(clean)
        return rows


def parser():
    result = base.parser()
    result.description = __doc__
    return result


def resolve_pql_config(args):
    if args.capacity != 1_000_000:
        raise ValueError("Only the requested one-million replay PQL case is authorized")
    config, override = base.resolve_config(args)
    override["original_pql_warmup_collections"] = config.pql.warmup_collection_steps
    # BASE collected twice, then did four collect+8-update rounds: 6 collections,
    # 32 updates. PQL eligibility includes the warmup threshold collection.
    config.pql.enabled, config.pql.mode = True, "async"
    config.pql.warmup_collection_steps = 3
    config.pql.critic_sample_ratio = args.updates_per_round
    config.pql.collector_device = "cuda:0"
    config.pql.learner_device = config.pql.policy_device = "cuda:1"
    config.pql.worker_timeout_s = 600
    config.runtime.log_interval_steps = args.envs
    override.update(diagnostic_pql_warmup_collections=3,
                    random_action_collections=2, expected_updates=args.rounds * args.updates_per_round)
    return config.validate(), override


def run_probe(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    run = sampler = vector = engine = execution = None
    phase = "hardware_guard"
    report = {"schema_version": 1, "status": "running", "phase": phase,
              "scope": "bounded async PQL allocation and real-update diagnostic; not convergence",
              "requested": vars(args), "stages": [], "updates": [], "collection_steps": 0,
              "software": {"torch":torch.__version__,"torch_cuda":torch.version.cuda},
              "devices": {"worlds": "cuda:0", "collector": "cuda:0", "critic": "cuda:1",
                          "policy": "cuda:1", "replay": "cuda:1"},
              "source_git_commit": os.environ.get("UAV_GIT_COMMIT"),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "base_probe_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
              "compiler": {"probe_compile_requested": args.compile,
                           "production_inner_compile_models": False,
                           "strategy": "outer vectorized critic plus both worker actor encoders"},
              "full_training_verified": False}

    def persist():
        report["wall_seconds"] = time.perf_counter() - started
        temporary = output / "result.json.tmp"
        temporary.write_text(json.dumps(base.json_safe(report), indent=2, allow_nan=False))
        temporary.replace(output / "result.json")

    def snapshot(name, synchronize=True):
        nonlocal phase
        phase = report["phase"] = name
        memory = []
        if report.get("hardware"):
            for index in range(2):
                if synchronize:
                    torch.cuda.synchronize(index)
                memory.append({"device": index, "allocated_bytes": torch.cuda.memory_allocated(index),
                               "reserved_bytes": torch.cuda.memory_reserved(index),
                               "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
                               "peak_reserved_bytes": torch.cuda.max_memory_reserved(index)})
        stage = {"phase": name, "wall_seconds": time.perf_counter()-started, "torch": memory}
        if sampler:
            sampler._sample()
            stage["physical_samples"] = sampler.samples[-2:]
        report["stages"].append(stage)
        persist()
        if run:
            run.log({"probe/phase": name, "probe/wall_seconds": stage["wall_seconds"],
                     **{f"memory/gpu{row['device']}/{key}": value
                        for row in memory for key,value in row.items() if key != "device"}})

    try:
        report["hardware"] = base.require_two_t4()
        torch.cuda.init()
        for index in range(2):
            torch.cuda.reset_peak_memory_stats(index)
        from uav_network.experiments.gpu_resources import GPUResourceSampler
        sampler = GPUResourceSampler()
        sampler.start()
        snapshot("tracking_init")
        if not os.environ.get("WANDB_API_KEY","").strip():
            raise RuntimeError("Online W&B credential required before large allocations")
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                         name=args.wandb_name or "MASAC-PQL-E2048-B4096-R1M-FP16-COMPILE-FUSED",
                         id=args.wandb_run_id, resume="never", mode="online", dir=str(output),
                         job_type="memory-probe", tags=["diagnostic","masac","pql","dual-t4","fp16"],
                         config={"probe": vars(args)})
        if run is None or run.settings.mode != "online":
            raise RuntimeError("W&B online run was not initialized")
        report["wandb"] = {"id":run.id,"url":run.url,"mode":run.settings.mode}
        from uav_network.environment.observations import observation_dim, state_dim
        from uav_network.environment.vector import make_vector
        from uav_network.experiments.pql_system import PQLSystem
        from uav_network.experiments.training import seed_runtime
        config, report["warmup_override"] = resolve_pql_config(args)
        report["replay_expected_bytes"] = base.replay_bytes(
            args.capacity, (config.world.num_uavs,observation_dim(config)),state_dim(config)) + 9*args.capacity
        report["resolved_config"] = config.to_dict()
        (output / "resolved_config.json").write_text(json.dumps(report["resolved_config"],indent=2))
        run.config.update({"resolved":report["resolved_config"],"warmup_override":report["warmup_override"]})
        if not report["source_git_commit"]:
            import subprocess
            report["source_git_commit"] = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
        seed_runtime(config)
        snapshot("world_allocation")
        vector = make_vector(config,device="cuda:0")
        snapshot("world_reset")
        obs, _ = vector.reset(seed=config.runtime.seed)
        report["environment"] = {"allocated_bytes":vector.allocated_bytes,
                                 "device":str(vector.device),"backend":vector.environment_backend,
                                 "num_cpu_workers":vector.num_cpu_workers}
        snapshot("pql_models_and_replay_allocation")
        engine = PQLSystem(config)
        report["pql_workers"] = [{"name":thread.name,"alive":thread.is_alive()} for thread in engine.threads]
        report["replay"] = {"capacity":engine.replay.capacity,"allocated_bytes":engine.replay.allocated_bytes,
                            "storage":{key:{"dtype":str(value.dtype),"device":str(value.device),
                                            "bytes":value.numel()*value.element_size()}
                                       for key,value in engine.replay.data.items()},
                            "pql_metadata_bytes":9*args.capacity}
        report["parameters"] = {"dtypes":sorted({str(p.dtype) for learner in (engine.critic,engine.policy)
                                                 for p in learner.parameters()})}
        optimizers = [engine.critic.critic_optimizer,*engine.policy.actor_optimizers,
                      engine.policy.alpha_optimizer]
        report["fused"] = {"requested":args.fused,
                           "all_param_groups_fused":all(group.get("fused") is True
                               for optimizer in optimizers for group in optimizer.param_groups)}
        snapshot("pql_allocated")
        engine.act_batch(obs["observation"],return_tensors=True)
        report["collector_forward"] = {"executed":True,"device":str(engine.collector.device),"batch":args.envs}
        next_seed = config.runtime.seed + args.envs
        execution = PQLProbeExecution(engine,report["compiler"],compile_enabled=args.compile)
        with execution:
            for collection in range(2 + args.rounds):
                # Sampling while workers run retains the actual asynchronous scheduling.
                snapshot(f"collection_{collection+1}",synchronize=False)
                collect_started = time.perf_counter()
                action = (vector.random_actions(obs["observation"]) if collection < 2 else
                          engine.act_batch(obs["observation"],step=collection*args.envs,return_tensors=True))
                following,reward,terminated,truncated,infos = vector.step(
                    {"continuous":action.continuous,"destination":action.destination})
                engine.record_collection_time(time.perf_counter()-collect_started)
                engine.submit((obs["observation"],obs["state"],action,reward,
                               following["observation"],following["state"],terminated))
                report["collection_steps"] += 1
                report["joint_transitions"] = report["collection_steps"] * args.envs
                obs = following
                indices = list(infos["device_episode_infos"])
                if indices:
                    seeds = [None] * args.envs
                    for index in indices:
                        seeds[index] = next_seed
                        next_seed += 1
                    obs,_ = vector.reset(seed=seeds,
                                         options={"reset_mask":(terminated|truncated).cpu().numpy()})
                run.log({"probe/collections":report["collection_steps"],
                         "pql/critic_updates":engine.critic.updates,"pql/actor_updates":engine.policy.actor_updates})
            phase = report["phase"] = "worker_drain"
            persist()
            engine.flush()
            torch.cuda.synchronize(1)
        report["updates"] = execution.timings()
        report["amp"] = execution.evidence()
        report["pql_metrics"] = engine.metrics()
        report["learning_metrics"] = dict(engine.latest)
        report["critic_updates"],report["actor_updates"] = engine.critic.updates,engine.policy.actor_updates
        expected = args.rounds * args.updates_per_round
        if report["critic_updates"] != expected or report["actor_updates"] != expected:
            raise RuntimeError(f"PQL update budget mismatch: expected {expected} each")
        if {row["thread"] for row in report["updates"]} != {"pql-critic","pql-policy"}:
            raise RuntimeError("Actual asynchronous worker execution was not observed")
        if not all(row["autocast_enabled"] for row in report["updates"]):
            raise RuntimeError("A worker update ran without autocast")
        report["evidence_verified"] = base.verify_evidence(args,report)
        report["evidence_verified"]["async_pql_workers"] = True
        report["probe_outcome"],report["status"] = "passed","updates_complete"
        run.log({**engine.metrics(),**{f"learner/{k}":v for k,v in engine.latest.items()}})
        run.summary.update({"pql_metrics":engine.metrics(),"learning_metrics":engine.latest,
                            "critic_updates":engine.critic.updates,"actor_updates":engine.policy.actor_updates})
        snapshot("complete")
    except BaseException as error:
        root = error
        while root.__cause__ is not None:
            root = root.__cause__
        report.update(status="oom" if isinstance(root,torch.OutOfMemoryError) else "failed",
                      phase=phase,error_type=type(root).__name__,error=str(error),traceback=traceback.format_exc())
        if execution:
            report["amp"] = execution.evidence()
            report["worker_rows_completed"] = len(execution.rows)
        if engine:
            report.update(critic_updates=engine.critic.updates,actor_updates=engine.policy.actor_updates)
        try:
            snapshot(phase,synchronize=False)
        except Exception as measurement_error:
            report["failure_measurement_error"] = str(measurement_error)
    finally:
        if engine:
            try:
                engine.close()
                report["pql_workers_stopped"] = all(not thread.is_alive() for thread in engine.threads)
            except Exception as error:
                report["worker_close_error"] = str(error)
                if report["status"] in {"passed","updates_complete"}:
                    report["status"],report["probe_outcome"] = "failed","failed"
        base.finalize_probe(report,output,run,vector,sampler,started,persist)
    return base.json_safe(report)


if __name__ == "__main__":
    result = run_probe(parser().parse_args())
    print(json.dumps({"status":result["status"],"phase":result["phase"],
                      "result":str(Path(result["requested"]["output"])/"result.json")}),flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)
