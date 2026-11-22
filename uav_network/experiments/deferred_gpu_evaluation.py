"""Post-training MATD3 evaluation: vectorized fixed-seed Warp worlds on 1-2 GPUs.

This never trains a network. A worker runs 5 independent world seeds on one
CUDA device with the same immutable Actor checkpoint, then changes the Actor
weights for the next checkpoint. Two workers split checkpoint indices over
distinct GPUs. No fabricated/interpolated evaluation observations.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..algorithms.action_utils import canonical_continuous, valid_destinations
from ..algorithms.models import DeterministicActor, HybridActor
from ..configuration import Config, _apply, config_fingerprint
from ..environment.observations import observation_dim, state_dim
from ..algorithms.checkpoints import validate_observation_version
from .comparison_metrics import evaluation_protocol
from .gpu_mission_metrics import gpu_episode_record, aggregate_gpu_mission_metrics, gpu_eval_wandb_payload
from .gpu_eval_layout import plan_test
from .gpu_mission_metrics import PERIODIC_EVAL_AXIS, periodic_gpu_eval_wandb_payload
from .deferred_eval_snapshots import sha256


class GPUActorOnlyPolicy:
    """No critics, Adam, learner/replay buffer or target networks allocated."""
    def __init__(self, config, device):
        if config.runtime.algorithm not in {"masac", "matd3"}:
            raise ValueError("GPU Evaluation requires MASAC or MATD3")
        self.config = config
        self.device = torch.device(device)
        self.n = config.world.num_uavs
        self.destinations = self.n + 1
        self.continuous_dim = config.continuous_dim
        self.algorithm = config.runtime.algorithm
        algorithm_config = config.algorithm_config
        self.share = algorithm_config.share_actor_parameters
        actor_dim = observation_dim(config) + (self.n if self.share else 0)
        if self.algorithm == "masac":
            make_actor = lambda: HybridActor(
                actor_dim, self.destinations, algorithm_config.hidden_sizes,
                config.continuous_dim, layer_norm=algorithm_config.actor_layer_norm)
        else:
            make_actor = lambda: DeterministicActor(
                actor_dim, self.destinations, algorithm_config.hidden_sizes,
                config.continuous_dim, config.matd3.output_normalization,
                config.motion_dim, layer_norm=algorithm_config.actor_layer_norm,
                batch_norm=config.matd3.actor_batch_norm)
        self.actors = nn.ModuleList(
            make_actor() for _ in range(1 if self.share else self.n)
        ).to(self.device).eval()

    def load(self, payload):
        if payload.get("purpose") != "evaluation_only" or payload.get("version") != 1:
            raise ValueError("GPU evaluation requires an immutable Actor-only checkpoint")
        validate_observation_version(payload)
        c = Config()
        _apply(c, payload["config"])
        c.validate()
        if (c.runtime.algorithm != self.algorithm or
                c.runtime.environment_backend != "warp_packet" or
                payload["observation_dim"] != observation_dim(c) or
                payload["state_dim"] != state_dim(c) or
                config_fingerprint(c) != config_fingerprint(self.config)):
            raise ValueError("Checkpoint policy/scenario/configuration mismatch")
        if not payload.get("actors"):
            raise ValueError("Missing Actor weights")
        self.actors.load_state_dict(payload["actors"], strict=True)

    @torch.no_grad()
    def act_batch(self, observations):
        obs = torch.as_tensor(observations, device=self.device, dtype=torch.float32)
        n = self.n
        if self.share:
            batch = obs.shape[0]
            flat = obs.reshape(batch * n, -1)
            identity = F.one_hot(torch.arange(n, device=self.device)
                                 .expand(batch, n).reshape(-1), num_classes=n).to(flat.dtype)
            actor_in = torch.cat((flat, identity), -1)
            cont, prob = self.actors[0](actor_in, valid_destinations(flat, self.destinations))
            cont = cont.reshape(batch, n, -1)
            dest = prob.reshape(batch, n, -1).argmax(-1)
        else:
            c, destinations = [], []
            for i, actor in enumerate(self.actors):
                local = obs[:, i]
                cont_i, prob_i = actor(local, valid_destinations(local, self.destinations))
                c.append(cont_i)
                destinations.append(prob_i.argmax(-1))
            cont = torch.stack(c, 1)
            dest = torch.stack(destinations, 1)
        cont = canonical_continuous(
            cont, obs, dest, learn_power=self.config.network.power_control == "learned")
        return {"continuous": cont.contiguous(), "destination": dest.contiguous()}


def verify_manifest(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if (manifest.get("schema") != 1 or manifest.get("kind") != "deferred_gpu_eval_actor_snapshots"
            or manifest.get("count") != len(manifest.get("snapshots", []))):
        raise ValueError("Invalid deferred snapshot manifest")
    checked = []
    seen_files, seen_versions = set(), set()
    for row in manifest["snapshots"]:
        name = row.get("file")
        if (not isinstance(name, str) or Path(name).name != name or not name.endswith(".pt")
                or name in seen_files or not isinstance(row.get("policy_version"), int)
                or row["policy_version"] in seen_versions):
            raise ValueError("Invalid/duplicate snapshot manifest entry")
        path = folder / name
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError("Missing/changed Actor snapshot " + name)
        seen_files.add(name)
        seen_versions.add(row["policy_version"])
        checked.append(row)
    if any(checked[i]["global_step"] > checked[i+1]["global_step"]
           for i in range(len(checked)-1)):
        raise ValueError("Snapshot training order was not monotonic")
    return manifest


def gpu_validate_one(policy, payload, *, gpu, seeds):
    from ..environment.gpu.vector import WarpVectorEnv
    config = deepcopy(policy.config)
    config.runtime.num_envs = len(seeds)
    config.runtime.device = "cuda" if torch.device(gpu).type == "cuda" else "cpu"
    env = WarpVectorEnv(config, device=gpu)
    done = torch.zeros(len(seeds), dtype=torch.bool, device=policy.device)
    totals = torch.zeros(len(seeds), dtype=torch.float64, device=policy.device)
    infos = [None for _ in seeds]
    try:
        obs, _ = env.reset(seed=list(seeds))
        for step in range(config.world.max_steps):
            actions = policy.act_batch(obs["observation"])
            # Warp kernels guard terminated worlds. Invalid stale actions from
            # already-finished worlds must never affect collected returns.
            actions["continuous"][done] = 0
            actions["destination"][done] = 0
            obs, reward, terminated, truncated, detail = env.step(actions)
            totals += reward.to(dtype=torch.float64) * (~done)
            just_done = (terminated | truncated) & (~done)
            for i in torch.nonzero(just_done).flatten().cpu().tolist():
                info = detail["device_episode_infos"].get(i)
                if info is None:
                    raise RuntimeError("Warp terminal episode missing its true metrics")
                infos[i] = info
            done |= terminated | truncated
            if bool(done.all()):
                break
        if not bool(done.all()) or any(x is None for x in infos):
            raise RuntimeError("GPU evaluation did not terminate every world")
        returns = totals.cpu().tolist()
        records = []
        for i, (seed, info) in enumerate(zip(seeds, infos)):
            records.append(gpu_episode_record(info, config, seed=seed, episode=i+1,
                                              episode_return=returns[i]))
        progress = payload.get("progress", {})
        config_orig = policy.config
        return {"algorithm": config_orig.runtime.algorithm, "episodes": len(records), "device": str(policy.device),
                "mean_return": float(np.mean([r["return"] for r in records])),
                "success_rate": float(np.mean([r["success"] for r in records])),
                "mean_search_rate_percent": float(np.mean([r["search_rate_percent"] for r in records])),
                "mean_delivery_rate_percent": float(np.mean([r["delivery_rate_percent"] for r in records])),
                "records": records,
                "protocol": evaluation_protocol(config_orig, list(seeds), progress),
                "evaluation_backend": "warp_vector_cuda" if policy.device.type == "cuda" else "warp_vector_cpu_smoke"}
    finally:
        env.close()


def evaluate_snapshot(path, *, device="cuda:0", episodes=5, seed=None, policy=None):
    path = Path(path)
    digest = sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_observation_version(payload)
    config = Config()
    _apply(config, payload["config"])
    config.validate()
    if config.runtime.environment_backend != "warp_packet":
        raise ValueError("GPU evaluator currently requires Warp packet physics")
    if seed is None:
        seed = config.tracking.periodic_evaluation_seed
    if episodes < 1:
        raise ValueError("Evaluation requires at least one seed")
    if policy is None:
        policy = GPUActorOnlyPolicy(config, device)
    policy.load(payload)
    result = gpu_validate_one(policy, payload, gpu=device,
                              seeds=list(range(seed, seed+episodes)))
    return result_from_records(path, payload, result["records"], device=device)


def result_from_records(path, payload, records, *, device, engine=None):
    """Reconstruct the original best-selection provenance for one real Actor."""
    config = Config()
    _apply(config, payload["config"])
    config.validate()
    seeds = [int(record["seed"]) for record in records]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Missing or duplicate seed in GPU evaluation")
    if seeds != list(range(seeds[0], seeds[0]+len(seeds))):
        raise ValueError("GPU evaluation seed bank must be consecutive")
    path = Path(path)
    digest = sha256(path)
    result = {"algorithm": config.runtime.algorithm, "episodes": len(records),
        "device": str(device), "records": records,
        "mean_return": float(np.mean([x["return"] for x in records])),
        "return_std": float(np.std([x["return"] for x in records], ddof=1))
                      if len(records)>1 else 0.0,
        "success_rate": float(np.mean([x["success"] for x in records])),
        "mean_search_rate_percent": float(np.mean([x["search_rate_percent"] for x in records])),
        "mean_delivery_rate_percent": float(np.mean([x["delivery_rate_percent"] for x in records])),
        "protocol": evaluation_protocol(config, seeds, payload.get("progress", {})),
        "evaluation_backend": engine or ("warp_vector_cuda" if torch.device(device).type == "cuda"
                                        else "warp_vector_cpu_smoke")}
    result.update(aggregate_gpu_mission_metrics(records))
    result["policy_provenance"] = {"checkpoint_sha256": digest,
        "config_sha256": config_fingerprint(config), "algorithm": config.runtime.algorithm,
        "training_mode": "pql" if config.pql.enabled else "base",
        "amp_mode": config.runtime.amp_mode}
    source = payload["source"]
    result["source"] = {
        "kind": "periodic_validation", "checkpoint": str(path),
        "checkpoint_sha256": digest, "global_step": int(source["global_step"]),
        "episodes": int(source["episodes"]), "policy_version": int(source["policy_version"]),
        "seed_first": seeds[0], "seed_last": seeds[-1],
        "evaluation_episodes": len(records)}
    snapshot_reason = payload.get("progress", {}).get("snapshot_reason")
    if snapshot_reason is not None:
        result["source"]["snapshot_reason"] = snapshot_reason
    result["checkpoint_episode"] = int(source["episodes"])
    result["checkpoint_global_step"] = int(source["global_step"])
    return result



def worker(folder, out, index, workers, device, *, policies_per_batch=8,
           world_pool_size=0, checkpoint_assignment="strided"):
    folder, out = Path(folder), Path(out)
    manifest = verify_manifest(folder)
    out.mkdir(parents=True, exist_ok=True)
    seed = manifest["validation_seed_first"]
    n = manifest["validation_episodes"]
    if policies_per_batch < 1 or world_pool_size < 0:
        raise ValueError("Invalid policy batch / Warp world pool size")
    if world_pool_size and world_pool_size < n * policies_per_batch:
        raise ValueError("World pool must fit complete validation policy batch")
    from .gpu_eval_layout import plan_validation
    layout = plan_validation(manifest["count"], n, n * policies_per_batch,
                             gpu_index=index, gpu_count=workers,
                             assignment=checkpoint_assignment)
    pending_batches, completed = [], 0
    for group in layout:
        pending = []
        for idx in group.checkpoint_indices:
            row = manifest["snapshots"][idx]
            dest = out / ("eval_%04d.json" % idx)
            if dest.exists():
                existing = json.loads(dest.read_text())
                if existing.get("policy_provenance", {}).get("checkpoint_sha256") != row["sha256"]:
                    raise ValueError("Existing GPU evaluation provenance mismatch: " + str(dest))
                completed += 1
            else:
                pending.append((idx, row, dest))
        if pending:
            pending_batches.append(pending)
    policy = None
    pool = None
    try:
        for batch in pending_batches:
            snapshots = []
            for idx, row, dest in batch:
                path = folder / row["file"]
                payload = torch.load(path, map_location="cpu", weights_only=True)
                validate_observation_version(payload)
                snapshots.append((idx, row, dest, path, payload))
            start_batch_s = time.perf_counter()
            config = Config()
            _apply(config, snapshots[0][4]["config"])
            config.validate()
            if config.algorithm_config.share_actor_parameters and (len(snapshots) > 1 or world_pool_size):
                from .deferred_gpu_batched import evaluate_group
                payloads = [x[4] for x in snapshots]
                if world_pool_size:
                    if pool is None:
                        from ..environment.gpu.vector import WarpVectorEnv
                        env_config = deepcopy(config)
                        env_config.runtime.num_envs = world_pool_size
                        env_config.runtime.device = ("cuda" if torch.device(device).type == "cuda" else "cpu")
                        pool = WarpVectorEnv(env_config, device=device)
                    # Fixed 8192 world pool: up to 81 independent policies
                    # each use exactly 100 true worlds. Any unused slots are
                    # excluded from actor inference and from all metrics.
                    records_by_policy = evaluate_group(
                        config, payloads, device=device, seed=seed, episodes=n,
                        vector=pool)
                else:
                    records_by_policy = evaluate_group(config, payloads,
                                                        device=device, seed=seed, episodes=n)
                engine = ("warp_vector_vmap_cuda" if torch.device(device).type == "cuda"
                          else "warp_vector_vmap_cpu_smoke")
                results = [result_from_records(item[3], item[4], records,
                                               device=device, engine=engine)
                           for item, records in zip(snapshots, records_by_policy)]
            else:
                # Non-shared actors and single checkpoint groups use the already
                # parity-tested deterministic per-policy evaluation path.
                if policy is None:
                    policy = GPUActorOnlyPolicy(config, device)
                results = [evaluate_snapshot(item[3], device=device, seed=seed,
                                             episodes=n, policy=policy)
                           for item in snapshots]
            elapsed_batch_s = time.perf_counter() - start_batch_s
            peak_mib = (torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2)
                        if torch.device(device).type == "cuda" else 0.0)
            for (idx, row, dest, path, payload), result in zip(snapshots, results):
                result["gpu_eval_batch_policies"] = len(snapshots)
                result["gpu_eval_world_pool_size"] = (world_pool_size if pool is not None else 0)
                result["gpu_eval_batch_time_s"] = elapsed_batch_s
                result["gpu_eval_gpu_peak_allocated_mib"] = peak_mib
                if result["policy_provenance"]["checkpoint_sha256"] != row["sha256"]:
                    raise ValueError("GPU evaluation source changed after manifest verification")
                temporary = dest.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(result, allow_nan=False, indent=2))
                temporary.replace(dest)
                completed += 1
                print("GPU_EVAL_COMPLETE", idx, row["episodes"], row["global_step"],
                      result["evaluation_backend"], flush=True)
    finally:
        if pool is not None:
            pool.close()
    return completed



def publish_metrics(folder, results, *, wandb_run_id=None, hf_best=False,
                    external_wandb_owner=False):
    from .best_checkpoints import BestCheckpoint
    manifest = verify_manifest(folder)
    if not manifest["count"] or len(results) != manifest["count"]:
        raise RuntimeError("Incomplete GPU validation results; refusing to pick best")
    first_payload = torch.load(Path(folder) / manifest["snapshots"][0]["file"],
                               map_location="cpu", weights_only=True)
    config = Config()
    _apply(config, first_payload["config"])
    config.validate()
    best = BestCheckpoint(config, Path(folder).parent / "deferred_gpu_results")
    for result in results:
        best.consider(result, result["source"])
    if best.metadata is None:
        raise RuntimeError("No valid best checkpoint after GPU validation")
    output = best.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation_results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False))
    import csv
    # Paper-facing export uses the same fields as W&B, including empty values
    # for metrics with missing denominators (never substitute zero J/target).
    metric_columns = [
        "discovery_rate_percent", "gcs_delivery_rate_percent",
        "discovery_time_capped_mean_s", "gcs_delivery_time_capped_mean_s",
        "energy_per_delivered_target_j", "discovery_auc", "delivery_auc",
        "mission_success_rate_percent", "mission_completion_time_capped_s",
        "map_coverage_percent", "false_confirmation_per_1000_measurements",
        "first_discovery_to_gcs_delay_delivered_mean_s", "packet_pdr_percent",
        "mean_total_uav_energy_j", "mean_targets_found", "mean_targets_delivered",
    ]
    with (output / "validation_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "checkpoint_episode", "checkpoint_global_step", "policy_version",
            "mean_return", "search_rate_percent", "delivery_rate_percent",
            "evaluation_episodes", "backend", *metric_columns])
        writer.writeheader()
        for result in results:
            writer.writerow({
                "checkpoint_episode": result["source"]["episodes"],
                "checkpoint_global_step": result["source"]["global_step"],
                "policy_version": result["source"]["policy_version"],
                "mean_return": result["mean_return"],
                "search_rate_percent": result["mean_search_rate_percent"],
                "delivery_rate_percent": result["mean_delivery_rate_percent"],
                "evaluation_episodes": result["episodes"],
                "backend": result["evaluation_backend"],
                **{key: result["mission_metrics"].get(key) for key in metric_columns},
            })
    # Even an off-schedule checkpoint can win validation and the final test.
    # Match the plotted curve to the periodic schedule recorded in the manifest.
    # Older snapshot banks did not label reasons and keep their original series.
    # Include the final off-grid checkpoint as a real point (never interpolate).
    periodic_steps = {row["global_step"] for row in manifest["snapshots"]
                      if any(reason in row.get("reasons", ("periodic",))
                             for reason in ("periodic", "final"))}
    # W&B/HF publication must not invalidate completed, verified GPU metrics.
    # Both services can fail transiently or be unconfigured on a recovery run.
    publication = {"wandb_sync": "not_requested", "hf_best": "not_requested"}
    hub = None
    if external_wandb_owner:
        # The one-session parent owns the same resumed W&B run throughout
        # Eval + Final Test. Workers write atomic JSONs; the parent streams
        # evaluated metrics into W&B without concurrent SDK run owners.
        publication["wandb_sync"]="external_fullflow_owner"
    if wandb_run_id and not external_wandb_owner:
        try:
            import wandb
            if not os.environ.get("WANDB_API_KEY"):
                raise RuntimeError("W&B logging requested without WANDB_API_KEY")
            # Retry-safe: this GPU evaluator can resume after a Kaggle deadline.
            # Never append duplicated history points for checkpoints already logged.
            history = wandb.Api(timeout=50).run(
                f"{config.tracking.entity}/{config.tracking.project}/{wandb_run_id}")
            legacy_axis = "gpu_eval/checkpoint_global_step"
            recorded = (getattr(history, "_attrs", {}).get("historyKeys") or {}).get("keys")
            done = {}
            for axis in (legacy_axis, PERIODIC_EVAL_AXIS):
                done[axis] = set() if recorded is not None and axis not in recorded else {
                    int(row[axis]) for row in history.scan_history(keys=[axis], page_size=500)
                    if isinstance(row.get(axis), (int, float))}
            last_snapshot = max(row["global_step"] for row in manifest["snapshots"])
            native_steps = {row["global_step"] for row in manifest["snapshots"]
                            if any(reason in row.get("reasons", ()) for reason in ("periodic", "final"))
                            or ("reasons" not in row and row["global_step"] < last_snapshot)}
            pending = []
            for result in results:
                step = result["source"]["global_step"]
                if step not in periodic_steps:
                    continue
                legacy = gpu_eval_wandb_payload(result)
                row = {}
                if step not in done[legacy_axis]:
                    row.update(legacy)
                    row.update({
                        "gpu_eval/batch_policies": result.get("gpu_eval_batch_policies", 1),
                        "gpu_eval/batch_time_s": result.get("gpu_eval_batch_time_s", 0.0),
                        "gpu_eval/gpu_peak_allocated_mib": result.get("gpu_eval_gpu_peak_allocated_mib", 0.0),
                    })
                if step in native_steps and step not in done[PERIODIC_EVAL_AXIS]:
                    row.update(periodic_gpu_eval_wandb_payload(legacy))
                if row:
                    # Deferred evaluation runs AFTER training; use the saved
                    # checkpoint's REAL training X, never evaluation wall time.
                    row["global_step"] = int(step)
                    pending.append(row)
            if pending:
                with wandb.init(entity=config.tracking.entity, project=config.tracking.project,
                                id=wandb_run_id, resume="must", reinit=True) as run:
                    run.define_metric("global_step")
                    for axis in (legacy_axis, PERIODIC_EVAL_AXIS):
                        if any(axis in row for row in pending):
                            run.define_metric(axis)
                            run.define_metric(axis.split("/", 1)[0] + "/*",
                                              step_metric="global_step", step_sync=False)
                    for row in pending:
                        run.log(row)
            publication["wandb_sync"] = "completed"
        except Exception as error:
            publication["wandb_sync"] = "failed"
            publication["wandb_error_type"] = type(error).__name__
            print("GPU_EVAL_WANDB_PUBLICATION_FAILED", type(error).__name__, flush=True)
    if hf_best:
        try:
            from ..experiments.checkpoint_store import ensure_private_store
            api, repo = ensure_private_store(config)
            prefix = "deferred-gpu-eval/" + (wandb_run_id or "manual")
            for item in ("best.pt", "best.json"):
                api.upload_file(path_or_fileobj=str(output / item),
                                path_in_repo=prefix + "/" + item,
                                repo_id=repo, repo_type="model",
                                commit_message=f"GPU validated best {config.runtime.algorithm.upper()} Actor")
            hub = prefix
            publication["hf_best"] = "completed"
        except Exception as error:
            publication["hf_best"] = "failed"
            publication["hf_error_type"] = type(error).__name__
            print("GPU_EVAL_HF_PUBLICATION_FAILED", type(error).__name__, flush=True)
    return {"validation_completed": len(results), "best_checkpoint": str(output / "best.pt"),
            "best_source_episode": best.metadata["episode"],
            "best_global_step": best.metadata["global_step"], "hf_prefix": hub,
            "publication": publication}




def _sharded_final_test(best_snapshot, *, folder, output, gpus, episodes, seed, capacity):
    """Deterministic 2-T4 Final Test. Every held-out seed is evaluated ONCE.

    Test results are never used to choose a checkpoint. Failed/missing shards
    raise, rather than silently shrinking the denominator. All shards are
    written atomically and can be resumed by rerunning this eval-only job.
    """
    from .gpu_eval_layout import plan_test
    if capacity<1:
        raise ValueError('Final Test worlds-per-GPU capacity must be positive')
    plan=plan_test(int(episodes),int(gpus),int(capacity))
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    processes=[]
    for index, ranges in enumerate(plan):
        env=dict(os.environ)
        env['CUDA_VISIBLE_DEVICES']=str(index)
        env['OMP_NUM_THREADS']='1'
        env['MKL_NUM_THREADS']='1'
        env['OPENBLAS_NUM_THREADS']='1'
        command=[sys.executable,'-m','uav_network.experiments.deferred_gpu_evaluation',
          '--folder',str(folder),'--output',str(output),'--test-worker',
          '--test-checkpoint',str(best_snapshot),'--test-seed',str(seed),
          '--test-ranges-json',json.dumps(ranges),
          '--index',str(index),'--device','cuda:0']
        processes.append((index,subprocess.Popen(command,env=env)))
    failures=[]
    for index,process in processes:
        code=process.wait()
        if code:
            failures.append((index,code))
    if failures:
        raise RuntimeError('Final Test GPU workers failed: '+repr(failures))
    checkpoint_sha=sha256(best_snapshot)
    records=[]
    for index,ranges in enumerate(plan):
        for start,stop in ranges:
            path=output/f'final_test_gpu{index}_{start:06d}_{stop:06d}.json'
            if not path.is_file():
                raise RuntimeError('Missing real Final Test GPU shard: '+str(path))
            data=json.loads(path.read_text())
            if (data.get('checkpoint_sha256')!=checkpoint_sha or
                data.get('start')!=start or data.get('stop')!=stop or
                data.get('gpu_index')!=index):
                raise RuntimeError('Final Test shard provenance mismatch: '+str(path))
            samples=data.get('records',[])
            if len(samples)!=stop-start or [row['seed'] for row in samples]!=list(
                    range(seed+start,seed+stop)):
                raise RuntimeError('Final Test shard seeds incomplete/duplicated')
            records.extend(samples)
    records.sort(key=lambda row:row['seed'])
    if len(records)!=episodes or [x['seed'] for x in records]!=list(range(seed,seed+episodes)):
        raise RuntimeError('Final Test 4000 scenarios not exactly complete')
    payload=torch.load(best_snapshot,map_location='cpu',weights_only=True)
    result=result_from_records(best_snapshot,payload,records,
                               device='cuda:0',engine='warp_vector_cuda_split_two_t4')
    result['gpu_test_worker_count']=len(plan)
    result['gpu_test_assignments']=[
        {'gpu_index':i,'ranges':[list(x) for x in ranges]} for i,ranges in enumerate(plan)]
    result['gpu_test_capacity_per_gpu']=capacity
    return result


def _run_final_test_worker(args,folder,output):
    """Run one GPU's disjoint seed subranges sequentially (no cross-GPU RNG)."""
    if not args.test_checkpoint or args.test_ranges_json is None:
        raise ValueError('Final Test worker requires --test-checkpoint and --test-ranges-json')
    checkpoint=Path(args.test_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError('Validated best Actor checkpoint missing')
    expected_sha=sha256(checkpoint)
    ranges=json.loads(args.test_ranges_json)
    if not isinstance(ranges,list):
        raise ValueError('Test shard ranges must be an ordered list')
    output.mkdir(parents=True,exist_ok=True)
    completed=0
    for pair in ranges:
        if (not isinstance(pair,(list,tuple)) or len(pair)!=2 or
            type(pair[0]) is not int or type(pair[1]) is not int or
            not 0<=pair[0]<pair[1]):
            raise ValueError('Invalid held-out shard seed offsets')
        start,stop=pair
        target=output/f'final_test_gpu{args.index}_{start:06d}_{stop:06d}.json'
        if target.exists():
            existing=json.loads(target.read_text())
            if (existing.get('checkpoint_sha256')==expected_sha and
                existing.get('start')==start and existing.get('stop')==stop and
                existing.get('gpu_index')==args.index and
                [r['seed'] for r in existing.get('records',[])]==list(
                    range(args.test_seed+start,args.test_seed+stop))):
                completed+=stop-start
                print('FINAL_TEST_SHARD_RESUMED',args.index,start,stop,flush=True)
                continue
            raise RuntimeError('Existing Final Test shard is stale: '+str(target))
        result=evaluate_snapshot(checkpoint,device=args.device,
                                 episodes=stop-start,seed=args.test_seed+start)
        records=result['records']
        if len(records)!=stop-start:
            raise RuntimeError('Final Test shard evaluator returned incomplete result')
        item={'gpu_index':args.index,'start':start,'stop':stop,
              'checkpoint_sha256':expected_sha,'records':records}
        temporary=target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(item,allow_nan=False))
        temporary.replace(target)
        completed+=len(records)
        print('FINAL_TEST_SHARD_COMPLETE',args.index,start,stop,
              'seeds',len(records),flush=True)
    return completed

def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate Actor-only checkpoints on two CUDA GPUs")
    p.add_argument("--folder", required=True, help="Directory containing manifest.json and snapshot *.pt")
    p.add_argument("--output", required=True, help="Directory for deterministic results")
    p.add_argument("--gpus", type=int, choices=(1,2), default=2)
    p.add_argument("--checkpoint-assignment", choices=("strided","contiguous_batches","contiguous_adaptive"), default="strided")
    p.add_argument("--world-pool-size", type=int, default=0,
                   help="Opt-in reusable Warp worlds per GPU worker; 0=legacy evaluator")
    p.add_argument("--policies-per-batch", type=int, default=8,
                   help="Number of distinct Shared Actor checkpoints per GPU vector launch")
    # V8 one_session.py passed these THREE flags, but the evaluator did not
    # recognize any of them, so Validation was never started.
    p.add_argument("--max-eval-worlds-per-gpu", type=int, default=0)
    p.add_argument("--max-test-worlds-per-gpu", type=int, default=0)
    p.add_argument("--final-test-batch-size", type=int, default=0)
    p.add_argument("--test-worker", action="store_true")
    p.add_argument("--test-checkpoint", default=None)
    p.add_argument("--test-ranges-json", default=None)
    p.add_argument("--test-seed", type=int, default=2000000044)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--wandb-run-id", default=None)
    p.add_argument("--hf-best", action="store_true")
    p.add_argument("--external-wandb-owner", action="store_true",
                   help="Parent one_session maintains the sole W&B SDK run")
    p.add_argument("--final-test", action="store_true",
                   help="Run an independent 30-episode final test on the selected best Actor")
    args = p.parse_args(argv)
    folder, output = Path(args.folder).resolve(), Path(args.output).resolve()
    manifest = verify_manifest(folder)
    if args.test_worker:
        count=_run_final_test_worker(args,folder,output)
        print('FINAL_TEST_GPU_SHARD_COUNT',count,flush=True)
        return 0
    if args.max_eval_worlds_per_gpu<0 or args.max_test_worlds_per_gpu<0 or args.final_test_batch_size<0:
        raise ValueError('GPU world/batch caps must be nonnegative')
    if (args.max_eval_worlds_per_gpu and
            args.policies_per_batch*manifest['validation_episodes']>args.max_eval_worlds_per_gpu):
        raise ValueError('Validation policy batch exceeds GPU world capacity')
    if args.worker:
        count = worker(folder, output, args.index, args.workers, args.device,
                       policies_per_batch=args.policies_per_batch,
                       world_pool_size=args.world_pool_size,
                       checkpoint_assignment=args.checkpoint_assignment)
        print(json.dumps({"worker": args.index, "evaluated": count}), flush=True)
        return 0
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpus:
        raise RuntimeError("Expected %d CUDA GPUs for post-training evaluation" % args.gpus)
    if not manifest["count"]:
        raise RuntimeError("No snapshots available for validation")
    output.mkdir(parents=True, exist_ok=True)
    processes = []
    for index in range(args.gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(index)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        command = [sys.executable, "-m", "uav_network.experiments.deferred_gpu_evaluation",
                   "--worker", "--folder", str(folder), "--output", str(output),
                   "--index", str(index), "--workers", str(args.gpus), "--device", "cuda:0",
                   "--policies-per-batch", str(args.policies_per_batch),
                   "--world-pool-size", str(args.world_pool_size),
                   "--checkpoint-assignment", args.checkpoint_assignment]
        processes.append(subprocess.Popen(command, env=env))
    failed = []
    for index, process in enumerate(processes):
        status = process.wait()
        if status != 0:
            failed.append((index, status))
    if failed:
        raise RuntimeError("GPU validation workers failed: " + repr(failed))
    results = [json.loads((output / ("eval_%04d.json" % i)).read_text())
               for i in range(manifest["count"])]
    results.sort(key=lambda v: (v["checkpoint_global_step"], v["checkpoint_episode"]))
    report = publish_metrics(folder, results, wandb_run_id=args.wandb_run_id,
                             hf_best=args.hf_best,
                             external_wandb_owner=args.external_wandb_owner)
    # Mirror validated best + paper-ready CSV next to the per-checkpoint JSONs.
    import shutil
    best_dir = Path(report["best_checkpoint"]).parent
    for name in ("best.pt", "best.json", "validation_results.json", "validation_results.csv"):
        shutil.copy2(best_dir / name, output / name)
    report["best_checkpoint"] = str(output / "best.pt")
    if args.final_test:
        # Final-test seeds were held out of all 200 checkpoint selections.
        best_snapshot = Path(report["best_checkpoint"])
        payload = torch.load(best_snapshot, map_location="cpu", weights_only=True)
        final_config = Config()
        _apply(final_config, payload["config"])
        final_config.validate()
        # Genuine 4k held-out scenarios across BOTH T4s (2000 per T4 at
        # capacity>=2000); never allocate all 4000 worlds on CUDA:0.
        test_capacity=(args.final_test_batch_size or
                       args.max_test_worlds_per_gpu or
                       max(1,(final_config.runtime.eval_episodes+args.gpus-1)//args.gpus))
        if args.max_test_worlds_per_gpu:
            test_capacity=min(test_capacity,args.max_test_worlds_per_gpu)
        final = _sharded_final_test(best_snapshot,folder=folder,output=output,
                                    gpus=args.gpus,
                                    episodes=final_config.runtime.eval_episodes,
                                    seed=final_config.runtime.eval_seed,
                                    capacity=test_capacity)
        if (final["protocol"]["training_seed_overlap"] is not False
                or final["protocol"]["validation_seed_overlap"] is not False):
            raise RuntimeError("Final-test bank is not held out from training/validation")
        final_path = output / "final_test.json"
        final_path.write_text(json.dumps(final, indent=2, allow_nan=False))
        report["final_test"] = {
            "episodes": final["episodes"],
            "search_rate_percent": final["mean_search_rate_percent"],
            "delivery_rate_percent": final["mean_delivery_rate_percent"],
            "mission_success_rate_percent": 100. * final["success_rate"],
            "mean_return": final["mean_return"],
            "mission_metrics": final["mission_metrics"],
            "path": str(final_path),
        }
        if args.wandb_run_id and not args.external_wandb_owner:
            # Held-out final test belongs in W&B run.summary, NOT in the
            # validation convergence series used to select the best Actor.
            import wandb
            summary = {f"gpu_final/{name}": value
                       for name, value in final["mission_metrics"].items()
                       if value is not None}
            summary.update({"gpu_final/episodes": final["episodes"],
                            "gpu_final/mean_return": final["mean_return"],
                            "gpu_final/validation_best_global_step": report["best_global_step"]})
            try:
                wbrun = wandb.Api(timeout=50).run(
                    f"{final_config.tracking.entity}/{final_config.tracking.project}/{args.wandb_run_id}")
                wbrun.summary.update(summary)
                # Final Test is ONE held-out observation of the best checkpoint,
                # not a validation convergence series; preserve summary too.
                with wandb.init(entity=final_config.tracking.entity,
                                project=final_config.tracking.project,
                                id=args.wandb_run_id, resume="must", reinit=True) as history_run:
                    history_run.define_metric("global_step")
                    history_run.define_metric("gpu_final/*", step_metric="global_step", step_sync=False)
                    history_run.log({"global_step": int(report["best_global_step"]),
                                     **summary})
                report["final_test"]["wandb_summary_synced"] = True
            except Exception as error:
                # Never invalidate a scientifically complete final test
                # because the W&B API is temporarily unavailable.
                report["final_test"]["wandb_summary_synced"] = False
                report["final_test"]["wandb_error_type"] = type(error).__name__
    report_file = output / "report.json"
    report_file.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
