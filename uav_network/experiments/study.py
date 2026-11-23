"""One-factor 20/15 cases; explicit sequential execution, no cloud submissions."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import yaml

from ..configuration import _apply, load_config, config_fingerprint
from .comparison_metrics import evaluation_protocol
from .checkpoint_store import file_sha256
from .best_checkpoints import selection_identity, CRITERIA

VARIANTS = {
    "baseline": {},
    "link_budget": {"network": {"power_control": "link_budget"}},
    "buffer16": {"network": {"buffer_bytes": 16384}},
    "buffer32": {"network": {"buffer_bytes": 32768}},
    "ttl600": {"network": {"report_ttl_s": 600.}},
    "coverage8": {"sensor": {"coverage_grid_cells": 8}},
    "battery2x": {"energy": {"battery_capacity_j": 554400.}},
}
DEFAULT_SEEDS = (44, 100044, 200044)
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/comparison-20k-hybrid75-1kb.yaml"
METRICS = ("mean_return", "mean_search_rate_percent", "mean_delivery_rate_percent", "success_rate")


def _choices(values, allowed, name):
    values = list(values)
    if not values or len(set(values)) != len(values) or any(v not in allowed for v in values):
        raise ValueError(f"Invalid or duplicate {name}")
    return values


def _validate_seeds(config, seeds):
    seeds = list(seeds)
    if (not seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Training seeds must be distinct unsigned integers")
    span = config.runtime.episodes + config.runtime.num_envs
    intervals = sorted((s, s+span-1) for s in seeds)
    reserved = [(config.tracking.periodic_evaluation_seed,
                 config.tracking.periodic_evaluation_seed+config.tracking.periodic_evaluation_episodes-1),
                (config.runtime.eval_seed, config.runtime.eval_seed+config.runtime.eval_episodes-1)]
    if (any(high >= 2**32 for _, high in intervals)
            or any(a[1] >= b[0] for a,b in zip(intervals,intervals[1:]))
            or any(lo <= rh and rl <= hi for lo,hi in intervals for rl,rh in reserved)
            or reserved[0][0] <= reserved[1][1] and reserved[1][0] <= reserved[0][1]):
        raise ValueError("Training/validation/test seed banks must be disjoint")
    return seeds


def plan_study(config_path=DEFAULT_CONFIG, output_dir=None, *, seeds=DEFAULT_SEEDS,
               variants=tuple(VARIANTS), algorithms=("masac","matd3"),
               modes=("base","pql"), overrides=()):
    """Materialize resolved YAMLs and a manifest; never start training here."""
    c = load_config(config_path, overrides)
    if c.runtime.amp_mode != "off":
        raise ValueError("The controlled study requires FP32 (AMP off)")
    if c.runtime.warm_start_checkpoint != "none":
        raise ValueError("A new controlled study requires fresh training, not an old warm-start")
    w,n,s,e = c.world,c.network,c.sensor,c.energy
    if ((w.num_uavs,w.map_size_m,w.num_targets,w.num_obstacles)!=(6,1000,20,15)
            or (n.power_control,n.link_budget_margin_db,n.buffer_bytes,n.report_ttl_s)
                !=("learned",3.,3072,300.)
            or s.coverage_grid_cells != 0 or e.battery_capacity_j != 277200
            or not n.enabled or not e.enabled):
        raise ValueError("Study baseline must be the controlled 6-UAV/1-km/20-target/15-obstacle profile")
    seeds = _validate_seeds(c,seeds)
    variants = _choices(variants,VARIANTS,"variants")
    algorithms = _choices(algorithms,("masac","matd3"),"algorithms")
    modes = _choices(modes,("base","pql"),"modes")
    if output_dir is None:
        raise ValueError("Study output_dir is required")
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"Study output already exists: {root}")
    cases = []
    for variant in variants:
        for algorithm in algorithms:
            for mode in modes:
                for seed in seeds:
                    case = deepcopy(c)
                    _apply(case,VARIANTS[variant])
                    case.runtime.algorithm,case.runtime.seed = algorithm,seed
                    case.pql.enabled = mode == "pql"
                    if case.pql.enabled and algorithm=="matd3":
                        case.runtime.compile_models = False
                    geometry = "-xyz-bayes" if c.motion_dim==3 else ""
                    identity = f"20t15o{geometry}-{variant}-{algorithm}-{mode}-s{seed}"
                    config_file = f"configs/{identity}.yaml"
                    train_output,evaluation_output = f"runs/{identity}/train",f"runs/{identity}/evaluation"
                    case.runtime.output_dir = str(root/train_output)
                    case.tracking.run_name = identity
                    case.tracking.group = f"uav-20t15o-{variant}-{root.name}"
                    case.tracking.tags = list(dict.fromkeys([
                        "uav-network","scenario-v3","controlled-20t15o",f"ablation-{variant}",
                        f"training-seed-{seed}",f"env{case.runtime.num_envs}",
                        f"batch{getattr(case,algorithm).batch_size}",f"steps{case.world.max_steps}",
                        f"power-{case.network.power_control}","amp-off",
                        *(["xyz","bayesian-sensing"] if c.motion_dim==3 else [])]))
                    case.validate()
                    cases.append({"id":identity,"variant":variant,"algorithm":algorithm,"mode":mode,
                                  "seed":seed,"config_path":config_file,"train_output":train_output,
                                  "evaluation_output":evaluation_output,
                                  "scenario_hash":evaluation_protocol(case,[],{})["scenario_hash"],
                                  "config_hash":config_fingerprint(case),
                                  "config":case.to_dict()})
    root.mkdir(parents=True)
    (root/"configs").mkdir()
    for case in cases:
        (root/case["config_path"]).write_text(yaml.safe_dump(case.pop("config"),sort_keys=False))
    manifest = {"version":1,"study":"controlled-20t15o"+("-xyz-bayes" if c.motion_dim==3 else ""),"training_seeds":seeds,
                "variants":variants,"algorithms":algorithms,"modes":modes,
                "submitted_cloud_jobs":0,"case_count":len(cases),"cases":cases,
                "execution":"Explicit sequential local run; plan does not launch jobs",
                "uncertainty_scope":"Across independent training-seed means; evaluation episodes are not training seeds"}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
    return manifest


def _inside(root, relative):
    target=(root/relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Study path escapes its output directory")
    return target


def _load(manifest_path):
    path=Path(manifest_path).resolve()
    m=json.loads(path.read_text())
    if m.get("version")!=1 or m.get("study") not in {"controlled-20t15o","controlled-20t15o-xyz-bayes"} or not isinstance(m.get("cases"),list):
        raise ValueError("Unsupported study manifest")
    ids=[case["id"] for case in m["cases"]]
    if len(ids)!=len(set(ids)):
        raise ValueError("Study case IDs must be unique")
    return path.parent,m


def _case_config(root,case):
    c=load_config(_inside(root,case["config_path"]))
    if (config_fingerprint(c)!=case["config_hash"]
            or c.runtime.amp_mode!="off" or c.runtime.seed!=case["seed"]
            or c.runtime.algorithm!=case["algorithm"] or c.pql.enabled!=(case["mode"]=="pql")
            or evaluation_protocol(c,[],{})["scenario_hash"]!=case["scenario_hash"]
            or Path(c.runtime.output_dir).resolve()!=_inside(root,case["train_output"])):
        raise ValueError("Study case configuration/protocol changed after planning")
    return c


def run_study(manifest_path,case_ids=None):
    """Run selected cases locally, in order; existing train outputs are preserved."""
    from .training import train
    from .evaluation import evaluate
    root,m=_load(manifest_path)
    ids=None if case_ids is None else set(_choices(case_ids,{x["id"] for x in m["cases"]},"case IDs"))
    rows=[]
    for case in m["cases"]:
        if ids is not None and case["id"] not in ids:
            continue
        c=_case_config(root,case)
        train_output=_inside(root,case["train_output"])
        evaluation_output=_inside(root,case["evaluation_output"])
        if train_output.exists() or evaluation_output.exists():
            raise FileExistsError(f"Case output already exists: {case['id']}")
        try:
            trained=train(c)
            if trained.get("episode_budget_complete") is not True:
                rows.append({"id":case["id"],"status":"episode_budget_incomplete"})
                continue
            evaluate(trained["checkpoint"],episodes=c.runtime.eval_episodes,
                     output_dir=evaluation_output,seed=c.runtime.eval_seed)
            rows.append({"id":case["id"],"status":"completed"})
        except Exception as error:
            # Detailed sanitized evidence belongs to the existing trainer's failure files.
            rows.append({"id":case["id"],"status":"failed","error_type":type(error).__name__})
    result={"cases":rows,"submitted_cloud_jobs":0}
    (root/"execution.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def summarize_study(manifest_path):
    """Count only provenance-verified final evaluations of complete training budgets."""
    root,m=_load(manifest_path)
    groups=defaultdict(list)
    expected=defaultdict(list)
    protocols={}
    missing,rejected=[],[]
    for case in m["cases"]:
        key=(case["variant"],case["algorithm"],case["mode"])
        expected[key].append(case["seed"])
        try:
            c=_case_config(root,case)
            compact = {"scenario_hash":case["scenario_hash"], "amp_mode":c.runtime.amp_mode,
                       "requested_training_episodes":c.runtime.episodes, "num_envs":c.runtime.num_envs,
                       "max_steps":c.world.max_steps, "num_uavs":c.world.num_uavs,
                       "map_size_m":c.world.map_size_m, "num_targets":c.world.num_targets,
                       "num_obstacles":c.world.num_obstacles, "power_control":c.network.power_control,
                       "link_budget_margin_db":c.network.link_budget_margin_db,
                       "buffer_bytes":c.network.buffer_bytes, "report_ttl_s":c.network.report_ttl_s,
                       "coverage_grid_cells":c.sensor.coverage_grid_cells,
                       "battery_capacity_j":c.energy.battery_capacity_j,
                       "evaluation_episodes":c.runtime.eval_episodes,
                       "evaluation_seed_first":c.runtime.eval_seed}
            if key in protocols and protocols[key] != compact:
                raise ValueError("Study group protocols differ")
            protocols[key] = compact
            tp=_inside(root,case["train_output"])/"summary.json"
            ep=_inside(root,case["evaluation_output"])/"metrics.json"
            if not tp.exists():
                missing.append(case["id"]);continue
            trained=json.loads(tp.read_text())
            if (trained.get("episode_budget_complete") is not True
                    or trained.get("episodes")!=c.runtime.episodes):
                rejected.append({"id":case["id"],"reason":"episode_budget_incomplete"});continue
            resolved=load_config(tp.parent/"resolved_config.yaml")
            if config_fingerprint(resolved)!=case["config_hash"]:
                raise ValueError("Executed training configuration differs from planned case")
            if not ep.exists():
                missing.append(case["id"]);continue
            ev=json.loads(ep.read_text())
            protocol=ev.get("protocol",{})
            policy=ev.get("policy_provenance",{})
            selection=trained.get("checkpoint_selection",{})
            checkpoint=tp.parent/"final.pt"
            policy_step=trained.get("global_step")
            if selection.get("kind")=="validation_best":
                checkpoint=tp.parent/"best.pt"
                policy_step=selection["global_step"]
                if (json.loads((tp.parent/"best.json").read_text())!=selection
                        or selection.get("identity")!=selection_identity(c)
                        or selection.get("criteria")!=list(CRITERIA)
                        or selection.get("checkpoint_sha256")!=file_sha256(checkpoint)
                        or selection.get("validation_seeds")!=list(range(c.tracking.periodic_evaluation_seed,
                            c.tracking.periodic_evaluation_seed+c.tracking.periodic_evaluation_episodes))
                        or type(policy_step) is not int or not 0<=policy_step<=trained["global_step"]
                        or type(selection.get("episode")) is not int
                        or not 0<=selection["episode"]<=trained["episodes"]):
                    raise ValueError("Selected validation policy differs from study")
            elif selection.get("kind") not in (None,"unvalidated_final_fallback"):
                raise ValueError("Unsupported checkpoint selection")
            expected_seeds=list(range(c.runtime.eval_seed,c.runtime.eval_seed+c.runtime.eval_episodes))
            if (Path(trained.get("checkpoint","")).resolve()!=checkpoint
                    or policy.get("config_sha256")!=case["config_hash"]
                    or policy.get("checkpoint_sha256")!=file_sha256(checkpoint)
                    or policy.get("algorithm")!=case["algorithm"]
                    or policy.get("training_mode")!=case["mode"]
                    or policy.get("amp_mode")!="off"
                    or ev.get("algorithm")!=case["algorithm"] or ev.get("episodes")!=c.runtime.eval_episodes
                    or len(ev.get("records",[]))!=c.runtime.eval_episodes
                    or [r.get("seed") for r in ev["records"]]!=expected_seeds
                    or protocol.get("seeds")!=expected_seeds
                    or protocol.get("training_seed")!=case["seed"]
                    or protocol.get("scenario_hash")!=case["scenario_hash"]
                    or protocol.get("training_global_step")!=policy_step
                    or protocol.get("deterministic_policy") is not True
                    or protocol.get("training_seed_overlap") is not False
                    or protocol.get("validation_seed_overlap") is not False
                    or any(type(ev.get(k)) not in (int,float) or not np.isfinite(ev[k]) for k in METRICS)):
                raise ValueError("Final evaluation provenance/metrics differ")
            groups[key].append({"seed":case["seed"],"episodes":ev["episodes"],
                                "global_step":trained["global_step"],"policy_step":policy_step,"values":{k:ev[k] for k in METRICS}})
        except (ValueError,KeyError,TypeError,OSError) as error:
            rejected.append({"id":case["id"],"reason":"invalid_provenance_or_config",
                             "error_type":type(error).__name__})
    results=[]
    for key,seeds in expected.items():
        rows=sorted(groups[key],key=lambda r:r["seed"])
        values={k:{"mean":float(np.mean([r["values"][k] for r in rows])) if rows else None,
                   "std_over_training_seeds":float(np.std([r["values"][k] for r in rows],ddof=1))
                       if len(rows)>1 else None} for k in METRICS}
        steps=[r["global_step"] for r in rows]
        results.append({"variant":key[0],"algorithm":key[1],"mode":key[2],
                        "protocol":protocols.get(key),
                        "expected_training_seeds":seeds,"training_seeds":[r["seed"] for r in rows],
                        "training_seed_count":len(rows),"evaluation_episodes_per_seed":[r["episodes"] for r in rows],
                        "training_global_steps":steps,
                        "selected_policy_global_steps":[r["policy_step"] for r in rows],
                        "equal_environment_interaction_budget":
                            len(set(steps))==1 if steps else None,"metrics":values})
    return {"study":m["study"],"complete":not missing and not rejected,"groups":results,
            "missing_cases":missing,"rejected_cases":rejected,
            "uncertainty_scope":m["uncertainty_scope"],"overall_winner":None}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="action",required=True)
    plan=subs.add_parser("plan")
    plan.add_argument("--config",default=str(DEFAULT_CONFIG))
    plan.add_argument("--output-dir",required=True)
    plan.add_argument("--seeds",type=int,nargs="+",default=list(DEFAULT_SEEDS))
    plan.add_argument("--variants",nargs="+",choices=list(VARIANTS),default=list(VARIANTS))
    plan.add_argument("--algorithms",nargs="+",choices=["masac","matd3"],default=["masac","matd3"])
    plan.add_argument("--modes",nargs="+",choices=["base","pql"],default=["base","pql"])
    plan.add_argument("--set",dest="overrides",action="append",default=[])
    run=subs.add_parser("run");run.add_argument("--manifest",required=True);run.add_argument("--case-ids",nargs="+")
    summary=subs.add_parser("summarize");summary.add_argument("--manifest",required=True)
    args=p.parse_args(argv)
    if args.action=="plan":
        return plan_study(args.config,args.output_dir,seeds=args.seeds,variants=args.variants,
                          algorithms=args.algorithms,modes=args.modes,overrides=args.overrides)
    if args.action=="run":
        return run_study(args.manifest,args.case_ids)
    return summarize_study(args.manifest)
