#!/usr/bin/env python3
"""V112 C20/C09/C04 DS-only, non-learning, strictly paired first-terminal experiment.

Each pair has the same master seed, world index, UAV spawn, target RNG recipe,
controller implementation and horizon. Exactly one YAML parameter changes in
an edge. 5 master seeds are statistical CLUSTERS; worlds within one seed are
NOT counted as independent seeds. Results are diagnostic, not proofs of
beneficial PEER/MULE nor of correlated full-mission false-alarm safety.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'research/zhang_xue/src'), str(ROOT / 'scripts/gpu')]
from v98_scenario_ablation_suite import _run_case
from zhang_xue.configuration.loader import load_scenario

CONFIGS = {
    'base': 'scenario_v103_ds_only_1km.yaml',
    'rf_only': 'scenario_v111_ds_rf_stress_no_peer_1km.yaml',
    'rf_peer': 'scenario_v111_ds_rf_stress_peer_only_1km.yaml',
    'rf_peer_mule': 'scenario_v105_ds_rf_stress_1km.yaml',
    'base_peer': 'scenario_v111_ds_base_peer_only_1km.yaml',
    'base_threshold099': 'scenario_v111_ds_ablation_th099_1km.yaml',
}
EDGES = {
    'rf_noise_only': ('base','rf_only','mac_noise_power_w'),
    'peer_effect_under_rf_stress': ('rf_only','rf_peer','enable_peer_dissemination'),
    'mule_increment_under_rf_stress': ('rf_peer','rf_peer_mule','enable_data_mule_forwarding'),
    'peer_effect_under_open_rf': ('base','base_peer','enable_peer_dissemination'),
    'threshold_090_to_099': ('base','base_threshold099','confirm_th'),
}
FIELDS=('gcs_delivery_pct','legal_coverage_pct','fp_ever','mean_battery_j','return')


def bootstrap_seed_blocks(a, b, seeds, *, n=5000, random_seed=280112):
    assert a['seed']==b['seed'] and a['env_id']==b['env_id'], 'pair alignment lost'
    rng=np.random.default_rng(random_seed)
    out={}
    seedarr=np.array(a['seed'])
    for field in FIELDS:
        delta=np.asarray(b[field],dtype=np.float64)-np.asarray(a[field],dtype=np.float64)
        block=np.array([delta[seedarr==seed].mean() for seed in seeds],dtype=np.float64)
        boot=rng.choice(block, size=(n,len(seeds)), replace=True).mean(axis=1)
        out[field]={
            'mean_right_minus_left':float(delta.mean()),
            'per_seed_means':{str(s):float(m) for s,m in zip(seeds,block)},
            'bootstrap_seed_cluster_ci95': [float(x) for x in np.quantile(boot,[.025,.975])],
            'num_independent_master_seeds':len(seeds),
            'num_world_pairs':len(delta),
        }
    return out


def verify_configs(paths):
    from dataclasses import asdict
    configs={k: asdict(load_scenario(v)) for k,v in paths.items()}
    for edge,(a,b,param) in EDGES.items():
        changed=[k for k in configs[a] if configs[a][k]!=configs[b][k]]
        if changed != [param]:
            raise AssertionError(f'{edge}: expected [{param}], actual {changed}')
    return {k:{'filename':v.name,'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in paths.items()}


def run(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--envs-per-seed',type=int,default=4)
    ap.add_argument('--seeds',default='44,45,46,47,48')
    ap.add_argument('--steps',type=int,default=900)
    ap.add_argument('--cases',default=','.join(CONFIGS))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(argv)
    seeds=[int(x) for x in args.seeds.split(',') if x]
    if len(seeds)<5 or len(set(seeds))!=len(seeds):raise ValueError('>=5 distinct master seeds required')
    cases=[x for x in args.cases.split(',') if x]
    if len(set(cases))!=len(cases) or any(x not in CONFIGS for x in cases):raise ValueError('invalid cases')
    if args.steps!=900:raise ValueError('V112 gate requires full 900 steps')
    paths={k: ROOT/'research/zhang_xue/configs'/v for k,v in CONFIGS.items()}
    digests=verify_configs(paths)
    out={
        'status':'RUNNING_NOT_FROZEN','gate':'V112_PAIRED_DS_SCENARIO',
        'scenario_frozen':False,'no_rl_training':True,
        'device':args.device,'steps':args.steps,'master_seeds':seeds,
        'envs_per_master_seed':args.envs_per_seed,
        'statistical_unit':'master_seed; GPU worlds nested per master seed',
        'controller':'non_oracle_frontier_actions_np (local obs/mask only)',
        'terminal_rule':'first terminal or step900; no reset; no terminal leakage',
        'independence_caveat':'true-target positions depend on master_seed and world_index; same fixture across YAML. No correlated-sensor mission model.',
        'confounding_check':'exactly one scenario dataclass field differs per edge',
        'config_sha256':digests,
        'cases':{},'edges':{},'warnings':[],
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save(): args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    save()
    for case in cases:
        print('START',case,flush=True)
        t=time.monotonic()
        res=_run_case(paths[case],(),envs=args.envs_per_seed,steps=args.steps,seeds=seeds,device=args.device)
        res['duration_v112_sec']=round(time.monotonic()-t,3)
        out['cases'][case]=res
        save()
        print('DONE',case,'delivery',res['mean_gcs_delivery_pct'],'peer',res['mac_telemetry_mean']['stats_peer'],'sec',res['duration_v112_sec'],flush=True)
    for name,(left,right,changed) in EDGES.items():
        if left not in out['cases'] or right not in out['cases']:continue
        a=out['cases'][left]; b=out['cases'][right]
        out['edges'][name]={
            'left':left,'right':right,'only_changed':changed,
            'paired':bootstrap_seed_blocks(a['per_world'],b['per_world'],seeds),
            'tx_actions_left':a['actions_selected'],'tx_actions_right':b['actions_selected'],
            'network_stats_left':a['mac_telemetry_mean'],'network_stats_right':b['mac_telemetry_mean'],
        }
    error_cases={k:{'safety_fail':v['safety_failed_worlds'],'overflow':v['overflow_worlds']} for k,v in out['cases'].items()
                 if v['safety_failed_worlds'] or v['overflow_worlds']}
    out['status']='MEASUREMENT_VALID_NOT_FROZEN' if not error_cases else 'SAFETY_OR_OVERFLOW_FAIL_NOT_FROZEN'
    if error_cases:out['warnings'].append({'invalid_cases':error_cases})
    out['warnings']+=['DO NOT infer beneficial PEER/MULE if paired seed CI crosses zero or peer packets not observed',
                       'No upstream SimPy packet-level parity in this test',
                       'Uncalibrated DS correlation remains an independent freeze blocker',
                       'Exploratory bootstrap uncertainty over 5 master seeds is imprecise']
    save()
    print('DONE V112',out['status'],flush=True)
    return 0 if not error_cases else 2

if __name__=='__main__':raise SystemExit(run())
