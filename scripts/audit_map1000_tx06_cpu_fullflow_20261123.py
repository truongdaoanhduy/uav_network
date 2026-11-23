"""CPU smoke of SAME MAP1000 TX0.6 Warp physics + MATD3 train/Val-best/Test.

Only shortened horizon/episodes/mini-batches for integration correctness: this
is emphatically NOT evidence of production E4096 CUDA T4x2 throughput or 20k
training policy quality. No W&B/HF network changes or API credentials.
"""
from pathlib import Path
import argparse,json,time,hashlib,os
from uav_network.configuration import load_config
from uav_network.experiments.training import train
from uav_network.experiments.evaluation import evaluate

CONFIG='configs/matd3-nooracle-posterior-only-map1000-h256x256-orthogonal-compile-tx06-fastgrid-20261123.yaml'
DEFAULT_OUTPUT=Path('/home/aduy/Downloads/MAP1000_TX06_CPU_FULLFLOW_SMOKE_20261123')


def main(output_dir=None):
    OUT=Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError('Use a fresh CPU audit output folder; refusing to overwrite saved evidence: '+str(OUT))
    c=load_config(CONFIG)
    c.runtime.device='cpu';c.runtime.replay_device='cpu';c.runtime.num_envs=2;c.runtime.episodes=6
    c.runtime.compile_models=False;c.runtime.fused_optimizers=False
    c.world.max_steps=12
    c.matd3.hidden_sizes=[32,32];c.matd3.batch_size=16
    c.matd3.learning_starts=4;c.matd3.updates_per_step=1;c.matd3.replay_capacity=1024
    c.runtime.output_dir=str(OUT/'train')
    c.tracking.enabled=False;c.tracking.deferred_gpu_eval_enabled=True
    c.tracking.deferred_gpu_eval_keep_all=True
    c.tracking.deferred_gpu_eval_collection_stride=6
    c.validate()
    os.environ['UAV_PERIODIC_ACTORS_ONLY']='true'
    began=time.perf_counter()
    result=train(c)
    assert result['episodes']==6 and result['episode_budget_complete'],result
    manifest=json.loads((OUT/'train'/'deferred-gpu-eval'/'manifest.json').read_text())
    assert manifest['count']>=2,manifest
    assert all('training_metric_best_candidate' not in x['reasons'] for x in manifest['snapshots'])
    rankings=[]
    for j,row in enumerate(manifest['snapshots']):
        chk=OUT/'train'/'deferred-gpu-eval'/row['file']
        assert chk.is_file(),chk
        val=evaluate(str(chk),episodes=2,seed=1000000044,output_dir=str(OUT/'validation'/f'actor_{j:03d}'))
        rankings.append({'j':j,'checkpoint':str(chk),'source_step':row['global_step'],
                         'sha256':hashlib.sha256(chk.read_bytes()).hexdigest(),
                         'success':val['success_rate'],'delivery':val['mean_delivery_rate_percent'],
                         'discovery':val['mean_search_rate_percent'],'mean_return':val['mean_return']})
    # Same fixed held-out Val seeds for ALL policy snapshots, lexicographic best.
    best=max(rankings,key=lambda x:(x['success'],x['delivery'],x['discovery'],x['mean_return']))
    heldout=evaluate(best['checkpoint'],episodes=2,seed=2000000044,output_dir=str(OUT/'test'))
    report={'kind':'REAL_CPU_WARP_MATD3_BOUNDED_FULLFLOW_NOT_KAGGLE_T4_TRAIN20K',
            'map_m':[1000,1000],'grid_cell_m':25,'tx_power_cap_w':.6,
            'posterior_confirm_threshold':.99,'lr_gate':False,
            'fullflow_parts':['train','snapshot_every_6_smoke_vector_steps','validation_every_snapshot_2_seeds','best_from_validation','heldout_test_2_seeds'],
            'train_episodes':result['episodes'],'heldout_test_episodes':2,
            'validation_scenarios_per_snapshot':2,'validation_checkpoint_count':len(rankings),
            'validation':rankings,'best':best,
            'test_success_rate':heldout['success_rate'],
            'test_discovery_percent':heldout['mean_search_rate_percent'],
            'test_delivery_percent':heldout['mean_delivery_rate_percent'],
            'test_mean_return':heldout['mean_return'],
            'wall_seconds':time.perf_counter()-began}
    dest=OUT/'fullflow_report.json';dest.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('MAP1000_TX06_REAL_CPU_FULLFLOW_OK',json.dumps({k:report[k] for k in ('train_episodes','validation_checkpoint_count','validation_scenarios_per_snapshot','heldout_test_episodes','test_discovery_percent','test_delivery_percent','test_mean_return','wall_seconds')},allow_nan=False),flush=True)
    print('REPORT',dest,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(DEFAULT_OUTPUT))
    opts=parser.parse_args()
    main(opts.output)
