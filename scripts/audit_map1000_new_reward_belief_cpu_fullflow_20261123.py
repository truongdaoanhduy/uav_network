"""Real bounded CPU MATD3->7 Actor snapshots->Val each->Best->heldout Test on NEW scenario.

Never load the old MATD3 checkpoint; this is a fresh six-episode smoke only.
"""
from pathlib import Path
import argparse,json
import scripts.audit_map1000_tx06_cpu_fullflow_20261123 as original
from uav_network.configuration import load_config

NEW='configs/matd3-nooracle-map1000-tx06-prior0021875-reward02-actorstable-20261123.yaml'
OUTPUT='/home/aduy/Downloads/MAP1000_NEW_REWARD_BELIEF_CPU_FULLFLOW_20261123'


def main(output):
    previous=original.CONFIG
    try:
        original.CONFIG=NEW
        original.main(output)
    finally:
        original.CONFIG=previous
    c=load_config(NEW)
    p=Path(output)/'fullflow_report.json'
    report=json.loads(p.read_text())
    assert report['train_episodes']==6 and report['validation_checkpoint_count']>=2
    assert report['heldout_test_episodes']==2
    report['science_ablation']={
        'config':NEW,'belief_prior':c.sensor.belief_prior,
        'reward_communication_efficiency':c.reward.communication_efficiency,
        'actor_output_normalization':c.matd3.output_normalization,
        'actor_output_regularization':c.matd3.actor_output_regularization,
        'warm_start':c.runtime.warm_start_checkpoint,
        'policy_quality_unverified_by_six_episodes':True,
    }
    p.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('NEW_MDP_FULL_CPU_TRAIN_VAL_BEST_TEST_OK',report['science_ablation'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=OUTPUT)
    opts=parser.parse_args();main(opts.output)
