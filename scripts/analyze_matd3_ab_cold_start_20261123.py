"""Read-only report correction for cold-start-contaminated GPU MATD3 A/B.

The first job in each fresh Kaggle Python environment can pay Warp compilation,
CUDA allocator and cache costs. Never mistake that for a method speedup.
"""
from __future__ import annotations
import json,statistics,sys
from pathlib import Path


def analyze(runs):
    if len(runs)<4 or not all(isinstance(x,dict) for x in runs):
        raise ValueError('Need at least two AB/BA pairs')
    for r in runs:
        if r['variant'] not in ('baseline','candidate') or r['train_joint_transitions_per_second']<=0:
            raise ValueError('Malformed training A/B record')
    if len({x['science_sha256'] for x in runs})!=1:
        raise ValueError('Cannot compare mismatched training MDPs')
    def median(items):
        return {kind:statistics.median(x['train_joint_transitions_per_second']
              for x in items if x['variant']==kind) for kind in ('baseline','candidate')}
    raw=median(runs)
    # First process is cold. Discard it from BOTH scientific conclusions and
    # comparative reporting; candidates cannot benefit by being launched second.
    warm=runs[1:]
    counts={kind:sum(x['variant']==kind for x in warm) for kind in ('baseline','candidate')}
    if min(counts.values())==0:raise ValueError('No warm sample of both variants')
    steady=median(warm)
    science_equal=len({x['science_sha256'] for x in runs})==1
    model_equal=len({x['model_sha256'] for x in runs})==1
    episodes_equal=len({x['episode_sha256'] for x in runs})==1
    informative=all(x.get('completed_episodes',0)>0 for x in runs)
    warm_gain=100*(steady['candidate']/steady['baseline']-1)
    return dict(first_run_startup_excluded=True,
        cold_start_variant=runs[0]['variant'],
        raw_all_runs_transitions_per_second=raw,
        raw_all_runs_speedup_percent=100*(raw['candidate']/raw['baseline']-1),
        steady_state_transitions_per_second=steady,
        steady_state_candidate_speedup_percent=warm_gain,
        warm_run_counts=counts,
        science_match=science_equal,model_match=model_equal,
        episode_records_match=episodes_equal,
        episode_records_informative=informative,
        # A short <1500-step pilot doesn't validate outcomes or steady speed:
        # two warm full-episode samples per arm are the MINIMUM verification.
        candidate_verified_beneficial=bool(science_equal and model_equal and
            episodes_equal and informative and min(counts.values())>=2 and warm_gain>2.0))


if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('usage: analyze_matd3_ab_cold_start_20261123.py IN.json OUT.json')
    source=Path(sys.argv[1]);destination=Path(sys.argv[2])
    a=json.loads(source.read_text());a['corrected_benchmark_analysis']=analyze(a['runs'])
    destination.write_text(json.dumps(a,indent=2,allow_nan=False))
    print('CORRECTED_TRAIN_AB_ANALYSIS',json.dumps(a['corrected_benchmark_analysis'],indent=2))
