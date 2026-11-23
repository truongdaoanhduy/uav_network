"""Read-only, source-inspired reward-convergence figure comparisons.

Training returns: MADDPG-style fixed completed-episode block means; EPyMARL-style
rolling mean; and per-global-transition-bin completed-episode means. These are
NOT fixed-policy validation results. Raw records and run IDs remain traceable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RUNS = {
    'MATD3 BASE': '971f7cc2',
    'MATD3 PQL': '30bdc182',
    'MASAC BASE': '62cfc303',
    'MASAC PQL (partial)': '43494a2c',
}
REQUIRED = ('episode/index', 'episode/return', 'episode/length', 'global_step')


def fixed_episode_blocks(episode, reward, steps, *, width=256):
    if width < 1:
        raise ValueError('width must be positive')
    episode = np.asarray(episode, dtype=np.int64)
    reward, steps = np.asarray(reward, dtype=float), np.asarray(steps, dtype=float)
    bins = (episode - 1) // width
    output = []
    for k in np.unique(bins):
        mask = bins == k
        output.append({'episode': float(np.mean(episode[mask])),
                       'global_step': float(np.mean(steps[mask])),
                       'reward': float(np.mean(reward[mask])),
                       'std': float(np.std(reward[mask])),
                       'n': int(mask.sum()), 'complete': int(mask.sum()) == width})
    return output


def epymarl_rolling_mean(episode, reward, *, width=200, stride=50):
    """EPyMARL plot_results.py smooth_data(): mean of each fixed-size window.

    For display/data export only, downsample the resulting window means.
    """
    episode, reward = np.asarray(episode,dtype=float),np.asarray(reward,dtype=float)
    if reward.size < width:
        return []
    avg = np.convolve(reward, np.ones(width) / width, mode='valid')
    x = np.convolve(episode, np.ones(width) / width, mode='valid')
    idx = np.unique(np.append(np.arange(0, len(avg), stride), len(avg) - 1))
    return [{'episode':float(x[i]),'reward':float(avg[i])} for i in idx]


def transition_bin_means(steps, reward, *, width=250000, min_count=5):
    """Group by when completed episodes were LOGGED, not by start time.

    This is a diagnostic, not a new policy score; cells with <5 episodes are
    omitted, and missing cells are explicitly NaN gaps, never interpolated.
    """
    steps, reward = np.asarray(steps,dtype=float), np.asarray(reward,dtype=float)
    bins = np.floor(steps / width).astype(int)
    out = []
    for k in range(int(bins.min()),int(bins.max())+1):
        m = bins == k
        n = int(m.sum())
        out.append({'global_step':float((k+0.5)*width),
                    'reward':float(np.mean(reward[m])) if n>=min_count else None,
                    'n':n})
    return out


def fetch_history(api, project, run_id):
    run = api.run(f'{project}/{run_id}')
    rows=[]
    for item in run.scan_history(keys=list(REQUIRED),page_size=2000):
        if not all(k in item and item[k] is not None for k in REQUIRED):
            continue
        rows.append(tuple(float(item[k]) for k in REQUIRED))
    if not rows:
        raise RuntimeError(f'{run_id}: no episode reward history')
    rows=np.array(rows,dtype=float)
    rows=rows[np.argsort(rows[:,0])]
    if len(np.unique(rows[:,0]))!=len(rows):
        raise RuntimeError(f'{run_id}: duplicate episode indices')
    world=run.config.get('world',{})
    return run,rows,tuple(world.get(k) for k in ('num_uavs','num_targets','num_obstacles','max_steps','map_size_m'))


def draw_curves(aggregates, selector, xlabel, title, path):
    fig,ax=plt.subplots(figsize=(11,5.8),layout='constrained')
    for name,data in aggregates.items():
        rows=data[selector]
        xx=np.array([r['global_step'] if selector=='stepbins' else r['episode'] for r in rows],dtype=float)
        yy=np.array([r['reward'] if r['reward'] is not None else np.nan for r in rows],dtype=float)
        if selector=='stepbins': xx=xx/1e6
        if selector=='blocks':
            for i,row in enumerate(rows):
                if not row['complete']: yy[i]=np.nan
        ax.plot(xx,yy,label=f"{name} (n={data['observed_episodes']:,})",linewidth=1.8,alpha=.88)
    ax.axhline(0,lw=.8,ls=':',color='.5')
    ax.set_xlabel(xlabel);ax.set_ylabel('Training episode return (mean)')
    ax.set_title(title);ax.grid(alpha=.25);ax.legend(fontsize=8,loc='best')
    fig.savefig(path,dpi=190);plt.close(fig)


def plot_all(aggregates,output):
    draw_curves(aggregates,'blocks','Completed training episodes',
                'Training return: mean of 256 completed episodes (MADDPG-style)',output/'training_reward_mean256.png')
    draw_curves(aggregates,'epymarl','Completed training episodes',
                'Training return: moving average 200 episodes (EPyMARL-style)',output/'training_reward_moving200.png')
    draw_curves(aggregates,'stepbins','Environment transitions (millions)',
                'Training return: episodes completed per 0.25M transitions (diagnostic)',output/'training_reward_stepbins.png')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='outputs/reward-method-study')
    parser.add_argument('--key-file',default='/home/aduy/Documents/NCKH/uav_network/key.txt')
    parser.add_argument('--project',default='truongdaoanhduy-uit/uav_search_target')
    parser.add_argument('--run',action='append',metavar='LABEL=RUN_ID',
                        help='Repeat to override the four default run IDs')
    args=parser.parse_args()
    from uav_network.configuration import load_config
    from uav_network.experiments.dashboard import _authentication
    import wandb
    profile=load_config('configs/comparison-independent-35t25o-h512-xyz-sensing.yaml',
                        [f'tracking.credentials_file={args.key_file}'])
    entity,project=_authentication(profile)
    api=wandb.Api(timeout=35)
    run_choices=dict(RUNS)
    if args.run:
        run_choices={}
        for item in args.run:
            if '=' not in item:
                parser.error('--run must be LABEL=RUN_ID')
            label,run_id=item.split('=',1)
            if not label.strip() or not run_id.strip():
                parser.error('--run must be LABEL=RUN_ID')
            run_choices[label.strip()]=run_id.strip()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    result={}; worlds={}
    for label,run_id in run_choices.items():
        cache=output/f'raw_{run_id}.npz'
        if cache.exists():
            with np.load(cache) as npz:
                rows=npz['rows']; world=tuple(npz['world'].tolist())
            print('CACHE',label,run_id,len(rows),flush=True)
        else:
            run,rows,world=fetch_history(api,args.project,run_id)
            np.savez_compressed(cache,rows=rows,world=np.array(world))
            print('FETCHED',label,run_id,len(rows),world,flush=True)
        worlds[run_id]=world
        result[label]={'run_id':run_id,'world':list(world),'observed_episodes':len(rows),
                       'blocks':fixed_episode_blocks(rows[:,0],rows[:,1],rows[:,3]),
                       'epymarl':epymarl_rolling_mean(rows[:,0],rows[:,1]),
                       'stepbins':transition_bin_means(rows[:,3],rows[:,1])}
        print('ANALYZED',label,len(result[label]['blocks']),len(result[label]['epymarl']),len(result[label]['stepbins']),flush=True)
    if len(set(worlds.values()))!=1:
        raise RuntimeError(f'Incomparable scenario configurations: {worlds}')
    plot_all(result,output)
    (output/'reward_methods_data.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('EXPORT_OK',output,'world',next(iter(worlds.values())),flush=True)
    for key in result:
        a=result[key]
        print('STATS',key,'ep',a['observed_episodes'],'initial_block',round(a['blocks'][0]['reward'],1),
              'last_full_block',round([r for r in a['blocks'] if r['complete']][-1]['reward'],1),flush=True)

if __name__=='__main__':
    main()
