"""Small-window Reward previews for user-selected W&B MASAC BASE and MATD3 BASE.

Training data only: every episode/index maps to an in-sample episode return;
no fixed-policy evaluation, PQL or final-test data is mixed in.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.plot_training_reward_from_wandb import fetch_history, fixed_episode_blocks

# The explicitly user-selected BASE runs, verified online on 2026-11-08.
# MATD3 default was specifically requested instead of newer Kaiming.
RUNS = {
    'MATD3 BASE (default)': '971f7cc2',
    'MASAC BASE (mask-aware)': 'f6a196a7',
}
WINDOWS = (8, 16, 32, 64)


def moving_average(episode, reward, width):
    """Return the mean episode and reward of each valid consecutive window."""
    xs = np.asarray(episode,dtype=float)
    ys = np.asarray(reward,dtype=float)
    if width <= 0:
        raise ValueError('width must be positive')
    if len(ys) < width:
        return np.empty(0),np.empty(0)
    kernel = np.ones(width,dtype=float)/width
    return np.convolve(xs,kernel,mode='valid'), np.convolve(ys,kernel,mode='valid')


def breakpoints(rows, width, label):
    episode, reward, step = rows[:,0], rows[:,1], rows[:,3]
    if label == 'block':
        blocks=fixed_episode_blocks(episode,reward,step,width=width)
        return np.array([r['episode'] for r in blocks if r['complete']]), np.array([r['reward'] for r in blocks if r['complete']])
    return moving_average(episode,reward,width)


def plot_mosaic(data, method, output):
    fig, axes=plt.subplots(2,2,figsize=(15,9),sharex=True,sharey=True,layout='constrained')
    for ax,width in zip(axes.flat,WINDOWS):
        for name,rows in data.items():
            x,y=breakpoints(rows,width,method)
            ax.plot(x,y,linewidth=1.35,alpha=.88,label=name)
        ax.set_title(f'{"Non-overlap block mean" if method=="block" else "Moving average"}: {width} completed episodes')
        ax.grid(alpha=.2);ax.set_xlim(0,20000)
        ax.axvline(4096,ls=':',lw=.75,color='.65')
        ax.axvline(8192,ls=':',lw=.75,color='.65')
        ax.axvline(12288,ls=':',lw=.75,color='.65')
        ax.axvline(16384,ls=':',lw=.75,color='.65')
        ax.legend(fontsize=8)
    for ax in axes[-1,:]:ax.set_xlabel('Completed training episodes')
    for ax in axes[:,0]:ax.set_ylabel('Mean training episode return')
    fig.suptitle(f'User-selected W&B BASE runs — {method} reward, small-window comparison (no PQL)',fontsize=14)
    fig.savefig(output,dpi=180);plt.close(fig)


def plot_window(data, method, width, output):
    fig,ax=plt.subplots(figsize=(12,5.9),layout='constrained')
    for name,rows in data.items():
        x,y=breakpoints(rows,width,method)
        ax.plot(x,y,lw=1.25,alpha=.9,label=name)
    for boundary in (4096,8192,12288,16384):
        ax.axvline(boundary,ls=':',lw=.8,color='.60',alpha=.7)
    ax.set_xlabel('Completed training episodes')
    ax.set_ylabel('Mean training episode return')
    ax.set_xlim(0,20000);ax.grid(alpha=.23)
    ax.legend(loc='best');ax.set_title(f'Selected BASE training Reward — {method} {width}-episode window (no PQL)')
    fig.savefig(output,dpi=190);plt.close(fig)


def plot_boundary_zoom(data, output):
    fig,axes=plt.subplots(1,2,figsize=(14,5.8),layout='constrained',sharey=True)
    for ax,(name,rows) in zip(axes,data.items()):
        ep,reward=rows[:,0],rows[:,1]
        sel=(ep>=3800)&(ep<=4450)
        ax.plot(ep[sel],reward[sel],lw=.7,alpha=.15,color='gray',label='Raw episode return')
        for width in WINDOWS:
            x,y=moving_average(ep,reward,width)
            use=(x>=3800)&(x<=4450)
            ax.plot(x[use],y[use],lw=1.5,label=f'MA {width}')
        ax.axvline(4096,ls='--',lw=1.3,color='black',alpha=.6,label='4,096 episode boundary')
        ax.set_title(name);ax.set_xlabel('Completed training episodes')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0].set_ylabel('Training episode return')
    fig.suptitle('Boundary zoom: can 8/16/32/64 smoothing remove the apparent jump?',fontsize=13)
    fig.savefig(output,dpi=190);plt.close(fig)


def evaluate_jumps(data):
    out={}
    for name,rows in data.items():
        ep,reward=rows[:,0],rows[:,1]
        out[name]={}
        for bound in (4096,8192,12288,16384):
            before=reward[(ep>bound-64)&(ep<=bound)]
            after=reward[(ep>bound)&(ep<=bound+64)]
            if len(before)<32 or len(after)<32:continue
            out[name][str(bound)]={'before64_mean':float(np.mean(before)),
                                   'after64_mean':float(np.mean(after)),
                                   'delta':float(np.mean(after)-np.mean(before))}
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='outputs/reward-small-window-default-matd3-maskaware-masac')
    parser.add_argument('--key-file',default='/home/aduy/Documents/NCKH/uav_network/key.txt')
    parser.add_argument('--project',default='truongdaoanhduy-uit/uav_search_target')
    parser.add_argument('--matd3-id',default=RUNS['MATD3 BASE (default)'])
    parser.add_argument('--masac-id',default=RUNS['MASAC BASE (mask-aware)'])
    args=parser.parse_args()
    from uav_network.configuration import load_config
    from uav_network.experiments.dashboard import _authentication
    import wandb
    c=load_config('configs/comparison-independent-35t25o-h512-xyz-sensing.yaml',
                  [f'tracking.credentials_file={args.key_file}'])
    _authentication(c)  # Explicit key file, avoids requiring user to paste token.
    api=wandb.Api(timeout=35)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    choices={'MATD3 BASE (default)':args.matd3_id,'MASAC BASE (mask-aware)':args.masac_id}
    data={};provenance={};worlds={}
    for label,run_id in choices.items():
        cache=out/f'raw_{run_id}.npz'
        if cache.exists():
            with np.load(cache) as z:
                rows=z['rows'];world=tuple(z['world'].tolist())
            run=api.run(f'{args.project}/{run_id}')
            print('CACHE',label,run_id,len(rows),flush=True)
        else:
            run,rows,world=fetch_history(api,args.project,run_id)
            np.savez_compressed(cache,rows=rows,world=np.array(world))
            print('FETCHED',label,run_id,len(rows),flush=True)
        if bool(run.config.get('pql',{}).get('enabled')):
            raise ValueError(f'{run_id} is PQL; excluded by design')
        if len(rows)<1000:raise ValueError(f'{run_id}: incomplete/benchmark data')
        data[label]=rows;worlds[label]=world
        provenance[label]={'id':run_id,'name':run.name,'created_at':run.created_at,
                           'state':run.state,'episodes':len(rows),'world':list(world),
                           'algorithm':run.config.get('runtime',{}).get('algorithm'),
                           'seed':run.config.get('runtime',{}).get('seed')}
    if len(set(worlds.values()))!=1:
        raise ValueError(f'Mismatched scenarios: {worlds}')
    for method in ('block','ma'):
        plot_mosaic(data,method,out/f'latest_base_{method}_8_16_32_64.png')
        for width in (16,32,64):
            plot_window(data,method,width,out/f'latest_base_{method}_{width}.png')
    plot_boundary_zoom(data,out/'latest_base_boundary_4096_zoom.png')
    info={'provenance':provenance,'windows':list(WINDOWS),'jumps':evaluate_jumps(data)}
    (out/'analysis.json').write_text(json.dumps(info,indent=2,ensure_ascii=False))
    print('RESULT_JSON',json.dumps({'provenance':provenance,'jumps':info['jumps']},ensure_ascii=False),flush=True)
    print('GENERATED',[(p.name,p.stat().st_size) for p in out.glob('*.png')],flush=True)

if __name__=='__main__':main()
