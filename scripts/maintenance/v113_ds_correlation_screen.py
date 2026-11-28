#!/usr/bin/env python3
"""V113 DS-only sensor temporal-correlation robustness screen (not a mission gate).

Stress-model assumption (NOT claimed from Zhang or Denoeux): when revisiting
one static map cell, with probability rho reuse the last binary reading, else
sample a fresh Bernoulli observation with Pd/Pf. Marginal Bernoulli accuracy is
therefore unchanged; consecutive errors become correlated. This probe exercises
exact V103/Zhang-style equal-mass count DS thresholds, NOT PHY, route, FoV,
mission-level GCS aggregation, or a real-world-calibrated sensor model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if not (0 <= k <= n and n > 0):
        raise ValueError('0<=k<=n required')
    p = k / n
    d = 1.0 + z*z/n
    center = (p + z*z/(2*n))/d
    half = (z/d)*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return [max(0., center-half), min(1., center+half)]


def simulate(*, n_trials: int, visits: int, seed: int, rho: float,
             probability: float, thresholds: tuple[float,...], mf: float = 0.8,
             me: float = 0.8) -> dict:
    if not (0. <= rho < 1. and 0. < probability < 1. and 0. < mf < 1. and 0. < me < 1.):
        raise ValueError('rho, probability, masses outside allowed range')
    if n_trials < 1 or visits < 1 or not all(0. < v < 1. for v in thresholds):
        raise ValueError('bad trials, visits, thresholds')
    rng = np.random.default_rng(seed)
    prev = np.zeros(n_trials, dtype=np.bool_)
    p = np.zeros(n_trials, dtype=np.int32)
    n = np.zeros(n_trials, dtype=np.int32)
    first = {th: np.zeros(n_trials, dtype=np.int16) for th in thresholds}
    for t in range(1, visits+1):
        fresh = rng.random(n_trials) < probability
        # Visit 1 always fresh; rho memory persists between visits of same cell.
        measured = fresh if t == 1 else np.where(rng.random(n_trials) < rho, prev, fresh)
        prev = measured
        p += measured
        n += ~measured
        # Exact Dempster mass for finite positive/negative evidence counts;
        # all computations here are for <= 50 visits with no underflow.
        a = np.exp(np.log1p(-mf) * p)
        b = np.exp(np.log1p(-me) * n)
        denom = a + b - a*b
        mt = (1.0-a)*b/denom
        mu = a*b/denom
        for th in thresholds:
            new = (first[th] == 0) & (mt >= th) & (mu <= .2)
            first[th][new] = t
    outputs={}
    for th, times in first.items():
        count=int(np.count_nonzero(times))
        outputs[str(th)]={
            'ever_confirmed':count,
            'ever_confirmed_rate':count/n_trials,
            'wilson_ci95':wilson_ci(count,n_trials),
            'first_confirmation_mean_visit_conditional':float(times[times>0].mean()) if count else None,
        }
    return {'n_trials':n_trials,'visits':visits,'seed':seed,'rho':rho,
            'sensor_positive_probability':probability,'results':outputs}


def run(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=100_000)
    parser.add_argument('--visits',type=int,default=20)
    parser.add_argument('--seed',type=int,default=44)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    result={
       'gate':'V113_DS_TEMPORAL_DEPENDENCE_ROBUSTNESS_SCREEN',
       'status':'DIAGNOSTIC_ONLY_NOT_FULL_MISSION',
       'scenario_frozen':False,
       'method':'sticky Markov Bernoulli with unchanged marginal Pd/Pf; synthetic rho (NOT sourced numerical model)',
       'paper_method':'Zhang 2024 DS evidence and count-based fusion; dependence caution Denoeux 2008',
       'assumptions':{'pd':.85,'pf':.03,'mf':.8,'me':.8,'uncertainty_ceiling':.2,'rho':[0.,.5,.9],
                      'confirm_thresholds':[.9,.99],'static_target':True,'prior_bba':[0.,0.,1.]},
       'sweeps':{},
       'limitations':['Single-cell repeated visits, not 6-UAV full mission',
                      'No GCS network/packet simulation or source-origin correlation',
                      'Not empirical sensor calibration',
                      'No end-to-end false-positive field validation'],
    }
    for prob_name,prob in [('empty_cell_pf',.03),('true_target_pd',.85)]:
        result['sweeps'][prob_name]={}
        for i,rho in enumerate((0.,.5,.9)):
            result['sweeps'][prob_name][str(rho)]=simulate(
                n_trials=args.trials,visits=args.visits,seed=args.seed+i,
                rho=rho,probability=prob,thresholds=(.9,.99),
            )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name, cells in result['sweeps'].items():
        for rho,r in cells.items():
            print(name,'rho',rho,'ever_confirmed_0.9',r['results']['0.9']['ever_confirmed_rate'],
                  'ever_confirmed_0.99',r['results']['0.99']['ever_confirmed_rate'])
    return 0

if __name__=='__main__':raise SystemExit(run())
