"""Exact absorbing-state probability of ever crossing a DS threshold.
Independent Bernoulli sensor model, single cell, no network/mission claims.
"""
import json
import numpy as np
from pathlib import Path

def passage(q, threshold, horizon=900):
    alive = np.array([1.0])
    hit = 0.0
    result = {}
    for t in range(1, horizon+1):
        a = np.zeros(t+1)
        a[:-1] += alive*(1-q)
        a[1:] += alive*q
        p = np.arange(t+1)
        n = t-p
        # Stable DS for equal positive/negative masses 0.8.
        x, y = -p*np.log(5.), -n*np.log(5.)
        s = np.maximum(x,y)
        tp = np.exp(y-s)*(-np.expm1(x))
        tn = np.exp(x-s)*(-np.expm1(y))
        u = np.exp(np.minimum(x,y))
        denom = tp+tn+u
        confirmed = (tp/denom >= threshold) & (u/denom <= .2)
        hit += float(a[confirmed].sum())
        a[confirmed] = 0
        alive = a
        if t in (2,5,10,20,100,900):
            result[str(t)] = hit
    return result

if __name__ == '__main__':
    result = {str(th): {'false':passage(.03,th),'true':passage(.85,th)}
              for th in (.9,.95,.99,.999)}
    out = Path('outputs/v107/ds_first_passage.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(out.read_text())
