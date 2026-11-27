#!/usr/bin/env python3
"""Exact finite-horizon per-cell DP under conditionally independent Pd/Pf readings.

Not a proof of exact 15-target joint map posterior. Not a measured UAV flight.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

def ds_mass(p:int,n:int,mf:float=.8,me:float=.8)->float:
    if p+n==0:return 0.
    a=p*math.log1p(-mf);b=n*math.log1p(-me)
    s=max(a,b);ea=math.exp(a-s);eb=math.exp(b-s);eu=math.exp(min(a,b))
    t=eb*(-math.expm1(a)) if p else 0.
    f=ea*(-math.expm1(b)) if n else 0.
    return t/(t+f+eu)

def bayesian_prob(p:int,n:int,pd:float=.85,pf:float=.03,prior:float=15/392)->float:
    l=math.log(prior/(1-prior))+p*math.log(pd/pf)+n*math.log((1-pd)/(1-pf))
    l=max(-40.,min(40.,l))
    return 1/(1+math.exp(-l))

def propagate(steps:int,p_pos:float,fn,threshold:float=.90):
    states={(0,0,False):1.0}
    for _ in range(steps):
        new={}
        for (p,n,ever),mass in states.items():
            for z,prob in ((True,p_pos),(False,1-p_pos)):
                pp=p+int(z);nn=n+int(not z)
                key=(pp,nn,ever or fn(pp,nn)>=threshold)
                new[key]=new.get(key,0.)+mass*prob
        states=new
    miss=sum(prob for (p,n,ever),prob in states.items() if fn(p,n)<threshold)
    ever=sum(prob for (p,n,ever),prob in states.items() if ever)
    return miss,ever

def brier(steps,pd,pf,prior,fn):
    total=0.
    for true,base,rate in ((True,prior,pd),(False,1-prior,pf)):
        for p in range(steps+1):
            prob=math.comb(steps,p)*rate**p*(1-rate)**(steps-p)
            post=fn(p,steps-p)
            total+=base*prob*((1-post)**2 if true else post**2)
    return total

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--pd",type=float,default=.85)
    ap.add_argument("--pf",type=float,default=.03)
    ap.add_argument("--threshold",type=float,default=.90)
    ap.add_argument("--legal-cells",type=int,default=392)
    ap.add_argument("--targets",type=int,default=15)
    args=ap.parse_args()
    assert 0<args.pf<args.pd<1 and args.targets<args.legal_cells
    prior=args.targets/args.legal_cells
    ds=lambda p,n:ds_mass(p,n)
    ba=lambda p,n:bayesian_prob(p,n,args.pd,args.pf,prior)
    rows=[]
    for k in (2,5,10,20):
        row={"k":k}
        for name,fn in (("paper_fixed_ds",ds),("independent_cell_bayes",ba)):
            miss,_=propagate(k,args.pd,fn,args.threshold)
            _,everfp=propagate(k,args.pf,fn,args.threshold)
            row[name]={"snapshot_miss_rate":miss,"expected_false_confirmed_ever_cells":everfp*(args.legal_cells-args.targets),
                       "expected_missed_targets":miss*args.targets,
                       "brier_unconditional":brier(k,args.pd,args.pf,prior,fn)}
        rows.append(row)
    output={
       "model":"Bernoulli independent readings per cell; fixed marginal prior only",
       "joint_fixed_count_posterior_exact":False,
       "measured_uav_success":False,
       "independent_temporal_measurements_assumed":True,
       "Pd":args.pd,"Pf":args.pf,"legal_cells":args.legal_cells,
       "targets":args.targets,"prior":prior,"threshold":args.threshold,
       "llr_positive":math.log(args.pd/args.pf),
       "llr_negative":math.log((1-args.pd)/(1-args.pf)),
       "rows":rows,
       "note":"DS target mass is NOT a calibrated posterior; Brier is an informative score only, not proof of probabilistic consistency.",
    }
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
