"""Re-bin real per-vector optimizer objectives from HF compressed diagnostics.

Every contribution is a *completed optimizer update*, not an environment
transition or an invented actor update. The mean/SD for any vector-step window
are weighted by the observed critic/actor/alpha update count respectively.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import tarfile

from .vector_step_optimizer_loss import METRICS

RAW_NAME='training-step-optimizer.jsonl'


def read_records(paths):
    previous=None
    for path in sorted(map(Path,paths)):
        if path.name.endswith('.tar.gz'):
            with tarfile.open(path,'r:gz') as tar:
                if RAW_NAME not in tar.getnames():continue
                with tar.extractfile(RAW_NAME) as f:
                    lines=list(f)
        elif path.name.endswith('.jsonl.gz'):
            with gzip.open(path,'rb') as f:lines=f.readlines()
        elif path.name.endswith('.jsonl'):
            lines=path.read_bytes().splitlines()
        else:continue
        for line in lines:
            if not line.strip():continue
            row=json.loads(line)
            step=int(row['global_step']);num_envs=int(row['num_envs'])
            if num_envs<1 or step%num_envs:
                raise ValueError('Optimizer step not aligned with full vector step')
            if previous is not None and step!=previous+num_envs:
                raise ValueError(f'Missing or duplicate optimizer vector step: {previous} -> {step}')
            metrics=row['metrics']
            if set(metrics)!=set(METRICS):
                raise ValueError('Missing original optimizer diagnostic fields')
            for name,item in metrics.items():
                if int(item['count'])<0 or not all(
                    math.isfinite(float(item[key])) for key in ('sum','sum_sq')):
                    raise ValueError(f'Invalid optimizer sufficient statistics: {name}')
                if int(item['count'])==0 and (item['sum']!=0 or item['sum_sq']!=0):
                    raise ValueError(f'Missing optimizer update must have zero sums: {name}')
            previous=step
            yield row


def rebin(rows,width,*,include_tail=False):
    width=int(width)
    if width<1:raise ValueError('Need positive vector-step window')
    buf=[]

    def calculate(group,full):
        result={
            'schema_version':1,
            'global_step':int(group[-1]['global_step']),
            'window_start_global_step':int(group[0]['global_step'])-int(group[0]['num_envs']),
            'window_vector_steps':len(group),
            'requested_vector_steps':width,
            'complete_window':bool(full),
            'metrics':{},
        }
        for name in METRICS:
            values=[row['metrics'][name] for row in group]
            count=sum(int(x['count']) for x in values)
            total=math.fsum(float(x['sum']) for x in values)
            squares=math.fsum(float(x['sum_sq']) for x in values)
            result['metrics'][name]={
                'count':count,'sum':total,'sum_sq':squares,
                'mean':total/count if count else None,
                'std':math.sqrt(max(0.,squares/count-(total/count)**2)) if count else None,
            }
        return result

    for row in rows:
        buf.append(row)
        if len(buf)==width:
            yield calculate(buf,True)
            buf=[]
    if include_tail and buf:
        yield calculate(buf,False)


def export_rebinned_gz(inputs,output_dir,windows=(1,2,4,8,16),*,include_tail=True):
    rows=list(read_records(inputs))
    if not rows:raise ValueError('No real optimizer vector-step journal records')
    folder=Path(output_dir)
    folder.mkdir(parents=True,exist_ok=True)
    result=[]
    for width in sorted(set(int(x) for x in windows)):
        path=folder/f'train-step-optimizer-vector{width}.jsonl.gz'
        with gzip.open(path,'wt',encoding='utf-8',compresslevel=6) as out:
            for row in rebin(rows,width,include_tail=include_tail):
                out.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
        result.append(path)
    return result,len(rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-dir',type=Path)
    p.add_argument('--hf-run-id')
    p.add_argument('--hf-repo',default='duy95/uav-search-diagnostics')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--windows',default='1,2,4,8,16,32')
    args=p.parse_args(argv)
    if bool(args.archive_dir)==bool(args.hf_run_id):
        p.error('Give exactly --archive-dir or --hf-run-id')
    directory=args.archive_dir
    if args.hf_run_id:
        from huggingface_hub import snapshot_download
        root=Path(snapshot_download(
            repo_id=args.hf_repo,repo_type='dataset',
            allow_patterns=[f'runs/{args.hf_run_id}/diagnostic-archives/*.tar.gz'],
            local_dir=args.output_dir/'hf-cache'))
        directory=root/f'runs/{args.hf_run_id}/diagnostic-archives'
    parts=sorted(directory.glob('diagnostics-*.tar.gz'))
    if not parts:raise FileNotFoundError(f'No diagnostic archive parts in {directory}')
    files,count=export_rebinned_gz(parts,args.output_dir,
        windows=[int(n) for n in args.windows.split(',')])
    print(f'OPTIMIZER_REBIN_OK: {count} real vector-step records')
    for file in files:print(file)

if __name__=='__main__':main()
