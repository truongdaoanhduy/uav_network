"""Reaggregate genuine 1-vector-step mission progress from HF .tar.gz archives.

No training needed; all non-overlapping window mean/std values are calculated
from the original per-world count, sum and sum of squares. Missing/duplicate
archive records cause an error rather than misleading interpolated plots.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import tarfile

from .vector_step_mission_progress import METRICS

RAW_NAME="training-step-progress.jsonl"


def read_records(paths):
    last_step=None
    for path in sorted(map(Path,paths)):
        if path.name.endswith('.tar.gz'):
            with tarfile.open(path,'r:gz') as archive:
                if RAW_NAME not in archive.getnames():
                    continue
                with archive.extractfile(RAW_NAME) as stream:
                    lines=list(stream)
        elif path.name.endswith('.jsonl.gz'):
            with gzip.open(path,'rb') as stream:
                lines=stream.readlines()
        elif path.name.endswith('.jsonl'):
            lines=path.read_bytes().splitlines()
        else:
            continue
        for content in lines:
            if not content.strip():continue
            row=json.loads(content)
            step=int(row['global_step'])
            n=int(row['num_envs'])
            if n<1 or step%n:
                raise ValueError(f'Invalid mission progress vector step: {step}')
            if last_step is not None and step!=last_step+n:
                raise ValueError(f'Missing or duplicate vector step: {last_step} -> {step}')
            metrics=row['metrics']
            if set(metrics)!=set(METRICS):
                raise ValueError('Incomplete mission progress metrics: expected all fields')
            for name,values in metrics.items():
                if int(values['count'])!=n:
                    raise ValueError(f'Incorrect population count for {name}')
                if not (math.isfinite(float(values['sum'])) and math.isfinite(float(values['sum_sq']))):
                    raise ValueError(f'Nonfinite mission progress statistics: {name}')
            last_step=step
            yield row


def rebin(rows,window,*,include_tail=False):
    window=int(window)
    if window<1:raise ValueError('window must be a positive number of vector steps')
    buf=[]

    def calc(items,complete):
        result={
            'schema_version':1,
            'global_step':int(items[-1]['global_step']),
            'window_start_global_step':int(items[0]['global_step'])-int(items[0]['num_envs']),
            'window_vector_steps':len(items),
            'requested_vector_steps':window,
            'complete_window':bool(complete),
            'metrics':{},
        }
        for name in METRICS:
            n=sum(int(row['metrics'][name]['count']) for row in items)
            total=math.fsum(float(row['metrics'][name]['sum']) for row in items)
            square=math.fsum(float(row['metrics'][name]['sum_sq']) for row in items)
            mean=total/n
            result['metrics'][name]={
                'count':n,'sum':total,'sum_sq':square,
                'mean':mean,'std':math.sqrt(max(0.,square/n-mean*mean)),
            }
        return result
    for row in rows:
        buf.append(row)
        if len(buf)==window:
            yield calc(buf,True)
            buf=[]
    if include_tail and buf:
        yield calc(buf,False)


def export_rebinned_gz(inputs,output_dir,windows=(1,2,4,8,16),*,include_tail=True):
    data=list(read_records(inputs))
    if not data:
        raise ValueError('No HF per-vector-step mission progress records found')
    target=Path(output_dir)
    target.mkdir(parents=True,exist_ok=True)
    paths=[]
    for width in sorted(set(int(n) for n in windows)):
        destination=target/f'train-step-progress-vector{width}.jsonl.gz'
        with gzip.open(destination,'wt',encoding='utf-8',compresslevel=6) as stream:
            for row in rebin(data,width,include_tail=include_tail):
                stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        paths.append(destination)
    return paths,len(data)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-dir',type=Path)
    p.add_argument('--hf-run-id')
    p.add_argument('--hf-repo',default='duy95/uav-search-diagnostics')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--windows',default='1,2,4,8,16,32')
    args=p.parse_args(argv)
    if bool(args.archive_dir)==bool(args.hf_run_id):
        p.error('Provide exactly one of --archive-dir or --hf-run-id')
    directory=args.archive_dir
    if args.hf_run_id:
        from huggingface_hub import snapshot_download
        root=Path(snapshot_download(
            repo_id=args.hf_repo,repo_type='dataset',
            allow_patterns=[f'runs/{args.hf_run_id}/diagnostic-archives/*.tar.gz'],
            local_dir=args.output_dir/'hf-cache'))
        directory=root/f'runs/{args.hf_run_id}/diagnostic-archives'
    archives=sorted(directory.glob('diagnostics-*.tar.gz'))
    if not archives:raise FileNotFoundError(f'No diagnostic archive parts in {directory}')
    files,count=export_rebinned_gz(archives,args.output_dir,
        windows=[int(x.strip()) for x in args.windows.split(',')])
    print(f'MISSION_PROGRESS_REBIN_OK: {count} exact vector-step records')
    for f in files:print(f)

if __name__=='__main__':main()
