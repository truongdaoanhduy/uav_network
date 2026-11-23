"""Reconstruct exact window means/SDs from public HF gzip diagnostics.

Raw sufficient statistics (count, sum, sumsq) are retained once per
vector step inside training-step-reward.jsonl members of .tar.gz archives.
All output is compressed JSONL for analysis; no fake interpolation.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import tarfile

RAW_NAME='training-step-reward.jsonl'


def read_records(paths):
    last=None
    for path in sorted(map(Path,paths)):
        if path.name.endswith('.tar.gz'):
            with tarfile.open(path,'r:gz') as t:
                if RAW_NAME not in t.getnames():
                    continue
                with t.extractfile(RAW_NAME) as reader:
                    lines=list(reader)
        elif path.name.endswith('.jsonl.gz'):
            with gzip.open(path,'rb') as f:
                lines=f.readlines()
        elif path.name.endswith('.jsonl'):
            lines=path.read_bytes().splitlines()
        else:
            continue
        for line in lines:
            if not line.strip():continue
            row=json.loads(line)
            if not all(key in row for key in ('global_step','count','reward_sum','reward_sum_sq','num_envs')):
                raise ValueError(f'Malformed source row in {path}')
            x=int(row['global_step']);n=int(row['num_envs'])
            if n<=0 or int(row['count'])!=n:
                raise ValueError('Unexpected reward count in a vector step')
            if last is not None and x!=last+n:
                raise ValueError(f'Missing or duplicate vector step: {last} -> {x}; check HF archive completeness')
            last=x
            yield row


def rebin(rows,window,*,include_tail=False):
    import math
    window=int(window)
    if window<1:raise ValueError('window must be >= 1')
    batch=[]
    def make(items,complete):
        count=sum(int(row['count']) for row in items)
        total=math.fsum(float(row['reward_sum']) for row in items)
        squares=math.fsum(float(row['reward_sum_sq']) for row in items)
        mean=total/count
        return {'schema_version':1,'global_step':int(items[-1]['global_step']),
                'window_start_global_step':int(items[0]['global_step'])-int(items[0]['count']),
                'window_vector_steps':len(items),'requested_vector_steps':window,
                'complete_window':bool(complete),'count':count,
                'reward_sum':total,'reward_sum_sq':squares,
                'reward_mean':mean,'reward_std':math.sqrt(max(0.,squares/count-mean*mean))}
    for item in rows:
        batch.append(item)
        if len(batch)==window:
            yield make(batch,True)
            batch=[]
    if include_tail and batch:
        yield make(batch,False)


def export_rebinned_gz(inputs,output_dir,windows=(1,2,4,8,16),*,include_tail=True):
    inputs=list(inputs)
    data=list(read_records(inputs))
    if not data:raise ValueError('No raw vector-step statistics found in provided HF archives')
    target=Path(output_dir)
    target.mkdir(parents=True,exist_ok=True)
    paths=[]
    for interval in sorted(set(int(x) for x in windows)):
        destination=target/f'train-step-reward-mean-vector{interval}.jsonl.gz'
        with gzip.open(destination,'wt',encoding='utf-8',compresslevel=6) as stream:
            for row in rebin(data,interval,include_tail=include_tail):
                stream.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        paths.append(destination)
    return paths,len(data)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir',type=Path,help='Directory containing downloaded diagnostics-*.tar.gz')
    parser.add_argument('--hf-run-id',help='Download public HF diagnostic archives for this W&B run')
    parser.add_argument('--hf-repo',default='duy95/uav-search-diagnostics')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--windows',default='1,2,4,8,16',help='e.g. 1,2,4,8,16')
    args=parser.parse_args(argv)
    if bool(args.archive_dir)==bool(args.hf_run_id):
        parser.error('Choose exactly one of --archive-dir or --hf-run-id')
    directory=args.archive_dir
    if args.hf_run_id:
        from huggingface_hub import snapshot_download
        directory=Path(snapshot_download(repo_id=args.hf_repo,repo_type='dataset',
               allow_patterns=[f'runs/{args.hf_run_id}/diagnostic-archives/*.tar.gz'],
               local_dir=args.output_dir/'hf-cache')) /f'runs/{args.hf_run_id}/diagnostic-archives'
    archives=sorted(directory.rglob('diagnostics-*.tar.gz'))
    if not archives:raise FileNotFoundError(f'No HF diagnostic .tar.gz archives under {directory}')
    exported,raw_count=export_rebinned_gz(archives,args.output_dir,
        windows=[int(x.strip()) for x in args.windows.split(',')])
    print(f'REBIN_SUCCESS: {raw_count} genuine vector-step records')
    for p in exported:print(p)

if __name__=='__main__':main()
