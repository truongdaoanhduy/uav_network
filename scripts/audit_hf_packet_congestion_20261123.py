"""Read-only public HF per-episode congestion audit for completed MATD3 runs.

Do NOT inject keys. This is NOT a matched global-step controlled experiment;
archive part numbers only provide reproducible samples across different run
policies, workloads and training durations. No production code changed.
"""
import argparse,csv,json,statistics
from pathlib import Path
import tarfile
from huggingface_hub import hf_hub_download

REPO='duy95/uav-search-diagnostics'
RUNS=('18d756ef','602a1e82')
PARTS=(25,60,80,100,120)


def stats(rows,key,subsection=None):
    y=[r.get(subsection,{}).get(key) if subsection else r.get(key) for r in rows]
    y=sorted(float(v) for v in y if isinstance(v,(int,float)) and not isinstance(v,bool))
    if not y:return None
    return {'n':len(y),'mean':statistics.mean(y),'median':statistics.median(y),
            'p90':y[int(.9*(len(y)-1))], 'min':y[0],'max':y[-1]}


def load_episode_part(run_id,part):
    filename=f'runs/{run_id}/diagnostic-archives/diagnostics-{part:06}.tar.gz'
    location=hf_hub_download(repo_id=REPO,repo_type='dataset',filename=filename)
    with tarfile.open(location,'r:gz') as archive:
        if 'episodes.jsonl' not in archive.getnames():return []
        return [json.loads(line) for line in archive.extractfile('episodes.jsonl') if line.strip()]


def audit(runs=RUNS,parts=PARTS,min_count=20):
    output=[]
    for runid in runs:
        for part in parts:
            episodes=load_episode_part(runid,part)
            if len(episodes)<min_count:continue
            row={'run_id':runid,'hf_part':part,'episodes':len(episodes),
                 'source':f'https://huggingface.co/datasets/{REPO}/blob/main/runs/{runid}/diagnostic-archives/diagnostics-{part:06}.tar.gz'}
            for key in ('generated_packets','received_packets','collisions'):
                result=stats(episodes,key,'network')
                if result:
                    row[key+'_mean']=result['mean'];row[key+'_median']=result['median']
                    row[key+'_p90']=result['p90'];row[key+'_n']=result['n']
            for key in ('targets_delivered','total_hypotheses','false_confirmed_cells','dropped_reports','regenerated_reports'):
                result=stats(episodes,key)
                if result:
                    row[key+'_mean']=result['mean'];row[key+'_median']=result['median']
            output.append(row)
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='/home/aduy/Downloads/MATD3_18d_VS_602a_HF_PACKET_CONGESTION_20261123')
    args=p.parse_args()
    data=audit();assert len(data)>=6,'Insufficient public HF samples'
    root=Path(args.output)
    root.parent.mkdir(parents=True,exist_ok=True)
    jsonfile=root.with_suffix('.json');csvfile=root.with_suffix('.csv')
    jsonfile.write_text(json.dumps({'kind':'read_only_unmatched_hf_per_episode_traffic_sampling',
        'runs':RUNS,'archive_parts':PARTS,'no_matched_global_steps':True,
        'rows':data},indent=2,allow_nan=False))
    columns=sorted(set().union(*(set(row.keys()) for row in data)))
    with csvfile.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(data)
    print('CONGESTION_AUDIT_ROWS',len(data),'JSON',jsonfile,'CSV',csvfile)
    for row in data:
        print(row['run_id'],row['hf_part'],'episodes',row['episodes'],
              'mean_packets',round(row.get('generated_packets_mean',0),2),
              'mean_collisions',round(row.get('collisions_mean',0),2),
              'mean_false',round(row['false_confirmed_cells_mean'],2) if 'false_confirmed_cells_mean' in row else 'not_logged')


if __name__=='__main__':main()
