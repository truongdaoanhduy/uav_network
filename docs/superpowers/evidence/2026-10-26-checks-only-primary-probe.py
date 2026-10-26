"""Bounded checks of the unchanged primary 1KiB hybrid4096 profile; no tracking."""
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime,timezone
import gc,hashlib,json,time,unittest
import torch
import test_cuda_exact_resume as existing
from uav_network.configuration import load_config
from uav_network.experiments.training import train

PROFILE='configs/comparison-20k-hybrid75-1kb.yaml'
OUT=Path('/kaggle/working/primary-full4096-result.json')
records=[]
class WorldChecks(existing.ResumeChecks,unittest.TestCase):
    def runTest(self):pass

def save():
    OUT.write_text(json.dumps({'scope':'Primary1KiB unchanged env4096/batch8192/replay1M/CPU75%; warmup two bounded segments, not20k learning','profile_sha256':hashlib.sha256(Path(PROFILE).read_bytes()).hexdigest(),'utc':datetime.now(timezone.utc).isoformat(),'checks':records,'status':'pass' if len(records)==6 and all(r['status']=='pass' for r in records) else 'in_progress'},indent=2))

def checkpoint_details(path,retain_replay=False):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    cfg=saved['config'];assert cfg['world']['generation_version']==2 and saved['observation_version']==3
    rep=saved['training']['replay'];env=saved['training']['environment']
    assert rep['layout']=='hybrid' and rep['capacity']==1000000 and rep['cpu_capacity']==750000
    assert rep['hot_device_type']=='cuda'
    torch.testing.assert_close(env['world']['arrays']['steps'].to(torch.int64),env['lengths'],rtol=0,atol=0)
    result={'lengths':env['lengths'].clone(),'seeds':env['seeds'].clone(),'next_seed':env['next_seed'],'progress':saved['progress'],'updates':saved['updates'],'index':rep['index'],'size':rep['size'],'rng':rep['rng'].clone(),'world_bytes':existing.world_bytes(env['world'])}
    if retain_replay:
        result['cpu_data']={k:v.clone() for k,v in rep['cpu_data'].items()}
        result['hot_data']={k:v.clone() for k,v in rep['hot_data'].items()}
    else:
        result['cpu_data']=rep['cpu_data']
        result['hot_data']=rep['hot_data']
    return result

assert torch.cuda.is_available() and torch.cuda.device_count()==2
def primary_world_roundtrip(device):
    from uav_network.environment.gpu.vector import WarpVectorEnv
    from exact_resume_helpers import assert_current_world
    c=load_config(PROFILE);c.tracking.enabled=False
    vector=None;case=WorldChecks()
    with TemporaryDirectory(prefix='uav-primary-world-') as directory:
        try:
            vector=WarpVectorEnv(c,device=device);obs,_=vector.reset(seed=44)
            for _ in range(3):obs,*_=vector.step(vars(vector.random_actions(obs['observation'])))
            start=Path(directory)/'start.pt';torch.save(vector.snapshot_state(),start)
            action=vector.random_actions(obs['observation'])
            actions=(action.continuous.cpu().clone(),action.destination.cpu().clone())
            _,reward,done,truncated,_=vector.step(vars(action))
            expected_result=tuple(x.cpu().clone() for x in (reward,done,truncated))
            expected=Path(directory)/'next.pt';torch.save(vector.snapshot_state(),expected)
            del action,obs;vector.close();vector=None;existing.release_cuda()
            vector=WarpVectorEnv(c,device=device);vector.reset(seed=144)
            snapshot=torch.load(start,map_location='cpu',weights_only=True)
            obs=vector.restore_state(snapshot);assert_current_world(case,vector,snapshot)
            del snapshot;gc.collect()
            action=vector.random_actions(obs['observation'])
            for actual,reference in zip((action.continuous.cpu(),action.destination.cpu()),actions):torch.testing.assert_close(actual,reference,rtol=0,atol=0)
            _,reward,done,truncated,_=vector.step(vars(action))
            for actual,reference in zip((reward.cpu(),done.cpu(),truncated.cpu()),expected_result):torch.testing.assert_close(actual,reference,rtol=0,atol=0)
            snapshot=torch.load(expected,map_location='cpu',weights_only=True);assert_current_world(case,vector,snapshot)
            torch.testing.assert_close(vector.action_generator.get_state().cpu(),snapshot['action_generator'],rtol=0,atol=0)
            del snapshot
        finally:
            if vector is not None:vector.close()

for device in ['cuda:0','cuda:1']:
    started=time.monotonic();record={'kind':'serialized4096world-and-next-step','device':device,'status':'running'};records.append(record);save()
    try:
        primary_world_roundtrip(device)
        record.update(status='pass',seconds=time.monotonic()-started)
    except BaseException as e:
        record.update(status='failed',error_type=type(e).__name__,error=str(e));save();raise
    finally:existing.release_cuda()
    save()

for algo,pql in [('masac',False),('masac',True),('matd3',False),('matd3',True)]:
    started=time.monotonic();record={'kind':'primary-hybrid-exact-resume','algorithm':algo,'pql':pql,'status':'running'};records.append(record);save()
    try:
        with TemporaryDirectory(prefix='uav-primary4096-') as directory:
            c=load_config(PROFILE);c.runtime.algorithm=algo;c.pql.enabled=pql
            assert (c.runtime.num_envs,c.algorithm_config.batch_size,c.algorithm_config.replay_capacity,c.runtime.replay_cpu_fraction)==(4096,8192,1000000,.75)
            assert (c.network.report_bytes,c.network.payload_bytes,c.network.buffer_bytes)==(1024,1024,3072)
            assert c.runtime.amp_mode=='off' and not c.runtime.compile_models and not c.runtime.fused_optimizers
            c.tracking.enabled=False;c.runtime.max_train_seconds=1e-9;c.runtime.output_dir=str(Path(directory)/'first')
            first=train(c)
            assert first['episodes']==0 and first['global_step']==4096 and first['updates']==0
            assert first['environment_state_saved'] is True
            old=checkpoint_details(first['resume_checkpoint'],True)
            assert old['size']==old['index']==4096 and old['progress']['records']==[]
            torch.testing.assert_close(old['lengths'],torch.ones_like(old['lengths']),rtol=0,atol=0)
            torch.testing.assert_close(old['seeds'],torch.arange(44,44+4096),rtol=0,atol=0)
            existing.release_cuda()
            c.runtime.output_dir=str(Path(directory)/'second');c.runtime.warm_start_checkpoint=first['resume_checkpoint']
            second=train(c)
            assert second['state_restore']=='training_state_exact_environment' and second['optimizer_replay_restored'] is True
            assert second['initial_global_step']==4096 and second['global_step']==8192
            assert second['initial_episodes']==second['episodes']==second['segment_episodes']==0
            assert second['updates']==0 and second['stop_reason']=='time_budget'
            assert second['environment_state_saved'] is True and second['collection_steps']==first['collection_steps']+1
            current=checkpoint_details(second['resume_checkpoint'])
            assert current['size']==current['index']==8192
            assert current['next_seed']==old['next_seed'] and current['progress']['records']==old['progress']['records']
            torch.testing.assert_close(current['seeds'],old['seeds'],rtol=0,atol=0)
            torch.testing.assert_close(current['lengths'],old['lengths']+1,rtol=0,atol=0)
            torch.testing.assert_close(current['rng'],old['rng'],rtol=0,atol=0)
            for field,value in old['cpu_data'].items():torch.testing.assert_close(current['cpu_data'][field][:4096],value,rtol=0,atol=0)
            for field,value in old['hot_data'].items():torch.testing.assert_close(current['hot_data'][field],value,rtol=0,atol=0)
            learner_role='critic' if pql else 'learner'
            for r in [first,second]:
                assert r['devices']['collector']=='cuda:0' and r['devices'][learner_role]=='cuda:1'
                assert r['replay']['replay/hybrid']==1
                assert r['num_cpu_env_workers']==0
            record.update(status='pass',seconds=time.monotonic()-started,global_step_before=4096,global_step_after=8192,replay_rows_preserved=4096,checkpoint_bytes=Path(second['resume_checkpoint']).stat().st_size,world_array_bytes=old['world_bytes'],devices=second['devices'],replay_cpu_bytes=second['replay']['replay/cpu_bytes'],replay_hot_bytes=second['replay']['replay/hot_bytes'])
            del old,current,first,second
    except BaseException as e:
        record.update(status='failed',error_type=type(e).__name__,error=str(e));save();raise
    finally:existing.release_cuda();gc.collect()
    save()
print('PRIMARY_FULL4096_SIX_CHECKS_PASS',flush=True)
