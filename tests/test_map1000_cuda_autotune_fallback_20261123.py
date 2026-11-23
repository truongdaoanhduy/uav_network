"""Execute real notebook autotune control-flow using synthetic GPU measurements.

Guard that SHA mismatches, errors, or <5% gains never alter runtime training.
This does not claim an actual T4 CUDA performance measurement.
"""
from pathlib import Path
import json,os,sys,types
import pytest
from scripts.build_map1000_tx06_autotune_fullflow_ui_20261123 import build_notebook
from scripts.build_nooracle_map1000_tx06_fastgrid_ui_20261123 import CONFIG


def execute_autotune(tmp_path,monkeypatch,scenario):
    nb=build_notebook()
    script=nb.cells[2].source
    start=script.index('# Conservative T4x2 speed autotune:')
    end=script.index('spec_file = Path("/kaggle/working/job.json")')
    snippet=script[start:end]
    root=tmp_path/'embedded_source';(root/'scripts').mkdir(parents=True)
    evidence=tmp_path/'autotune_result.json'
    snippet=snippet.replace('/kaggle/working/MAP1000_TX06_AUTOTUNE_RESULT.json',str(evidence))
    calls=[]
    def run(argv,*,cwd,env,capture_output,text,timeout):
        assert Path(cwd)==root
        assert env['PYTHONPATH']==str(root)
        assert not any(k in env for k in ('WANDB_API_KEY','HF_TOKEN','HUGGINGFACE_HUB_TOKEN'))
        marker=None
        if 'scripts.profile_nooracle_warp_phases_20261122' in argv:
            dim=int(argv[argv.index('--mac-block-dim')+1])
            if scenario=='error' and dim==0:
                raise RuntimeError('synthetic CUDA baseline failure')
            is_parity='--state-digest' in argv
            sha='match' if scenario!='mismatch' or dim!=32 else 'mismatch'
            speed={0:2000.0,32:2420.0,64:2100.0,128:1900.0}[dim]
            if scenario=='slow':speed={0:2000.0,32:2005.0,64:1990.0,128:1800.0}[dim]
            payload={'world_state_sha256':sha,'env_transitions_per_second':speed,
                     'per_kernel_ms_total':{'packet_step':3.0}}
            marker='UAV_WARP_PROFILE_JSON'
        else:
            assert 'scripts.benchmark_hybrid_replay_cuda_ab_20261123' in argv
            mode=argv[argv.index('--staging')+1]
            seconds=.1 if mode=='on' else .15
            if scenario=='slow':seconds=.20 if mode=='on' else .15
            sha='same' if not(scenario=='mismatch' and mode=='on') else 'wrong'
            payload={'median_sample_seconds':seconds,'all_sampled_tensor_sha256':sha,
                     'rng_state_sha256':'R','seen_bitmap_sha256':'S'}
            marker='HYBRID_REPLAY_CUDA_AB_JSON'
        calls.append((marker,argv))
        return types.SimpleNamespace(returncode=0,stdout=marker+' '+json.dumps(payload)+'\n',stderr='')
    monkeypatch.delenv('UAV_WARP_MAC_BLOCK_DIM',raising=False)
    monkeypatch.delenv('UAV_HYBRID_PINNED_STAGING',raising=False)
    namespace={'root':root,'python':sys.executable,'os':os,'subprocess':types.SimpleNamespace(run=run),
      'json':json,'SPEC':{'config':CONFIG,'source_bundle_sha256':'fake-sha',
                        'runs':{'matd3':{'name':'MATD3-MAP1000-TX06'}}},
      'AUTO_TUNE_MAC_REPLAY':True,'MAC_TUNE_CANDIDATES':[0,32,64,128],
      'MAC_TUNE_PARITY_ENVS':16,'MAC_TUNE_PARITY_STEPS':48,
      'MAC_TUNE_ENVS':4096,'MAC_TUNE_WARMUP':6,'MAC_TUNE_STEPS':48,
      'MAC_TUNE_REPEATS':2}
    exec(compile(snippet,'autotune_injected_notebook_source.py','exec'),namespace)
    r=json.loads(evidence.read_text())
    assert r['completed'] is True
    assert r['baseline_unchanged'] is True
    assert r['science_config']==CONFIG
    assert namespace['SPEC']['config']==CONFIG
    assert namespace['SPEC']['gpu_hardware_autotune']['full_report']==str(evidence)
    assert 'GPUAUTOTUNED' in namespace['SPEC']['runs']['matd3']['name']
    return r,os.environ['UAV_WARP_MAC_BLOCK_DIM'],os.environ['UAV_HYBRID_PINNED_STAGING'],calls


def test_autotune_uses_only_parity_verified_5pct_better_candidates(tmp_path,monkeypatch):
    r,mac,pin,calls=execute_autotune(tmp_path,monkeypatch,'fast')
    assert (mac,pin)==('32','1')
    assert r['chosen_mac_block_dim']==32
    assert r['chosen_pinned_replay_staging'] is True
    assert len(r['mac_parity'])==4 and len(r['mac_trials'])==8
    assert len(r['replay_trials'])==4


@pytest.mark.parametrize('scenario',['mismatch','slow','error'])
def test_autotune_never_promotes_mismatched_slow_or_failed_configuration(tmp_path,monkeypatch,scenario):
    r,mac,pin,_=execute_autotune(tmp_path,monkeypatch,scenario)
    if scenario=='mismatch':
        assert mac=='0' or mac=='64' # exact +5% must not be promoted
        assert pin=='0'
        assert any(rec['dim']==32 and not rec['matched'] for rec in r['mac_parity'])
    elif scenario=='slow':
        assert (mac,pin)==('0','0')
    else:
        assert mac=='0' # baseline GPU failure => stock fallback
        assert pin=='1' # independent replay A/B remains available
        assert any(e['phase']=='mac_baseline' for e in r['errors'])
