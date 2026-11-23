"""Exact scenario CPU end-to-end reproducibility; no tokens or Kaggle GPUs."""
from pathlib import Path
import subprocess,sys,os,json


def test_bounded_map1000_cpu_fullflow_train_validate_select_heldout_test(tmp_path):
    out=tmp_path/'cpu_fullflow'
    proc=subprocess.run([sys.executable,'-m','scripts.audit_map1000_tx06_cpu_fullflow_20261123',
                         '--output',str(out)],
        env={**os.environ,'TORCHINDUCTOR_COMPILE_THREADS':'1','OMP_NUM_THREADS':'1',
             'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
        capture_output=True,text=True,timeout=75)
    assert proc.returncode==0,(proc.stdout[-1800:],proc.stderr[-1900:])
    assert 'MAP1000_TX06_REAL_CPU_FULLFLOW_OK' in proc.stdout
    result=json.loads((out/'fullflow_report.json').read_text())
    assert result['map_m']==[1000,1000] and result['tx_power_cap_w']==.6
    assert result['train_episodes']==6 and result['heldout_test_episodes']==2
    assert result['validation_scenarios_per_snapshot']==2
    assert result['validation_checkpoint_count']>=2
    assert (out/'test'/'metrics.json').is_file()
    assert all(Path(z['checkpoint']).is_file() for z in result['validation'])
    # Fresh directory required to avoid overwriting previous user-run evidence.
    second=subprocess.run([sys.executable,'-m','scripts.audit_map1000_tx06_cpu_fullflow_20261123',
                         '--output',str(out)],capture_output=True,text=True,timeout=15)
    assert second.returncode!=0
    assert 'refusing to overwrite' in second.stderr
