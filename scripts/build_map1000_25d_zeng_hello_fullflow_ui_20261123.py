"""Source-sealed, from-scratch Kaggle T4x2 HELLO 1Hz + Zeng2019 ablation."""
from contextlib import contextmanager
from pathlib import Path
import ast
import hashlib
import nbformat

import scripts.build_map1000_25d_zeng2019_fullflow_ui_20261123 as zeng

ROOT=Path(__file__).resolve().parents[1]
CONFIG='configs/matd3-nooracle-map1000-25d-aoi-zeng2019-hello1hz-fresh2s-20261123.yaml'
NAME='MATD3-MAP1000-25D-ZENG2019-HELLO1HZ-FRESH2S-T4X2-VAL100-TEST1000-20261123.ipynb'
OUT=ROOT/'notebooks'/NAME

@contextmanager
def hello_config_context():
    previous=zeng.CONFIG
    try:
        zeng.CONFIG=CONFIG
        yield
    finally:
        zeng.CONFIG=previous


def build_notebook():
    from uav_network.configuration import load_config
    from uav_network.environment.observations import observation_version
    cfg=load_config(CONFIG)
    assert cfg.world.motion_safety=='reject'
    assert cfg.world.max_vertical_speed_mps==4
    assert cfg.sensor.confirmation_mode=='posterior_only'
    assert cfg.energy.propulsion_model=='zeng2019'
    assert cfg.network.hello_enabled and cfg.network.hello_period_s==1
    assert cfg.network.hello_freshness_s==2 and cfg.network.hello_bits==512
    assert observation_version(cfg)==12
    assert cfg.runtime.episodes==20000 and cfg.runtime.num_envs==4096
    assert cfg.matd3.batch_size==8192 and cfg.matd3.updates_per_step==4
    assert cfg.runtime.warm_start_checkpoint=='none'
    with hello_config_context():
        n=zeng.build_notebook()
    assert len(n.cells)==3
    assert n.metadata['science_config']==CONFIG
    assert n.metadata['energy_model']=='zeng2019'
    old='HELLO beacon is NOT implemented.'
    replacement='HELLO 1Hz over-the-air broadcast IS implemented in Warp (ACK-free beacon with MAC CCA, channel interference and 2s receiver-local freshness), not validated on T4 yet.'
    assert n.cells[0].source.count(old)==1
    n.cells[0].source=n.cells[0].source.replace(old,replacement,1)
    n.cells[0].source+='''

### Additional independently trained HELLO discovery ablation

- Peer radio advertisements carry sender XY/velocity/altitude/battery/free-buffer.
- A receiver can use data only after actual successful SINR+range+LOS decoding
  in the shared Warp MAC event loop. A failed/expired message yields zero peer
  features and a closed route mask; normalized freshness is observed.
- 512-bit ACK-free beacon per UAV per second, 2.0-s freshness cutoff.
- Beacon transmission/receiving consumes the *same battery* as propulsion.
- HELLO radio power is fixed to configured 0.6 W; never claim it is learned.
- No true target coordinate/ID is exposed to any decentralized Actor feature.
- The old NoOracle 11 checkpoint MUST NOT be loaded as observation version 12.
- Scientific comparison: HELLO OFF vs HELLO ON, same map and evaluation
  seeds, separate W&B groups and Hugging Face run IDs, fresh models.
- Full 20,000 MATD3 episodes, 100 held-out val episodes per saved Actor,
  1,000 final test episodes are *planned*, not performed in this notebook file.
- GPU/T4 latency and full Python UavNetSim protocol parity are NOT verified.
'''
    n.cells[1].source+='''
HELLO_BEACON_ENABLED=True
HELLO_INTERVAL_SECONDS=1.0
HELLO_FRESHNESS_SECONDS=2.0
HELLO_TOTAL_BITS=512
HELLO_OBSERVATION_VERSION=12
assert HELLO_BEACON_ENABLED and HELLO_FRESHNESS_SECONDS==2
'''
    bootstrap=n.cells[2].source
    # The inherited historical MAP800 preflight hardcoded observation v11.
    # Keep preflight strict but update its actual subprocess assertion to v12;
    # otherwise a fully valid HELLO run would die *before* training begins.
    old_guard="observation_version(c)==11; "
    assert bootstrap.count(old_guard)==1
    bootstrap=bootstrap.replace(old_guard,"observation_version(c)==12; ",1)
    old_print='BAYES_POSTERIOR_ONLY_VERSION11_CONFIG_OK'
    assert bootstrap.count(old_print)==1
    bootstrap=bootstrap.replace(old_print,'BAYES_POSTERIOR_ONLY_HELLO_VERSION12_CONFIG_OK',1)
    marker='spec_file = Path("/kaggle/working/job.json")'
    assert bootstrap.count(marker)==1
    preflight='''# The imported extracted *source* must contain the over-air HELLO model.
_hello_cfg=_load_25d_config(SPEC["config"])
from uav_network.environment.observations import observation_version as _hello_version
assert _hello_cfg.network.hello_enabled
assert _hello_cfg.network.hello_period_s == 1.0
assert _hello_cfg.network.hello_freshness_s == 2.0
assert _hello_cfg.network.hello_bits == 512
assert _hello_version(_hello_cfg) == 12
assert _hello_cfg.runtime.warm_start_checkpoint == "none"
SPEC["runs"]["matd3"]["name"] += "-HELLO1HZ-FRESH2S-NEWMDP"
SPEC["science_ablation"]["hello_enabled"] = True
SPEC["science_ablation"]["hello_period_s"] = 1.0
SPEC["science_ablation"]["hello_freshness_s"] = 2.0
SPEC["science_ablation"]["hello_bits"] = 512
SPEC["science_ablation"]["hello_observation_version"] = 12
SPEC["science_ablation"]["full_gpu_test_completed"] = False
print("SEALED_HELLO1HZ_ZENG2019_NEW_MDP_PREFLIGHT_OK",flush=True)
'''
    n.cells[2].source=bootstrap.replace(marker,preflight+marker,1)
    ast.parse(n.cells[2].source)
    n.metadata.update({
        'science_config':CONFIG,
        'experiment_kind':'25d_aoi_zeng2019_overair_hello_ablation',
        'beacon_implemented':True,
        'beacon_period_s':1.0,
        'beacon_freshness_s':2.0,
        'beacon_bits':512,
        'beacon_rf_simulator':'Warp PHY/MAC shared event loop',
        'beacon_upstream_uavnetsim_protocol_parity_verified':False,
        'observation_version':12,
        'new_mdp_requires_training_from_scratch':True,
        'runtime_cuda_validation_complete':False,
        'fullflow_20k_gpu_run_complete':False,
    })
    nbformat.validate(n)
    if len(nbformat.writes(n).encode())>=850_000:
        raise RuntimeError('Sealed HELLO notebook larger than the safe size cap')
    return n


def main():
    notebook=build_notebook()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(notebook,OUT)
    print('SEALED_HELLO_ZENG_NOTEBOOK',OUT,
          'REFERENCE_COMMIT',notebook.metadata['reference_commit'],
          'BUNDLE_SHA256',notebook.metadata['source_sha256'],
          'NOTEBOOK_SHA256',hashlib.sha256(OUT.read_bytes()).hexdigest())

if __name__=='__main__':
    main()
