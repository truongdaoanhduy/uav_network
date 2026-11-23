"""Tar extraction verifies the exact HELLO/Zeng code used on Kaggle."""
import ast
import base64
import hashlib
import io
import re
import tarfile
import yaml
from scripts.build_map1000_25d_zeng_hello_fullflow_ui_20261123 import build_notebook,CONFIG


def test_hello_sealed_kaggle_fullflow_contains_only_current_science_source():
    notebook=build_notebook()
    assert len(notebook.cells)==3
    assert notebook.metadata['science_config']==CONFIG
    assert notebook.metadata['beacon_implemented'] is True
    assert notebook.metadata['beacon_upstream_uavnetsim_protocol_parity_verified'] is False
    assert notebook.metadata['validation_episodes_per_checkpoint']==100
    assert notebook.metadata['test_episodes']==1000
    assert notebook.metadata['fullflow_20k_gpu_run_complete'] is False
    assert 'SEALED_HELLO1HZ_ZENG2019_NEW_MDP_PREFLIGHT_OK' in notebook.cells[2].source
    assert 'observation_version(c)==11' not in notebook.cells[2].source
    assert 'observation_version(c)==12' in notebook.cells[2].source
    assert 'BAYES_POSTERIOR_ONLY_HELLO_VERSION12_CONFIG_OK' in notebook.cells[2].source
    pattern=r'payload = base64\.b85decode\(("[^"\n]+")\)'
    match=re.search(pattern,notebook.cells[2].source)
    assert match
    payload=base64.b85decode(ast.literal_eval(match.group(1)))
    assert hashlib.sha256(payload).hexdigest()==notebook.metadata['source_sha256']
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        files=set(archive.getnames())
        assert CONFIG in files
        for name in ('uav_network/communication/gpu/hello.py',
                     'uav_network/communication/gpu/mac.py',
                     'uav_network/environment/gpu/observations.py',
                     'uav_network/environment/gpu/state.py'):
            assert name in files
        cfg=yaml.safe_load(archive.extractfile(CONFIG).read())
        assert cfg['network']['hello_enabled'] is True
        assert cfg['network']['hello_period_s']==1.0
        assert cfg['network']['hello_freshness_s']==2.0
        assert cfg['network']['hello_bits']==512
        assert cfg['energy']['propulsion_model']=='zeng2019'
        assert cfg['world']['motion_safety']=='reject'
        assert cfg['runtime']['episodes']==20000
        assert cfg['runtime']['warm_start_checkpoint']=='none'
        beacon=archive.extractfile('uav_network/communication/gpu/hello.py').read()
        assert b'def receive_hello' in beacon and b'def emit_hello' in beacon
        assert b'hello_rx_time' in beacon
