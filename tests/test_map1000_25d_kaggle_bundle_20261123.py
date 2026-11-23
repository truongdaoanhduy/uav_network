"""New 2.5D source sealing must include exact new scientific logic."""
import ast
import base64
import hashlib
import io
import re
import tarfile
import yaml

from scripts.build_map1000_25d_aoi_fullflow_ui_20261123 import build_notebook, CONFIG


def test_new_25d_kaggle_fullflow_embeds_actual_scenario_and_reward_source():
    notebook=build_notebook()
    assert notebook.metadata["science_config"]==CONFIG
    assert notebook.metadata["new_mdp_requires_training_from_scratch"] is True
    assert notebook.metadata["full_20000_training"] is True
    assert notebook.metadata["test_episodes"]==1000
    assert notebook.metadata["validation_episodes_per_checkpoint"]==100
    assert notebook.metadata["beacon_implemented"] is False
    assert notebook.metadata["zeng_propulsion_implemented"] is False
    assert "TRAIN_EPISODES = 20000" in notebook.cells[1].source
    assert "NEW_MAX_VERTICAL_SPEED_MPS=4" in notebook.cells[1].source
    assert 'SEALED_25D_AOI_SCIENCE_PREFLIGHT_OK' in notebook.cells[2].source

    matches=re.findall(r'payload = base64\.b85decode\(("[^"\n]+")\)',notebook.cells[2].source)
    assert len(matches)==1
    compressed=base64.b85decode(ast.literal_eval(matches[0]))
    assert hashlib.sha256(compressed).hexdigest()==notebook.metadata["source_sha256"]
    with tarfile.open(fileobj=io.BytesIO(compressed),mode="r:gz") as archive:
        files=set(archive.getnames())
        assert CONFIG in files
        assert "uav_network/communication/gpu/reports.py" in files
        assert "uav_network/environment/gpu/xyz_motion.py" in files
        cfg=yaml.safe_load(archive.extractfile(CONFIG).read())
        assert cfg["reward"]["delivery_aoi_weight"]==0.5
        assert cfg["reward"]["communication_efficiency"]==0.05
        assert cfg["world"]["max_vertical_speed_mps"]==4
        source=archive.extractfile("uav_network/communication/gpu/reports.py").read().decode()
        assert "true_delivery_freshness" in source
        assert "reward_delivery_aoi_weight" in source
        assert "peer_without_beacon" not in source  # MAC-only routing feature, not reward
