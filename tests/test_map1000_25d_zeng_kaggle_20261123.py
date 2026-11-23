"""Zeng fullflow notebook must seal the new energy MDP, not a stale YAML."""
import ast
import base64
import hashlib
import io
import re
import tarfile
import yaml
from scripts.build_map1000_25d_zeng2019_fullflow_ui_20261123 import build_notebook,CONFIG


def test_zeng_kaggle_package_provenance_and_runtime_model():
    nb=build_notebook()
    assert len(nb.cells)==3
    assert nb.metadata["science_config"]==CONFIG
    assert nb.metadata["zeng_propulsion_implemented"] is True
    assert nb.metadata["fullflow_20k_gpu_run_complete"] is False
    assert nb.metadata["new_mdp_requires_training_from_scratch"] is True
    assert nb.metadata["validation_episodes_per_checkpoint"] == 100
    assert nb.metadata["test_episodes"] == 1000
    assert "SEALED_ZENG2019_NEW_MDP_PREFLIGHT_OK" in nb.cells[2].source
    pattern=r'payload = base64\\.b85decode\\(("[^"\\n]+")\\)'
    match=re.search(pattern.replace('\\\\','\\'),nb.cells[2].source)
    assert match
    raw=base64.b85decode(ast.literal_eval(match.group(1)))
    assert hashlib.sha256(raw).hexdigest()==nb.metadata["source_sha256"]
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as archive:
        names=set(archive.getnames())
        assert CONFIG in names
        c=yaml.safe_load(archive.extractfile(CONFIG).read())
        assert c["energy"]["propulsion_model"]=="zeng2019"
        assert c["energy"]["rotor_weight_n"]==20
        assert c["runtime"]["warm_start_checkpoint"]=="none"
        motion=archive.extractfile("uav_network/environment/gpu/xyz_motion.py").read()
        assert b"rotor_p0" in motion
        native=archive.extractfile("uav_network/environment/energy.py").read()
        assert b"zeng2019" in native
