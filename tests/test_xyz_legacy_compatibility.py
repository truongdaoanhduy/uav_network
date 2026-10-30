"""Legacy XY winner bytes remain resumable after optional XYZ settings are added."""
import hashlib
import io
import json
from copy import deepcopy
import torch
from test_best_checkpoints import config,candidate
from uav_network.experiments.best_checkpoints import BestCheckpoint
from uav_network.configuration_xyz import legacy_xyz_defaults


def test_pre_xyz_xy_selected_winner_restores_without_relabeling(tmp_path):
    c=config()
    result,source=candidate(c,tmp_path/"candidate")
    selection=BestCheckpoint(c,tmp_path/"first")
    assert selection.consider(result,source)
    state=selection.state_dict()
    saved=torch.load(source["checkpoint"],weights_only=True)
    raw=deepcopy(saved["config"])
    for group,defaults in legacy_xyz_defaults().items():
        for key in defaults:raw[group].pop(key)
    saved["config"]=raw
    data=io.BytesIO();torch.save(saved,data);data=data.getvalue()
    identity={k:raw[k] for k in ("world","sensor","network","energy","reward",c.runtime.algorithm,"pql")}
    identity["runtime"]={k:raw["runtime"][k] for k in ("algorithm","seed","amp_mode","num_envs","environment_backend")}
    identity["validation"]=[c.tracking.periodic_evaluation_seed,c.tracking.periodic_evaluation_episodes]
    digest=lambda d:hashlib.sha256(json.dumps(d,sort_keys=True,allow_nan=False).encode()).hexdigest()
    state["metadata"].update(identity=digest(identity),config_sha256=digest(raw),checkpoint_sha256=hashlib.sha256(data).hexdigest())
    state["checkpoint_bytes"]=torch.frombuffer(bytearray(data),dtype=torch.uint8).clone()
    restored=BestCheckpoint(c,tmp_path/"restored",state)
    assert restored.result("last.pt")["checkpoint_selection"]["kind"]=="validation_best"
    assert (tmp_path/"restored"/"best.pt").read_bytes()==data


def test_legacy_fixed_height_above_xyz_ceiling_remains_valid():
    c=config();c.sensor.altitude_m=300
    c.validate()
