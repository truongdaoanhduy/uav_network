"""Build the new, source-sealed 2.5D + AoI scientific-ablation Kaggle notebook.

No previous notebook or historical science file is overwritten.
"""
from contextlib import contextmanager
from pathlib import Path
import ast
import hashlib
import nbformat

import scripts.build_nooracle_map1000_tx06_fastgrid_ui_20261123 as upstream_map1000
import scripts.build_map1000_tx06_autotune_fullflow_ui_20261123 as upstream_autotune
from scripts.build_map1000_tx06_autotune_fullflow_fe799_fix_20261123 import (
    build_notebook as validated_fullflow_builder, TARGET_KERNEL_REF,
)

ROOT=Path(__file__).resolve().parents[1]
CONFIG="configs/matd3-nooracle-map1000-25d-h35-130-vz4-20261123.yaml"
NAME="MATD3-MAP1000-25D-AOI-H35-130-VZ4-T4X2-VAL100-TEST1000-20261123.ipynb"
OUT=ROOT/"notebooks"/NAME
RUN_SUFFIX="-25D-AOI-H35-130-VZ4-NEWMDP"


@contextmanager
def patched_builder_config():
    saved_map1000=upstream_map1000.CONFIG
    saved_autotune=upstream_autotune.CONFIG
    try:
        upstream_map1000.CONFIG=CONFIG
        upstream_autotune.CONFIG=CONFIG
        yield
    finally:
        upstream_map1000.CONFIG=saved_map1000
        upstream_autotune.CONFIG=saved_autotune


def build_notebook():
    from uav_network.configuration import load_config
    config=load_config(CONFIG)
    config.validate()
    assert config.world.map_size_m==1000 and config.world.num_uavs==6
    assert config.world.altitude_min_m==35 and config.world.altitude_max_m==130
    assert config.world.max_vertical_speed_mps==4
    assert config.world.motion_safety=="reject"
    assert config.sensor.belief_prior==35/1600
    assert config.sensor.confirmation_mode=="posterior_only"
    assert config.reward.communication_efficiency==.05
    assert config.reward.delivery_aoi_weight==.5
    assert config.network.tx_power_max_w==.6 and config.network.enabled
    assert config.runtime.episodes==20000 and config.runtime.num_envs==4096
    assert config.matd3.batch_size==8192 and config.matd3.updates_per_step==4
    assert config.runtime.warm_start_checkpoint=="none"
    assert config.tracking.deferred_gpu_eval_enabled
    with patched_builder_config():
        n=validated_fullflow_builder()
    assert n.metadata["science_config"]==CONFIG
    assert len(n.cells)==3
    n.cells[0].source+="""

## New 2.5D/AoI scientific ablation: not the corrected baseline, not 59042708

- Three-dimensional flight remains enabled with altitude in [35, 130] m.
  The action-to-motion projection now limits |vertical velocity| to 4 m/s.
- Original 40x40 Bayesian target map, noisy Pd/Pf, LoS/NLoS, MAC CSMA/CA,
  queue, ACK, retransmissions, false reports and TTL are kept.
- Posterior-only target-cell confirmation remains threshold 0.99.
  No Actor access to true targets, remote node internals or idealized links.
- New RF reward coefficient 0.05 and delivered report freshness weight 0.5;
  both are unproven hypotheses that require independent learned-policy tests.
- CBF is NOT enabled in this notebook. Separate CBF config is a separate MDP.
- This notebook does NOT implement HELLO beacon nor Zeng propulsion.
- Fresh MATD3 train: 20,000 episodes, E4096/U4/B8192, T4x2; periodic
  actors evaluated on 100 held-out validation episodes each; final test 1000.
- Do not resume or overwrite previous run checkpoints. Speed gains and
  behavior improvement are NOT claimed without new real GPU evidence.
"""
    n.cells[1].source+="""
# Explicit scientific identity (confirmed again against sealed source below).
NEW_25D_ALTITUDE_RANGE=(35,130)
NEW_MAX_VERTICAL_SPEED_MPS=4
NEW_REWARD_COMM_EFFICIENCY=.05
NEW_DELIVERY_AOI_WEIGHT=.5
REQUIRES_FRESH_TRAIN=True
assert NEW_25D_ALTITUDE_RANGE==(35,130) and REQUIRES_FRESH_TRAIN
assert NEW_MAX_VERTICAL_SPEED_MPS==4
"""
    src=n.cells[2].source
    marker='spec_file = Path("/kaggle/working/job.json")'
    if src.count(marker)!=1:
        raise RuntimeError("Kaggle bootstrap source marker changed")
    preflight="""# Verify science on extracted actual ZIP source before spending GPU hours.
from uav_network.configuration import load_config as _load_25d_config
_new25d=_load_25d_config(SPEC["config"])
assert _new25d.world.altitude_min_m == 35
assert _new25d.world.altitude_max_m == 130
assert _new25d.world.max_vertical_speed_mps == 4
assert _new25d.world.motion_safety == "reject"
assert _new25d.sensor.belief_prior == 35/1600
assert _new25d.sensor.confirmation_mode == "posterior_only"
assert _new25d.reward.communication_efficiency == 0.05
assert _new25d.reward.delivery_aoi_weight == 0.5
assert _new25d.runtime.episodes == 20000
assert _new25d.matd3.batch_size == 8192
assert _new25d.runtime.warm_start_checkpoint == "none"
SPEC["runs"]["matd3"]["name"] += """+repr(RUN_SUFFIX)+"""
SPEC["science_ablation"] = {
  "altitude_min_m":35,"altitude_max_m":130,
  "max_vertical_speed_mps":4,"motion_safety":"reject",
  "belief_prior":35/1600, "communication_efficiency":0.05,
  "delivery_aoi_weight":0.5, "training_from_scratch":True,
  "new_mdp_not_comparable_to_59042708":True,
}
print("SEALED_25D_AOI_SCIENCE_PREFLIGHT_OK",flush=True)
"""
    src=src.replace(marker,preflight+marker,1)
    ast.parse(src)
    n.cells[2].source=src
    n.metadata.update({
       "science_config":CONFIG,"experiment_kind":"25d_aoi_science_ablation",
       "reference_commit":n.metadata["reference_commit"],
       "target_kernel_ref":TARGET_KERNEL_REF,
       "new_mdp_requires_training_from_scratch":True,"warm_start":"none",
       "max_vertical_speed_mps":4, "altitude_range_m":[35,130],
       "motion_safety":"reject","reward_communication_efficiency":0.05,
       "reward_delivery_aoi_weight":0.5,"belief_prior":0.021875,
       "beacon_implemented":False,"zeng_propulsion_implemented":False,
       "runtime_cuda_validation_complete":False,"fullflow_20k_gpu_run_complete":False,
    })
    nbformat.validate(n)
    if len(nbformat.writes(n).encode())>=850000:
        raise RuntimeError("Oversize Kaggle notebook; refuse to silently remove code")
    return n


def main():
    n=build_notebook()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(n,OUT)
    print("SEALED_25D_AOI_NOTEBOOK",OUT,
          "BYTES",OUT.stat().st_size,
          "REFERENCE_COMMIT",n.metadata["reference_commit"],
          "BUNDLE_SHA256",n.metadata["source_sha256"],
          "NOTEBOOK_SHA256",hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__=="__main__":
    main()
