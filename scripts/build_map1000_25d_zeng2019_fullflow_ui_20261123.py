"""Source-sealed Kaggle fullflow scientific ablation for calibrated Zeng propulsion."""
from contextlib import contextmanager
from pathlib import Path
import ast
import hashlib
import nbformat

import scripts.build_map1000_25d_aoi_fullflow_ui_20261123 as base

ROOT=Path(__file__).resolve().parents[1]
CONFIG="configs/matd3-nooracle-map1000-25d-aoi-zeng2019-h35-130-vz4-20261123.yaml"
NAME="MATD3-MAP1000-25D-AOI-ZENG2019-H35-130-VZ4-T4X2-VAL100-TEST1000-20261123.ipynb"
OUT=ROOT/"notebooks"/NAME


@contextmanager
def energy_config_context():
    original=base.CONFIG
    try:
        base.CONFIG=CONFIG
        yield
    finally:
        base.CONFIG=original


def build_notebook():
    from uav_network.configuration import load_config
    cfg=load_config(CONFIG)
    cfg.validate()
    assert cfg.energy.propulsion_model=="zeng2019"
    assert cfg.energy.rotor_weight_n==20
    assert cfg.energy.rotor_angular_velocity_rad_s==220
    assert cfg.world.max_vertical_speed_mps==4
    assert cfg.world.motion_safety=="reject"
    assert cfg.runtime.episodes==20000 and cfg.matd3.batch_size==8192
    assert cfg.runtime.warm_start_checkpoint=="none"
    with energy_config_context():
        n=base.build_notebook()
    assert n.metadata["science_config"]==CONFIG
    before="- This notebook does NOT implement HELLO beacon nor Zeng propulsion."
    after="- This notebook DOES implement a calibrated but unverified Zeng2019 rotary-wing propulsion model. HELLO beacon is NOT implemented."
    assert n.cells[0].source.count(before)==1
    n.cells[0].source=n.cells[0].source.replace(before,after)
    n.cells[0].source+="""

### Optional rotary-wing energy-model ablation (new training required)

- Different energy dynamics: Zeng, Xu & Zhang (IEEE TWC 2019) forward-flight model.
- Calibration: approximate 2 kg/20 N, rotor radius 0.5 m, omega 220 rad/s,
  induced hover speed 3.2145 m/s, and compatible air density/rotor coefficients.
- Approx hover 167.325 W and cruise at 10 m/s 128.873 W.
- Positive climb adds weight times vertical speed; no descent recovery.
- These are simulated assumptions, not measured aircraft parameters.
- This notebook compares energy-model *MDPs*; do not reuse old checkpoints.
- No actual T4x2 training, val/test results, or speedup measurements yet.
"""
    n.cells[1].source+="""
NEW_ENERGY_MODEL="zeng2019"
NEW_ROTOR_WEIGHT_N=20
NEW_ROTOR_OMEGA_RAD_S=220
assert NEW_ENERGY_MODEL=="zeng2019"
"""
    bootstrap=n.cells[2].source
    marker='spec_file = Path("/kaggle/working/job.json")'
    assert bootstrap.count(marker)==1
    verify="""# Energy-model identity on the *extracted* runtime source, not just metadata.
_zeng=_load_25d_config(SPEC["config"])
assert _zeng.energy.propulsion_model == "zeng2019"
assert _zeng.energy.rotor_weight_n == 20
assert _zeng.energy.rotor_angular_velocity_rad_s == 220
assert _zeng.runtime.warm_start_checkpoint == "none"
SPEC["runs"]["matd3"]["name"] += "-ZENG2019"
SPEC["science_ablation"]["propulsion_model"] = "zeng2019"
SPEC["science_ablation"]["rotor_weight_n"] = 20
SPEC["science_ablation"]["rotor_omega_rad_s"] = 220
SPEC["science_ablation"]["rotor_calibration_airframe_verified"] = False
print("SEALED_ZENG2019_NEW_MDP_PREFLIGHT_OK",flush=True)
"""
    n.cells[2].source=bootstrap.replace(marker,verify+marker,1)
    ast.parse(n.cells[2].source)
    n.metadata.update({
       "science_config":CONFIG,
       "experiment_kind":"25d_aoi_zeng2019_energy_ablation",
       "zeng_propulsion_implemented":True,
       "zeng_calibration_airframe_verified":False,
       "energy_model":"zeng2019",
       "rotor_weight_n":20,
       "new_mdp_requires_training_from_scratch":True,
       "runtime_cuda_validation_complete":False,
       "fullflow_20k_gpu_run_complete":False,
    })
    nbformat.validate(n)
    if len(nbformat.writes(n).encode()) >= 850000:
        raise RuntimeError("Oversize notebook, refusing to remove data or source")
    return n


def main():
    notebook=build_notebook()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    nbformat.write(notebook,OUT)
    print("SEALED_ZENG2019_NOTEBOOK",OUT,
          "SOURCE_COMMIT",notebook.metadata["reference_commit"],
          "BUNDLE_SHA256",notebook.metadata["source_sha256"],
          "NOTEBOOK_SHA256",hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__=="__main__":
    main()
