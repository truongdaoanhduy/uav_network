"""XYZ studies retain distinct provenance and coverage remains local."""
import numpy as np
from uav_network.experiments.study import plan_study


def test_xyz_manifest_labels_new_lineage(tmp_path):
    m=plan_study("configs/comparison-20k-hybrid75-1kb-xyz-sensing.yaml",tmp_path/"study",
                 seeds=[44],variants=["baseline"],algorithms=["masac"],modes=["base"])
    assert m["study"]=="controlled-20t15o-xyz-bayes"
    assert "xyz-bayes" in m["cases"][0]["id"]
    from uav_network.experiments.study import _load
    assert _load(tmp_path/"study"/"manifest.json")[1]["study"]==m["study"]


def test_xyz_coarse_coverage_matches_native_at_actual_height():
    from test_xyz_warp import cfg
    from uav_network.environment.env import SwarmSearchEnv
    from uav_network.environment.gpu.vector import WarpVectorEnv
    c=cfg();c.sensor.coverage_grid_cells=8;c.sensor.altitude_m=25
    native=SwarmSearchEnv(c);v=WarpVectorEnv(c,"cpu")
    try:
        native.reset(seed=44);v.reset(seed=44)
        native.positions[:,2]=150
        native.coverage.fill(0)
        from uav_network.environment.coverage import scan_coverage
        scan_coverage(native)
        import warp as wp
        wp.to_torch(v.s.altitudes).fill_(150);v.s.coverage.zero_()
        from uav_network.environment.gpu.geometry import scan_coverage as device_scan
        v._launch(device_scan,(1,c.world.num_uavs,64),v.s,v.p)
        np.testing.assert_array_equal(native.coverage,v.s.coverage.numpy()[0])
    finally:native.close();v.close()
