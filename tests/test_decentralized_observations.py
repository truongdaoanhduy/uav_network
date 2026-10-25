"""A distant GCS delivery cannot acknowledge a local camera sighting."""
import numpy as np
import pytest

from uav_network.configuration import Config
from uav_network.communication.reports import ReportStore
from uav_network.environment.env import SwarmSearchEnv
from uav_network.environment.observations import local_observations


def local_config(sighting):
    c = Config()
    c.world.num_uavs, c.world.num_targets, c.world.num_obstacles = 2, 2, 0
    c.sensor.altitude_m = 20
    c.network.enabled = False
    c.network.peer_range_m = c.network.gcs_range_m = 100
    c.network.rediscovery_recovery = True
    if sighting == "full":
        c.network.buffer_bytes = 1024
    return c


@pytest.mark.parametrize("sighting", ["fresh", "recovery", "full"])
def test_native_sighting_ignores_unannounced_gcs_delivery(sighting):
    observations = []
    before = []
    for delivered in (False, True):
        env = SwarmSearchEnv(local_config(sighting))
        try:
            env.reset(seed=44, options={"positions": [[100,100],[800,800]],
                                       "targets": [[300,100],[900,200]], "obstacles": []})
            env.steps = 301
            if sighting == "recovery":
                env.reports.known[0,0] = env.reports.recovery_pending[0,0] = True
            if sighting == "full":
                env.reports.discover(0, 1, env.time_s)
            # Only the distant GCS receipt differs; no message reached UAV0.
            env.reports.delivered[0] = delivered
            before.append(local_observations(env)[0])
            env.targets[0] = env.positions[0]
            env._sense()
            observations.append(local_observations(env)[0])
            assert (0 in env.reports.buffers[0]) == (sighting != "full")
            assert env.reports.recovery_pending[0,0] == (sighting == "full")
            if sighting != "full":
                assert env.reports.buffers[0][0].created_s == 301
        finally:
            env.close()
    np.testing.assert_array_equal(before[0], before[1])
    np.testing.assert_array_equal(observations[0], observations[1])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("sighting", ["fresh", "recovery", "full"])
def test_warp_sighting_ignores_unannounced_gcs_delivery(sighting, device):
    wp = pytest.importorskip("warp")
    import torch
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA hardware required")
    from uav_network.environment.gpu.state import allocate_state
    from uav_network.environment.gpu.geometry import sense
    from uav_network.environment.gpu.observations import observe
    from uav_network.communication.gpu.reports import sense_reports

    c = local_config(sighting)
    s, p = allocate_state(c, num_envs=2, device=device)
    s.positions.assign(np.array([[[100,100],[800,800]]] * 2, dtype=float))
    s.targets.assign(np.array([[[100,100],[900,200]]] * 2, dtype=float))
    s.steps.fill_(301)
    s.first_discovery.fill_(-1)
    s.delivered.assign(np.array([[0,0],[1,0]], dtype=np.int32))
    known = np.zeros((2,2,2), dtype=np.int32)
    if sighting == "recovery":
        known[:,0,0] = 1
        s.recovery.assign(known)
    if sighting == "full":
        known[:,0,1] = 1
        s.report_present.assign(known)
    s.known.assign(known)
    wp.launch(sense, dim=(2,p.n,p.targets_count), inputs=[s,p], device=device)
    wp.launch(observe, dim=(2,p.n), inputs=[s,p], device=device)
    before = s.observation.numpy()[:,0].copy()
    wp.launch(sense_reports, dim=2, inputs=[s,p], device=device)
    wp.launch(observe, dim=(2,p.n), inputs=[s,p], device=device)
    np.testing.assert_array_equal(before[0], before[1])
    np.testing.assert_array_equal(s.report_present.numpy()[:,0,0], [int(sighting != "full")] * 2)
    np.testing.assert_array_equal(s.recovery.numpy()[:,0,0], [int(sighting == "full")] * 2)
    if sighting != "full":
        np.testing.assert_array_equal(s.report_created.numpy()[:,0,0], [301,301])
    np.testing.assert_array_equal(s.observation.numpy()[0,0], s.observation.numpy()[1,0])


def test_gcs_deduplicates_reports_created_by_independent_local_sightings():
    store = ReportStore(local_config("fresh"))
    for source in (0,1):
        store.discover(source, 0, 0)
        assert 0 in store.buffers[source]
        assert store.transfer(source, 2, store.buffers[source][0], source + 1)
    assert store.delivered.tolist() == [True,False]
    assert store.direct_gcs_reports == 1
    assert store.delivery_times_s == [1]
    assert store.bytes_used(0) == store.bytes_used(1) == 0
