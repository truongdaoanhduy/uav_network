"""Network-only SoA memory report must be reproducible and label limitations."""
from scripts.maintenance.network_memory_report import memory_inventory


def test_budget_matrix_reports_per_configuration_and_distinguishes_vram():
    report = memory_inventory(env_counts=(2048,4096))
    assert report['measured_cuda_peak'] is False
    configs = {(v['num_uavs'],v['grid_w'],v['num_envs']):v for v in report['profiles']}
    assert configs[(6,20,2048)]['network_soa_gib'] < 1
    assert configs[(12,40,2048)]['network_soa_gib'] > 15
    assert configs[(30,40,2048)]['network_soa_gib'] > 190
    assert configs[(6,20,4096)]['network_soa_gib'] > configs[(6,20,2048)]['network_soa_gib']
    assert 'GPU' in report['limitations'][0]
