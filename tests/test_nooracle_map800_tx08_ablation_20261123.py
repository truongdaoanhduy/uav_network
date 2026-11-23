"""Optional 0.8W scientific ablation cannot silently replace 0.4W baseline."""
from copy import deepcopy

from uav_network.configuration import load_config
from uav_network.communication.link_budget import required_power_w, report_reference_tx_energy_j

BASE='configs/matd3-nooracle-adaptive-altitude-lr4-map800-h256x256-orthogonal-compile-20261123.yaml'
TX08='configs/matd3-nooracle-adaptive-altitude-lr4-map800-h256x256-orthogonal-compile-tx08-ablation-20261123.yaml'


def test_tx08_ablation_changes_only_max_tx_and_tracking_identity():
    original=load_config(BASE)
    alternative=load_config(TX08)
    assert original.network.tx_power_max_w==0.4
    assert alternative.network.tx_power_max_w==0.8
    old=original.to_dict();new=alternative.to_dict()
    old['network'].pop('tx_power_max_w');new['network'].pop('tx_power_max_w')
    old.pop('tracking');new.pop('tracking')
    old['runtime'].pop('output_dir');new['runtime'].pop('output_dir')
    assert old==new
    assert original.tracking.run_name!=alternative.tracking.run_name


def test_nominal_urban_nlos_power_needs_medium_range_not_full_500m():
    c=load_config(BASE).network
    def need(d):
        return required_power_w(d,c.carrier_frequency_hz,c.bandwidth_hz,
                                c.noise_density_dbm_hz,c.noise_figure_db,
                                c.sinr_threshold_db,c.nlos_model,False,c.obstacle_extra_loss_db)
    assert need(100)<0.4 < need(125) <0.8 < need(200)
    assert need(500)>30
    original_energy=report_reference_tx_energy_j(c)
    next_c=deepcopy(c);next_c.tx_power_max_w=0.8
    assert report_reference_tx_energy_j(next_c)>original_energy
