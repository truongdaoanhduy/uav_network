"""RED/GREEN: no silent YAML mismatch in the Warp capacity diagnostic."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import math
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / 'research/zhang_xue/configs/scenario_v67_datamule_1km.yaml'


def _scenario(**kwargs):
    from zhang_xue.configuration.loader import load_scenario
    return replace(load_scenario(SCENARIO), **kwargs)


def test_radio_phy_from_actual_scenario():
    from uav_network.experiments.radio_config import radio_phy_kwargs
    s = _scenario(tx_power_w=0.23, mac_cca_threshold_dbm=-79.0,
                  mac_noise_power_w=1e-12, mac_snr_threshold_db=9.0)
    params = radio_phy_kwargs(s)
    assert params['tx_power_w'] == pytest.approx(0.23)
    assert params['cca_threshold_w'] == pytest.approx(1e-3 * 10 ** (-79/10))
    assert params['noise_w'] == pytest.approx(1e-12)
    assert params['snr_threshold'] == pytest.approx(10 ** .9)


@pytest.mark.parametrize('flag', [True, False])
def test_radio_constructor_uses_explicit_peer_flag(flag):
    from uav_network.experiments.radio_config import network_kwargs
    x = network_kwargs(_scenario(compact3_decentralized_peer_select=flag))
    assert x['decentralized_peer_select'] is flag


def test_radio_constructor_preserves_data_mule_and_peer_ablation():
    from uav_network.experiments.radio_config import network_kwargs
    s = _scenario(enable_peer_dissemination=False, enable_data_mule_forwarding=False)
    kwargs = network_kwargs(s)
    assert kwargs['enable_peer'] is False
    assert kwargs['enable_data_mule'] is False


@pytest.mark.parametrize('field,value', [('mac_difs_us',70), ('mac_ack_timeout_us',200)])
def test_unsupported_compiled_warp_timing_fails_closed(field,value):
    from uav_network.experiments.radio_config import radio_config_manifest
    with pytest.raises(ValueError, match=field):
        radio_config_manifest(_scenario(**{field: value}))


def test_warp_radio_default_queue_divergence_is_disclosed():
    from uav_network.experiments.radio_config import radio_config_manifest
    data = radio_config_manifest(_scenario())
    assert data['phy_step_kwargs']['tx_power_w'] == pytest.approx(.1)
    assert data['network_kwargs']['decentralized_peer_select'] is True
    assert data['comparability']['cpu_queue_capacity'] == 64
    assert data['comparability']['warp_queue_capacity'] == 32
    assert data['comparability']['event_parity_verified'] is False
