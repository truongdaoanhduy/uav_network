"""Explicit scenario-to-Warp radio contract for diagnostics and dynamic v75 params."""
import math

from zhang_xue.communication.common import CsmaProtocolConfig
from zhang_xue.gpu.layout_v67 import Q_V67, Q_V67_MAX
from zhang_xue.gpu.state_v67 import (
    ACK_TIMEOUT_US, CW_MIN, DIFS_US, MAX_RETX, SIFS_US, SLOT_US,
)

# Default reference MAC timings (also supported dynamically via SimDimsV67 in v75).
_COMPILED_TIMING = {
    'mac_difs_us': DIFS_US, 'mac_sifs_us': SIFS_US,
    'mac_slot_us': SLOT_US, 'mac_cw_min': CW_MIN,
    'mac_max_retries': MAX_RETX, 'mac_ack_timeout_us': ACK_TIMEOUT_US,
}


def scenario_protocol_config(scenario) -> CsmaProtocolConfig:
    """Build and validate CsmaProtocolConfig from scenario MAC fields."""
    return CsmaProtocolConfig(
        difs_us=int(scenario.mac_difs_us),
        sifs_us=int(scenario.mac_sifs_us),
        slot_us=int(scenario.mac_slot_us),
        cw_min=int(scenario.mac_cw_min),
        max_retries=int(scenario.mac_max_retries),
        ack_timeout_us=int(scenario.mac_ack_timeout_us),
    )


def network_kwargs(scenario, *, warp_q_cap: int = Q_V67, include_protocol: bool = False):
    """All runtime constructor options supported by WarpCompressedRadioNetworkV67."""
    if isinstance(warp_q_cap, bool) or not isinstance(warp_q_cap, int) or not (1 <= warp_q_cap <= Q_V67_MAX):
        raise ValueError(f'warp_q_cap must be an integer in [1, {Q_V67_MAX}]')
    effective_eta = float(scenario.ds_aoi_eta) if scenario.ds_fusion_mode == 'sum' else 0.0
    kwargs = {
        'enable_data_mule': bool(scenario.enable_data_mule_forwarding),
        'enable_peer': bool(scenario.enable_peer_dissemination),
        'decentralized_peer_select': bool(scenario.compact3_decentralized_peer_select),
        'aoi_ewma': bool(scenario.aoi_ewma_end_to_end),
        'aoi_eta': effective_eta,
        'rx_lock_on_cca': False,
        'q_cap': int(warp_q_cap),
    }
    if include_protocol:
        kwargs['protocol'] = scenario_protocol_config(scenario)
    return kwargs


def radio_phy_kwargs(scenario):
    """Convert scenario dBm/dB values to the linear units required by Warp.step."""
    return {
        'tx_power_w': float(scenario.tx_power_w),
        'cca_threshold_w': 1e-3 * math.pow(10.0, float(scenario.mac_cca_threshold_dbm) / 10.0),
        'noise_w': float(scenario.mac_noise_power_w),
        'snr_threshold': math.pow(10.0, float(scenario.mac_snr_threshold_db) / 10.0),
    }


def radio_config_manifest(
    scenario, *, warp_q_cap: int = Q_V67,
    allow_dynamic_mac_timing: bool = False, require_queue_parity: bool = False,
):
    """Validate scenario->Warp contract and fail closed on unsupported modes.

    By default keeps strict v74 compile-time MAC timing guard and Q_V67=32
    compatibility; pass `allow_dynamic_mac_timing=True` and/or
    `warp_q_cap=scenario.mac_queue_capacity` (<= 127) to opt into v75 dynamic
    Warp SimDimsV67 parameters.
    """
    if scenario.comm_action_mode != 'compact_3':
        raise ValueError(f'comm_action_mode={scenario.comm_action_mode!r} unsupported by Warp compact_3')
    if scenario.enable_tx_power_control:
        raise ValueError('enable_tx_power_control=True unsupported by scalar-TX Warp step_v67')
    if not scenario.aoi_ewma_end_to_end:
        raise ValueError('aoi_ewma_end_to_end=False unsupported by compressed Warp ACK watermarks')
    if scenario.network_substeps != 1:
        raise ValueError(f'network_substeps={scenario.network_substeps} unsupported by single-step Warp probe')
    if scenario.coalesce_duplicate_queue:
        raise ValueError('coalesce_duplicate_queue=True unsupported by Warp FIFO queue')
    if require_queue_parity and int(scenario.mac_queue_capacity) != int(warp_q_cap):
        raise ValueError(
            f"queue parity required: CPU mac_queue_capacity={scenario.mac_queue_capacity} "
            f"!= Warp q_cap={warp_q_cap}; use --warp-q-cap {scenario.mac_queue_capacity} "
            "when <= 127"
        )
    proto = scenario_protocol_config(scenario)
    if not allow_dynamic_mac_timing:
        for field, supported in _COMPILED_TIMING.items():
            actual = getattr(scenario, field)
            if actual != supported:
                raise ValueError(f'{field}={actual} unsupported by Warp v67 compile-time value {supported}')
    phy = radio_phy_kwargs(scenario)
    if any(not math.isfinite(v) or v <= 0 for v in phy.values()):
        raise ValueError('Invalid PHY values after dBm/dB conversion')
    net_kw = network_kwargs(scenario, warp_q_cap=warp_q_cap, include_protocol=False)
    mac_effective = {
        'mac_difs_us': proto.difs_us, 'mac_sifs_us': proto.sifs_us,
        'mac_slot_us': proto.slot_us, 'mac_cw_min': proto.cw_min,
        'mac_max_retries': proto.max_retries, 'mac_ack_timeout_us': proto.ack_timeout_us,
    }
    return {
        'network_kwargs': net_kw,
        'phy_step_kwargs': phy,
        'warp_mac_compile_time': dict(_COMPILED_TIMING),
        'warp_mac_effective': mac_effective,
        'rf_obstacle_model': scenario.rf_obstacle_model,
        'rf_per_building_loss_db': float(scenario.rf_per_building_loss_db),
        'rf_phy_applied_by_capacity_probe': False,
        'dynamic_mac_timing_enabled': bool(allow_dynamic_mac_timing),
        'comparability': {
            'cpu_queue_capacity': int(scenario.mac_queue_capacity),
            'warp_queue_capacity': int(warp_q_cap),
            'queue_capacities_match': int(scenario.mac_queue_capacity) == int(warp_q_cap),
            'queue_parity_required': bool(require_queue_parity),
            'event_parity_verified': False,
            'full_gpu_rollout_verified': False,
            'note': 'CPU and Warp are not protocol/event equivalent just because endpoints match',
        },
    }
