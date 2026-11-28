#!/usr/bin/env python3
"""V119 immutable facts about the active DS canonical config and code provenance."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'research/zhang_xue/src')]
from zhang_xue.configuration.loader import load_scenario
from zhang_xue.communication.common import (GCS_POS_M, CANONICAL_BUILDINGS_5,
    CANONICAL_BLACKOUT_CENTERS_2, MAX_ENTRIES_PER_PKT)

SCENARIO = ROOT / 'research/zhang_xue/configs/scenario_v103_ds_only_1km.yaml'
LEARNER = ROOT / 'configs/learners/v85_canonical_matd3_masac_baseline.yaml'


def snapshot() -> dict:
    cfg = load_scenario(SCENARIO)
    data = asdict(cfg)
    fields = [
        'num_uavs', 'grid_w', 'map_size_m', 'num_targets', 'uav_alt_m',
        'v_max_mps', 'a_max_mps2', 'fov_radius_m', 'uav_safe_distance_m',
        'safety_horizon_s', 'pd', 'pf', 'ds_mf', 'ds_me', 'confirm_th',
        'ds_uncertainty_ceiling', 'ds_fusion_mode', 'static_evidence_mode',
        'ds_aoi_eta', 'belief_inference_mode', 'tx_power_w', 'mac_noise_power_w',
        'enable_tx_power_control', 'enable_peer_dissemination',
        'enable_data_mule_forwarding', 'comm_action_mode',
        'safety_failure_terminate', 'obs_pool_mode', 'obs_schema_v2',
        'exploration_reward_mode', 'enable_3d_sensing_occlusion',
        'enable_accel_energy', 'e_max_j', 'w_p2p', 'lambda_b',
        'continuous_aoi_penalty_w', 'r_tp', 'r_fp', 'w_loc_team', 'w_aoi',
    ]
    result = {'scenario_status': 'NOT_FROZEN',
              'scenario_sha256': hashlib.sha256(SCENARIO.read_bytes()).hexdigest(),
              'learner_sha256': hashlib.sha256(LEARNER.read_bytes()).hexdigest(),
              'source_paths': {'scenario': str(SCENARIO.relative_to(ROOT)),
                               'learner': str(LEARNER.relative_to(ROOT))},
              'active_fields': {k: data[k] for k in fields},
              'fixed_gcs_position_m': GCS_POS_M,
              'gcs_z_interpretation': 'fixed elevated antenna, not flying UAV',
              'buildings_5': CANONICAL_BUILDINGS_5,
              'blackout_centers_2': CANONICAL_BLACKOUT_CENTERS_2,
              'max_packet_entries': MAX_ENTRIES_PER_PKT,
              'policy_continuous_actions': ['vx_norm', 'vy_norm'],
              'policy_discrete_actions_learned': [],
              'communication_action_rule': 'event_driven_comm_action_np/torch',
              'active_belief_backend': 'dempster_shafer',
              'historical_bayes_backend_in_source': True,
              'open_claims': ['SimPy event-level MAC parity', 'correlated-sensor mission FP',
                              'urban RF per-building material calibration',
                              'full CPU/GPU scenario output parity',
                              '900-step untruncated natural-world efficacy'],
              }
    return result


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot(), indent=2, ensure_ascii=False) + '\n')
    print('SCENARIO_INVENTORY_PASS', args.output)
