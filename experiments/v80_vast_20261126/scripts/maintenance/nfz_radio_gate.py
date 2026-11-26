#!/usr/bin/env python3
"""Reproducible RF obstruction + hidden-terminal physics gate.

Runs a CPU packet collision *controlled overlap* fixture and independent NumPy
PHY reference. Optional Warp CPU/CUDA executes true kernels. No full RL claims.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]

from zhang_xue.communication.common import (INF_TIME_US, PKT_DATA, ST_TX_DATA,
                                            Packet, link_path_gain)
from zhang_xue.communication.network import SevenRadioCsmaCaNetwork
from zhang_xue.gpu.kernels.phy_v67 import compute_channel_gain_v67_np


POSITIONS = np.array([[[100., 100., 50.],
                       [400., 100., 50.],
                       [250., 230., 50.]]], dtype=np.float32)
OBSTACLES = ((175., 100., 20., 85.),
             (250., 100., 20., 85.),
             (325., 100., 20., 85.))
MODELS = ("binary_nlos", "multi_obstacle", "rf_opaque")


def cpu_gate(loss_db: float = 20.0) -> dict:
    if not math.isfinite(loss_db) or loss_db < 0:
        raise ValueError("loss_db must be finite and nonnegative")
    cases = {}
    for mode in MODELS:
        cpu = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                if i != j:
                    cpu[i, j] = link_path_gain(POSITIONS[0, i], POSITIONS[0, j],
                        OBSTACLES, (), obstacle_model=mode, per_building_loss_db=loss_db)
        ref = compute_channel_gain_v67_np(POSITIONS, OBSTACLES, (), 0, 20, 50.,
            obstacle_model=mode, per_building_loss_db=loss_db)[0]
        np.testing.assert_allclose(cpu, ref, rtol=2e-4, atol=1e-15)
        cases[mode] = {
            "gain_tx0_tx1": float(cpu[0, 1]),
            "gain_tx0_gcs": float(cpu[0, 2]),
            "gain_tx1_gcs": float(cpu[1, 2]),
            "max_abs_diff_vs_numpy": float(np.max(np.abs(cpu - ref))),
        }
    net = SevenRadioCsmaCaNetwork(
        num_uavs=2, buildings=OBSTACLES, blackout_centers=[], tx_power_w=0.1,
        noise_power_w=1.e-13, rf_obstacle_model="multi_obstacle",
        rf_per_building_loss_db=loss_db, seed=44)
    net.reset(POSITIONS[0], episode_id=2)
    net.hello_next_us = [INF_TIME_US] * 3
    rx_power = 0.1 * net.chan_gain
    hidden = bool(rx_power[0, 1] < net.cca_th_w and
                  rx_power[0, 2] > net.cca_th_w and
                  rx_power[1, 2] > net.cca_th_w)
    if not hidden:
        raise AssertionError("Geometry fails controlled hidden-terminal conditions")
    for src in (0, 1):
        net.radios[src].active_pkt = Packet(
            pkt_id=src+1, episode_id=2, pkt_type=PKT_DATA,
            app_src=src, app_dst=net.gcs_id, app_seq=src+1,
            create_us=0, expire_us=1_000_000, visited_mask=1 << src)
        net.radios[src].active_next_hop = net.gcs_id
    net._start_tx(0, is_ack=False)
    if net.radios[0].mac_state != ST_TX_DATA or net._is_channel_busy_at(1):
        raise AssertionError("First sender not hidden from second sender")
    net._start_tx(1, is_ack=False)
    net._finish_tx(0)
    if net.stats["sinr_drops"] <= 0 or net.stats["data_delivered_gcs"]:
        raise AssertionError("Controlled overlap failed to corrupt GCS frame")
    return {
        "status": "CPU_PHY_AND_CONTROLLED_OVERLAP_PASS_NOT_MAC_DISTRIBUTION_PARITY",
        "loss_db_per_intersected_cylinder": loss_db,
        "cases": cases,
        "hidden_terminal_cca": hidden,
        "sinr_drops_controlled_overlap": net.stats["sinr_drops"],
        "interpretation": "Deterministically forced simultaneous starts; not a randomized MAC collision-rate benchmark",
    }


def warp_gate(device: str = 'cpu', loss_db: float = 20.) -> dict:
    try:
        import warp as wp
    except ImportError:
        return {"status":"BLOCKED_NO_WARP", "gpu_executed":False}
    if device.startswith('cuda:'):
        if len(wp.get_cuda_devices()) <= int(device.split(':')[1]):
            return {"status":"BLOCKED_CUDA_REQUIRED", "gpu_executed":False}
    elif device != 'cpu':
        raise ValueError('Only cpu or cuda:N devices supported')
    from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
    radio = WarpCompressedRadioNetworkV67(1,num_uavs=2,grid_w=20,
                                         q_cap=64,device=device,seed=44)
    cases = {}
    for mode in MODELS:
        radio.update_phy_from_positions(POSITIONS,OBSTACLES,(),0,50.,
            obstacle_model=mode,per_building_loss_db=loss_db)
        got=radio.data['gain'].numpy()
        ref=compute_channel_gain_v67_np(POSITIONS,OBSTACLES,(),0,20,50.,
            obstacle_model=mode,per_building_loss_db=loss_db)
        np.testing.assert_allclose(got,ref,rtol=2e-4,atol=1e-15)
        cases[mode] = {"max_abs_diff":float(np.max(np.abs(got-ref))),
                       "gain_tx0_tx1":float(got[0,0,1])}
    return {"status":"WARP_PHY_REFERENCE_PASS_NOT_FULL_RL", "device":device,
            "gpu_executed":device.startswith('cuda:'), "cases":cases,
            "not_full_training":True}


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='none',help='none|cpu|cuda:N')
    p.add_argument('--loss-db',type=float,default=20.)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    out={"cpu":cpu_gate(args.loss_db)}
    if args.device != 'none':
        out['warp']=warp_gate(args.device,args.loss_db)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(out,indent=2,allow_nan=False))
    return 0 if out['cpu']['status'].startswith('CPU_PHY_') and (
        args.device=='none' or out['warp']['status'].startswith('WARP_PHY_')) else 3


if __name__ == '__main__':
    raise SystemExit(main())
