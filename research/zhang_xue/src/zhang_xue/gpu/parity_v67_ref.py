"""Pure-NumPy SoA Parity Reference Engine for Warp v6.7 (`make_shapes_v67`).

Executes the exact v6.7 Data-Mule + `ack_origin_gen` (`int16`) + `compact_3` logic
on the same fixed-dtype NumPy SoA dictionary in CPU-only sandboxes without CUDA/Warp.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .layout_v67 import ENV_SCALAR_FIELDS_V67, K_V67, Q_V67, make_shapes_v67
from .runtime_v67 import resolve_compact3_actions_v67_np as resolve_compact3_actions_np
from .state_v67 import PKT_DATA


def _pack_w0(scalable: bool, org: int, cid: int, gen: int) -> np.uint32:
    if not scalable:
        return np.uint32((org & 0x7) | ((cid & 0x1FF) << 3) | ((gen & 0xFFFFF) << 12))
    return np.uint32((org & 0x1F) | ((cid & 0xFFF) << 5) | ((gen & 0x7FFF) << 17))


def _unpack_w0(scalable: bool, w0: int) -> tuple[int, int, int]:
    if not scalable:
        return w0 & 0x7, (w0 >> 3) & 0x1FF, (w0 >> 12) & 0xFFFFF
    return w0 & 0x1F, (w0 >> 5) & 0xFFF, (w0 >> 17) & 0x7FFF


class NumPySoARefV67:
    """NumPy value/ideal-hop reference; NOT packet-event parity with Warp."""

    def __init__(
        self,
        num_envs: int = 1,
        num_uavs: int = 6,
        grid_w: int = 20,
        enable_data_mule: bool = True,
        enable_peer: bool = True,
        aoi_ewma: bool = True,
        aoi_eta: float = 0.02,
    ):
        self.envs = int(num_envs)
        self.num_uavs = int(num_uavs)
        self.radio_n = self.num_uavs + 1
        self.gcs_id = self.num_uavs
        self.grid_w = int(grid_w)
        self.num_cells = self.grid_w * self.grid_w
        self.scalable = (self.num_uavs > 8 or self.num_cells > 512)
        self.enable_data_mule = bool(enable_data_mule)
        self.enable_peer = bool(enable_peer)
        self.aoi_ewma = bool(aoi_ewma)
        self.aoi_eta = float(aoi_eta)
        self.data: Dict[str, np.ndarray] = {}
        for name, (sh, dt) in make_shapes_v67(self.num_uavs, self.grid_w).items():
            full_sh = (self.envs,) if name in ENV_SCALAR_FIELDS_V67 else (self.envs, *sh)
            self.data[name] = np.zeros(full_sh, dtype=dt)
        self.reset()

    def reset(self) -> None:
        """Reset *every* SoA field, not just sentinels, between episodes.

        This is an ideal-delivery value reference, not a MAC event simulator.
        """
        for array in self.data.values():
            array.fill(0)
        self.data["radio_alive"].fill(1)
        self.data["ledger_gen"].fill(-1)
        self.data["ack_origin_gen"].fill(-1)
        self.data["gcs_latest"].fill(-1)
        self.data["gcs_last_rx_us"].fill(-1)
        self.data["route_hop"].fill(-1)
        self.data["route_dist"].fill(127)
        self.data["hello_dv_hop"].fill(-1)
        self.data["hello_dv_dist"].fill(127)
        self.data["active_slot"].fill(-1)
        self.data["next_hop"].fill(-1)
        self.data["rx_lock"].fill(-1)
        self.data["deadline"].fill(2_000_000_000)
        self.data["pending_dst"].fill(-1)
        self.data["pending_seq"].fill(-1)
        self.data["delivered_seq"].fill(-1)
        self.data["received_pid"].fill(-1)
        self.data["delivered_ack_seq"].fill(-1)
        self.data["next_seq"].fill(1)
        self.data["next_pid"].fill(1)

    def sense_batch(self, cell_id: np.ndarray, detected: np.ndarray, t_s: int) -> None:
        for e in range(self.envs):
            for u in range(self.num_uavs):
                cid = int(cell_id[e, u])
                if cid < 0 or cid >= self.num_cells:
                    continue
                det = int(detected[e, u])
                if self.aoi_ewma:
                    last_t = int(self.data["ledger_gen"][e, u, u, cid])
                    ep = float(self.data["ledger_pos"][e, u, u, cid]) / 256.0
                    en = float(self.data["ledger_neg"][e, u, u, cid]) / 256.0
                    if last_t >= 0 and t_s > last_t:
                        d = math.exp(-self.aoi_eta * float(t_s - last_t))
                        ep *= d
                        en *= d
                    if det:
                        ep += 1.0
                    else:
                        en += 1.0
                    self.data["ledger_pos"][e, u, u, cid] = np.uint16(min(65535, max(0, round(ep * 256.0))))
                    self.data["ledger_neg"][e, u, u, cid] = np.uint16(min(65535, max(0, round(en * 256.0))))
                self.data["ledger_gen"][e, u, u, cid] = np.int16(min(32767, max(0, t_s)))

    def compute_dirty_dest(self) -> np.ndarray:
        out = self.data["dirty_dest"]
        out.fill(0)
        c_n = self.num_cells
        for e in range(self.envs):
            for u in range(self.num_uavs):
                if self.data["radio_alive"][e, u] == 0:
                    continue
                for dst in range(self.radio_n):
                    if dst == u:
                        continue
                    g_own = self.data["ledger_gen"][e, u, u]
                    wm_own = self.data["ack_origin_gen"][e, u, dst, u * c_n : (u + 1) * c_n]
                    if np.any((g_own >= 0) & (g_own > wm_own)):
                        out[e, u, dst] = 1
                        continue
                    if self.enable_data_mule:
                        for org in range(self.num_uavs):
                            if org == u or org == dst:
                                continue
                            g_org = self.data["ledger_gen"][e, u, org]
                            wm_org = self.data["ack_origin_gen"][e, u, dst, org * c_n : (org + 1) * c_n]
                            if np.any((g_org >= 0) & (g_org > wm_org)):
                                out[e, u, dst] = 1
                                break
        return out.astype(bool)

    def resolve_compact3(self, actions: np.ndarray, pos_xy: np.ndarray) -> np.ndarray:
        dirty = self.compute_dirty_dest()
        can_tx = self.data["radio_alive"][:, : self.num_uavs].astype(bool)
        return resolve_compact3_actions_np(
            actions, pos_xy, can_tx, enable_peer_dissemination=self.enable_peer, dirty_dest=dirty, can_tx=can_tx,
        )

    def inject_and_deliver_ideal(self, dest: np.ndarray) -> int:
        """Idealized single-hop custody transfer + E2E ACK watermark advance for CPU parity tests."""
        delivered = 0
        c_n = self.num_cells
        for e in range(self.envs):
            for u in range(self.num_uavs):
                dst = int(dest[e, u])
                if dst < 0 or dst == u or self.data["radio_alive"][e, u] == 0 or self.data["radio_alive"][e, dst] == 0:
                    continue
                origins = [u] + ([o for o in range(self.num_uavs) if o != u and o != dst] if self.enable_data_mule else [])
                sent = []
                for org in origins:
                    g_org = self.data["ledger_gen"][e, u, org]
                    wm = self.data["ack_origin_gen"][e, u, dst, org * c_n : (org + 1) * c_n]
                    for cid in np.flatnonzero((g_org >= 0) & (g_org > wm)):
                        w0 = _pack_w0(self.scalable, org, int(cid), int(g_org[cid]))
                        w1 = int(self.data["ledger_pos"][e, u, org, cid]) | (int(self.data["ledger_neg"][e, u, org, cid]) << 16)
                        sent.append((w0, w1))
                        if len(sent) >= K_V67:
                            break
                    if len(sent) >= K_V67:
                        break
                if not sent:
                    continue
                delivered += 1
                for w0, w1 in sent:
                    org, cid, gen = _unpack_w0(self.scalable, int(w0))
                    np_v = np.uint16(w1 & 0xFFFF)
                    nn_v = np.uint16((w1 >> 16) & 0xFFFF)
                    if gen > int(self.data["ledger_gen"][e, dst, org, cid]):
                        self.data["ledger_pos"][e, dst, org, cid] = np_v
                        self.data["ledger_neg"][e, dst, org, cid] = nn_v
                        self.data["ledger_gen"][e, dst, org, cid] = np.int16(gen)
                    if gen > int(self.data["ack_origin_gen"][e, u, dst, org * c_n + cid]):
                        self.data["ack_origin_gen"][e, u, dst, org * c_n + cid] = np.int16(gen)
        return delivered
