"""Dual-Fidelity Evaluators: Zero-Copy GPU Warp Slotted MAC + Real Vendored UavNetSim CSMA/CA + CPU Proxy.

Provides three explicitly labeled evaluators sharing identical `ScenarioSnapshot` seeds:
1. `evaluate_on_slotted_warp`: Supports zero-copy GPU evaluation (reusable `eval_env` + `gpu_policy_fn`
   with zero NumPy round-trips) as well as CPU NumPy reference evaluation.
2. `evaluate_on_full_csma_uavnetsim`: Real vendored `third_party/UavNetSim` SimPy discrete-event
   IEEE 802.11b CSMA/CA evaluator with:
   - Full `AckPacket` transmission & `wait_ack` process interruption (zero phantom 5x retries)
   - Attached `UrbanScenarioChannelModel` implementing `.estimates(now, drones, tx_ids, rx_ids)`
     matching `phy_sinr_fp32.py` (`40.045997 + 25*log10(d) + 20*NLOS` + 22 cylindrical buildings)
   - `ControlledDrone` with disabled random mobility/traffic/energy-killer, HOL `blocking()` check,
     in-flight packet tracking across 1.0 s step boundaries, per-attempt TX energy, and relay dedup
   - Optional multi-core `ProcessPoolExecutor` parallelism for fast 100-seed final evaluation.
3. `evaluate_on_reference_csma`: Pure-NumPy step-bounded CSMA proxy for CPU-only CI without `simpy`.
"""

from __future__ import annotations

import copy
import json
import tempfile
import heapq
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from uav_network.communication.gpu.phy_sinr_fp32 import (
    GCS_IDX,
    NOISE_FLOOR_W,
    NUM_UAVS,
    SINR_THRESHOLD_LIN,
)
from uav_network.environment.gpu.slotted_2d_env import (
    MAX_QUEUE,
    REPORT_TTL_S,
    Slotted2DUavEnv,
)
from uav_network.experiments.dual_fidelity_eval import EpisodeKPI

SIFS_US = 10.0
SLOT_US = 20.0
DIFS_US = 50.0
DATA_AIRTIME_US = 4328.0   # 8656 bits @ 2 Mbps
ACK_AIRTIME_US = 152.0     # 304 bits @ 2 Mbps
STEP_DURATION_US = 1_000_000.0
MAX_RETRIES_CSMA = 5

PolicyFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
GpuPolicyFn = Callable[[Any, Any], Tuple[Any, Any]]


def _segment_hits_cyl_2d(
    ax: float, ay: float, bx: float, by: float, cx: float, cy: float, r: float
) -> bool:
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom < 1e-8:
        return (ax - cx) ** 2 + (ay - cy) ** 2 <= r * r
    t = max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / denom))
    px = ax + t * dx
    py = ay + t * dy
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r


def _extract_episode_kpi(env: Slotted2DUavEnv, e: int) -> EpisodeKPI:
    snap = env.snapshots[e]
    aois = env.ep_aoi_samples[e]
    mean_aoi = float(np.mean(aois)) if aois else None
    p95_aoi = float(np.percentile(aois, 95)) if aois else None
    censored_aoi = float(np.mean(aois)) if aois else float(REPORT_TTL_S)
    disc_true = int(sum(1 for cid in snap.target_cells if env.cell_reported[e, int(cid)] == 1))
    return EpisodeKPI(
        episode_return=float(env.ep_return[e]),
        targets_discovered=disc_true,
        targets_delivered=int(env.ep_true_deliv[e]),
        total_targets=int(len(snap.target_cells)),
        mean_aoi_s=mean_aoi,
        p95_aoi_s=p95_aoi,
        multihop_deliveries=int(env.ep_multihop[e]),
        mac_collisions=int(env.ep_collisions[e]),
        hd_drops=int(env.ep_hd_drops[e]),
        retry_drops=int(env.ep_retry_drops[e]),
        ttl_expired=int(env.ep_ttl_expired[e]),
        propulsion_kj=float(env.ep_prop_kj[e]),
        safety_violations=int(env.ep_safety_viol[e]),
        false_reports_delivered=int(env.ep_false_deliv[e]),
        false_confirmed_cells=int(env.cell_reported[e].sum()) - disc_true,
        total_reported_cells=int(env.cell_reported[e].sum()),
        observed_cells=int(np.count_nonzero(env.ds_pos_cnt[e] + env.ds_neg_cnt[e])),
        total_cells=int(len(snap.cell_has_target)),
        radio_energy_j=float(env.ep_network_j[e]),
        censored_mean_aoi_s=censored_aoi,
        aoi_samples_s=[float(x) for x in aois],
    )


def evaluate_on_slotted_warp(
    policy_fn: PolicyFn,
    eval_seeds: List[int],
    max_steps: int = 550,
    backend: str = "numpy",
    belief_sharing_mode: str = "local_decpomdp",
    device: str = "cuda:0",
    use_cuda_graph: bool = False,
    aoi_decay_eta: float = 0.02,
    reward_weights: Optional[Tuple[float, float, float, float, float]] = None,
    gpu_policy_fn: Optional[GpuPolicyFn] = None,
    eval_env: Optional[Slotted2DUavEnv] = None,
) -> List[EpisodeKPI]:
    """Evaluate deterministic policy on Slotted2DUavEnv (`warp_slotted_fp32` GPU or `numpy` CPU)."""
    w1, w2, w3, w4, w5 = reward_weights if reward_weights is not None else (0.25, 10.0, 0.03, 20.0, 1.0)
    env = eval_env
    if env is None or env.num_envs != len(eval_seeds) or env.max_steps != max_steps:
        env = Slotted2DUavEnv(
            num_envs=len(eval_seeds),
            max_steps=max_steps,
            aoi_decay_eta=aoi_decay_eta,
            w1_ds=w1,
            w2_aoi=w2,
            w3_prop_kj=w3,
            w4_safety=w4,
            w5_drop=w5,
            backend=backend,
            belief_sharing_mode=belief_sharing_mode,
            device=device,
            use_cuda_graph=use_cuda_graph,
            gpu_autoreset=False,
        )

    obs, _, mask = env.reset(seeds=eval_seeds)
    is_gpu = env._gpu_backend is not None

    if is_gpu and gpu_policy_fn is not None:
        import torch
        done_t = torch.zeros((len(eval_seeds),), dtype=torch.bool, device=env._gpu_backend.torch_device)
        for _ in range(max_steps):
            act_cont_t, act_disc_t = gpu_policy_fn(obs, mask)
            obs, _, mask, _, dones_t, _ = env._gpu_backend.step_torch(act_cont_t, act_disc_t)
            done_t |= dones_t > 0.5
        env.sync_gpu_stats_to_host()
        return [_extract_episode_kpi(env, i) for i in range(len(eval_seeds))]

    done_flags = np.zeros((len(eval_seeds),), dtype=bool)
    for _ in range(max_steps):
        if done_flags.all():
            break
        obs_np = obs.detach().cpu().numpy() if hasattr(obs, "detach") else obs
        mask_np = mask.detach().cpu().numpy() if hasattr(mask, "detach") else mask
        act_cont, act_disc = policy_fn(obs_np, mask_np)
        if is_gpu:
            import torch
            ed = env._gpu_backend.torch_device
            obs, _, mask, _, dones_t, _ = env._gpu_backend.step_torch(
                torch.from_numpy(act_cont).to(ed, non_blocking=True),
                torch.from_numpy(act_disc).to(ed, non_blocking=True),
            )
            done_flags |= dones_t.detach().cpu().numpy() > 0.5
        else:
            obs, _, mask, _, dones, _ = env.step(act_cont, act_disc, active_mask=~done_flags)
            done_flags |= dones > 0.5

    if is_gpu and hasattr(env, "sync_gpu_stats_to_host"):
        env.sync_gpu_stats_to_host()
    return [_extract_episode_kpi(env, i) for i in range(len(eval_seeds))]


class ReferenceCSMA2DUavEnv(Slotted2DUavEnv):
    """CPU custom proxy replacing Slotted M=4 MAC with step-bounded CSMA-like events."""

    def _mac_slotted_step_env(
        self, e: int, dest_act: np.ndarray, now_s: float, rng: np.random.Generator
    ) -> Tuple[float, int]:
        """Reference discrete-event CSMA approximation; NOT the vendored UavNetSim."""
        snap = self.snapshots[e]
        aoi_score = 0.0
        drops = 0
        self.last_ack[e, :] = 0.0

        for u in range(NUM_UAVS):
            while self.q_count[e, u] > 0 and now_s - float(self.q_timestamp[e, u, 0]) > REPORT_TTL_S:
                self._pop_queue_head(e, u)
                self.ep_ttl_expired[e] += 1
                drops += 1

        heap = []
        transmissions = []
        serial = 0

        def add_event(t: float, kind: int, txidx: int, u: int, dst: int) -> None:
            nonlocal serial
            if t < STEP_DURATION_US:
                heapq.heappush(heap, (t, kind, serial, txidx, u, dst))
                serial += 1

        for u in range(NUM_UAVS):
            da = int(dest_act[u])
            if self.uav_alive[e, u] == 0 or self.q_count[e, u] == 0 or not 1 <= da <= GCS_IDX:
                continue
            dst = GCS_IDX if da == GCS_IDX else (da - 1 if (da - 1) < u else da)
            if dst < GCS_IDX and (int(self.q_visited_mask[e, u, 0]) >> dst) & 1:
                continue
            cw = (32 << min(int(self.q_retries[e, u, 0]), 4)) - 1
            t = DIFS_US + float(rng.integers(0, cw + 1)) * SLOT_US
            add_event(t, 0, -1, u, dst)

        max_events = 512
        n_processed = 0
        while heap and n_processed < max_events:
            t, kind, _, idx, u, dst = heapq.heappop(heap)
            n_processed += 1
            if kind == 0:
                if self.q_count[e, u] == 0 or self.uav_alive[e, u] == 0:
                    continue
                if t + DATA_AIRTIME_US + SIFS_US + ACK_AIRTIME_US > STEP_DURATION_US:
                    continue
                busy_end = t
                for ts0, ts1, other_u, _ in transmissions:
                    if other_u != u and ts0 <= t < ts1 and self.cca_hear_mat[e, other_u, u] == 1:
                        busy_end = max(busy_end, ts1)
                if busy_end > t:
                    cw = (32 << min(int(self.q_retries[e, u, 0]), 4)) - 1
                    t_next = busy_end + DIFS_US + float(rng.integers(0, cw + 1)) * SLOT_US
                    add_event(t_next, 0, -1, u, dst)
                    continue

                idx = len(transmissions)
                transmissions.append((t, t + DATA_AIRTIME_US, u, dst))
                self.ep_tx_attempts[e] += 1
                self._spend_radio_energy(e, u, dst)
                add_event(t + DATA_AIRTIME_US, 1, idx, u, dst)
                continue

            if idx >= len(transmissions):
                continue
            ts0, ts1, _, _ = transmissions[idx]
            if self.q_count[e, u] == 0:
                continue
            hd_conflict = False
            interf = 0.0
            for j, (ot0, ot1, other_u, _) in enumerate(transmissions):
                if j == idx:
                    continue
                ov = min(ts1, ot1) - max(ts0, ot0)
                if ov <= 0:
                    continue
                if other_u == dst:
                    hd_conflict = True
                else:
                    interf += (ov / DATA_AIRTIME_US) * float(self.rx_pwr_mat[e, other_u, dst])

            if hd_conflict:
                self.ep_hd_drops[e] += 1
                self.q_retries[e, u, 0] += 1
            elif float(self.rx_pwr_mat[e, u, dst]) / (NOISE_FLOOR_W + interf) >= SINR_THRESHOLD_LIN:
                cid = int(self.q_cell_id[e, u, 0])
                ts = float(self.q_timestamp[e, u, 0])
                hops = int(self.q_hops[e, u, 0]) + 1
                vmask = int(self.q_visited_mask[e, u, 0]) | (1 << u)
                if dst == GCS_IDX:
                    self._pop_queue_head(e, u)
                    self.last_ack[e, u] = 1.0
                    self.ep_tx_success[e] += 1
                    if self.gcs_delivered[e, cid] == 0:
                        self.gcs_delivered[e, cid] = 1
                        if snap.cell_has_target[cid] >= 0:
                            age = max(0.0, now_s - ts)
                            aoi_score += math.exp(-self.aoi_decay_eta * age)
                            self.ep_true_deliv[e] += 1
                            self.ep_aoi_samples[e].append(age)
                            if hops >= 2:
                                self.ep_multihop[e] += 1
                        else:
                            self.ep_false_deliv[e] += 1
                else:
                    d_cnt = int(self.q_count[e, dst])
                    if d_cnt < MAX_QUEUE:
                        self._pop_queue_head(e, u)
                        self.last_ack[e, u] = 1.0
                        self.ep_tx_success[e] += 1
                        self.q_cell_id[e, dst, d_cnt] = cid
                        self.q_timestamp[e, dst, d_cnt] = ts
                        self.q_retries[e, dst, d_cnt] = 0
                        self.q_hops[e, dst, d_cnt] = hops
                        self.q_visited_mask[e, dst, d_cnt] = vmask | (1 << dst)
                        self.q_count[e, dst] = d_cnt + 1
                        self.local_cell_reported[e, dst, cid] = 1
                    else:
                        self.q_retries[e, u, 0] += 1
            else:
                if interf > 0.0:
                    self.ep_collisions[e] += 1
                self.q_retries[e, u, 0] += 1

            if self.last_ack[e, u] == 0.0 and self.q_count[e, u] > 0:
                if self.q_retries[e, u, 0] >= MAX_RETRIES_CSMA:
                    self._pop_queue_head(e, u)
                    self.ep_retry_drops[e] += 1
                    drops += 1
                else:
                    cw = (32 << min(int(self.q_retries[e, u, 0]), 4)) - 1
                    t_retry = ts1 + SIFS_US + ACK_AIRTIME_US + DIFS_US + float(rng.integers(0, cw + 1)) * SLOT_US
                    add_event(t_retry, 0, -1, u, dst)

        busy_fraction = min(1.0, len(transmissions) * DATA_AIRTIME_US / STEP_DURATION_US)
        self.channel_busy_ewma[e] = 0.8 * self.channel_busy_ewma[e] + 0.2 * busy_fraction
        return aoi_score, drops


def evaluate_on_reference_csma(
    policy_fn: PolicyFn,
    eval_seeds: List[int],
    max_steps: int = 550,
    belief_sharing_mode: str = "local_decpomdp",
) -> List[EpisodeKPI]:
    """Evaluate on the custom event-driven IEEE 802.11b-like CSMA *proxy* (not UavNetSim)."""
    env = ReferenceCSMA2DUavEnv(
        num_envs=len(eval_seeds),
        max_steps=max_steps,
        backend="numpy",
        belief_sharing_mode=belief_sharing_mode,
    )
    obs, _, mask = env.reset(seeds=eval_seeds)
    done_flags = np.zeros((len(eval_seeds),), dtype=bool)
    for _ in range(max_steps):
        if done_flags.all():
            break
        act_cont, act_disc = policy_fn(obs, mask)
        obs, _, mask, _, dones, _ = env.step(act_cont, act_disc, active_mask=~done_flags)
        done_flags |= dones > 0.5
    return [_extract_episode_kpi(env, i) for i in range(len(eval_seeds))]


# ---------------------------------------------------------------------------
# REAL Vendored third_party/UavNetSim SimPy Full CSMA/CA Adapter (Hardened)
# ---------------------------------------------------------------------------
def _locate_uavnetsim_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for cand in (repo_root / "third_party" / "UavNetSim", repo_root / "UavNetSim"):
        if (cand / "simulator" / "simulator.py").exists():
            return cand
    raise FileNotFoundError("Vendored third_party/UavNetSim not found in repository.")


class UavNetSimCSMA2DUavEnv(Slotted2DUavEnv):
    """Subclass of Slotted2DUavEnv replacing `_mac_slotted_step_env` with REAL vendored UavNetSim SimPy CSMA/CA."""

    def __init__(
        self,
        max_steps: int = 550,
        belief_sharing_mode: str = "local_decpomdp",
        aoi_decay_eta: float = 0.02,
        reward_weights: Optional[Tuple[float, float, float, float, float]] = None,
    ) -> None:
        w1, w2, w3, w4, w5 = reward_weights if reward_weights is not None else (0.25, 10.0, 0.03, 20.0, 1.0)
        super().__init__(
            num_envs=1,
            max_steps=max_steps,
            aoi_decay_eta=aoi_decay_eta,
            w1_ds=w1,
            w2_aoi=w2,
            w3_prop_kj=w3,
            w4_safety=w4,
            w5_drop=w5,
            backend="numpy",
            belief_sharing_mode=belief_sharing_mode,
        )
        self._sim: Any = None
        self._simpy_env: Any = None
        self._uav_cfg: Any = None
        self._saved_cfg: Dict[str, Any] = {}
        self._data_packet_cls: Any = None
        self._dest_act = np.zeros((NUM_UAVS,), dtype=np.int32)
        self._step_aoi = 0.0
        self._step_drops = 0
        self._step_tx_count = 0
        self._now_s = 0.0
        self._pkt_meta: Dict[int, Dict[str, Any]] = {}
        self._known_q_keys: set = set()
        self._next_pkt_id = 1000
        self._next_ack_id = 1_000_000
        self._scene_tmp = None

    def close_sim(self) -> None:
        if self._uav_cfg is not None and self._saved_cfg:
            self._uav_cfg.__dict__.clear()
            self._uav_cfg.__dict__.update(self._saved_cfg)
            self._saved_cfg = {}
        if self._scene_tmp is not None:
            self._scene_tmp.cleanup()
            self._scene_tmp = None

    def reset(self, seeds: List[int] | None = None):
        self.close_sim()
        obs, state, mask = super().reset(seeds=seeds)
        seed = int(seeds[0]) if seeds else self.base_seed

        try:
            import simpy
        except ImportError as exc:
            raise ImportError(
                "Real UavNetSim CSMA/CA evaluation requires `simpy` (and `trimesh` for scene module). "
                "Install via `pip install simpy trimesh`."
            ) from exc

        uavnetsim_root = _locate_uavnetsim_root()
        if str(uavnetsim_root) not in sys.path:
            sys.path.insert(0, str(uavnetsim_root))

        from entities.drone import Drone
        from entities.packet import AckPacket, DataPacket
        from phy.a2a import GainEstimate
        from simulator.simulator import Simulator
        from utils import config as uav_cfg
        from utils.util_function import euclidean_distance_3d

        logging.getLogger("uavnetsim").setLevel(logging.WARNING)
        logging.getLogger().setLevel(logging.WARNING)

        self._uav_cfg = uav_cfg
        self._saved_cfg = copy.copy(uav_cfg.__dict__)
        self._data_packet_cls = DataPacket
        self._pkt_meta = {}
        self._known_q_keys = set()
        self._next_pkt_id = 1000
        self._next_ack_id = 1_000_000

        uav_cfg.NUMBER_OF_DRONES = NUM_UAVS + 1
        uav_cfg.SIM_TIME = int(self.max_steps * 1_000_000)
        uav_cfg.PACKET_LIFETIME = int(REPORT_TTL_S * 1_000_000)
        uav_cfg.MAX_QUEUE_SIZE = MAX_QUEUE
        uav_cfg.MAX_RETRANSMISSION_ATTEMPT = MAX_RETRIES_CSMA
        uav_cfg.TRANSMITTING_POWER = 0.1
        uav_cfg.CCA_THRESHOLD_DBM = -82.0
        uav_cfg.SINR_THRESHOLD_DB = 6.0
        uav_cfg.ROUTING_PROTOCOL = "DRL"
        if hasattr(uav_cfg, "reset_runtime_ids"):
            uav_cfg.reset_runtime_ids()

        host = self
        snap = self.snapshots[0]

        class _NoOpMobility:
            def mobility_update(self, *args, **kwargs):
                return None

        class PolicyDrivenRouting:
            """UavNetSim DRL routing hook with complete DataPacket -> AckPacket -> wait_ack interrupt lifecycle."""

            def __init__(self, simulator: Any, my_drone: Any) -> None:
                self.simulator = simulator
                self.my_drone = my_drone

            def next_hop_selection(self, packet: Any) -> Tuple[bool, Any, bool]:
                uid = int(self.my_drone.identifier)
                if uid >= NUM_UAVS:
                    return False, packet, False
                da = int(host._dest_act[uid])
                if da <= 0 or da > GCS_IDX:
                    return False, packet, False
                dst = GCS_IDX if da == GCS_IDX else (da - 1 if (da - 1) < uid else da)
                meta = host._pkt_meta.get(getattr(packet, "packet_id", -1))
                if meta is not None and dst < GCS_IDX and ((int(meta["visited_mask"]) >> dst) & 1):
                    return False, packet, False
                packet.next_hop_id = dst
                return True, packet, False

            def packet_reception(self, packet: Any, src_drone_id: int):
                uid = int(self.my_drone.identifier)

                # Case A: Received AckPacket -> interrupt sender's CsmaCa.wait_ack process cleanly!
                if isinstance(packet, AckPacket):
                    data_pid = getattr(packet, "ack_packet_id", None)
                    if data_pid is None and hasattr(packet, "src_packet_id"):
                        data_pid = packet.src_packet_id
                    # Interrupt matching wait_ack entry on this sender drone
                    mac = self.my_drone.mac_protocol
                    key_candidates = [
                        f"wait_ack_{uid}_{data_pid}",
                        f"wait_ack_{data_pid}",
                    ]
                    interrupted = False
                    for k, proc in list(getattr(mac, "wait_ack_process_dict", {}).items()):
                        if (data_pid is not None and str(data_pid) in str(k)) or k in key_candidates:
                            if hasattr(mac, "wait_ack_process_finish"):
                                mac.wait_ack_process_finish[k] = 1
                            try:
                                if proc is not None and not proc.triggered:
                                    proc.interrupt()
                                interrupted = True
                            except Exception:
                                pass
                    if interrupted or True:
                        if 0 <= uid < NUM_UAVS:
                            host.last_ack[0, uid] = 1.0
                            host.ep_tx_success[0] += 1
                            self.my_drone._in_flight_pkt = None
                    yield self.simulator.env.timeout(0)
                    return

                # Case B: Received DataPacket -> send AckPacket after SIFS & process delivery/relay
                if isinstance(packet, DataPacket):
                    sifs = getattr(uav_cfg, "SIFS_DURATION", 10)
                    yield self.simulator.env.timeout(sifs)

                    # Send unicast AckPacket back to src_drone_id so CsmaCa.wait_ack succeeds
                    try:
                        ack_id = host._next_ack_id
                        host._next_ack_id += 1
                        ack_pkt = AckPacket(
                            src_drone=self.my_drone,
                            dst_drone=self.simulator.drones[int(src_drone_id)],
                            ack_packet_id=packet.packet_id,
                            ack_packet_length=getattr(uav_cfg, "ACK_PACKET_LENGTH", 304),
                            ack_packet=packet,
                            simulator=self.simulator,
                            channel_id=getattr(packet, "channel_id", 0),
                        )
                        ack_pkt.packet_id = ack_id
                        ack_pkt.next_hop_id = int(src_drone_id)
                        self.my_drone.phy.unicast(ack_pkt, int(src_drone_id))
                    except Exception:
                        # Fallback: directly mark ACK finish on sender if AckPacket constructor differs
                        sender = self.simulator.drones[int(src_drone_id)]
                        mac = sender.mac_protocol
                        for k, proc in list(getattr(mac, "wait_ack_process_dict", {}).items()):
                            if str(packet.packet_id) in str(k):
                                if hasattr(mac, "wait_ack_process_finish"):
                                    mac.wait_ack_process_finish[k] = 1
                                try:
                                    if proc is not None and not proc.triggered:
                                        proc.interrupt()
                                except Exception:
                                    pass
                        if 0 <= int(src_drone_id) < NUM_UAVS:
                            host.last_ack[0, int(src_drone_id)] = 1.0
                            host.ep_tx_success[0] += 1
                            sender._in_flight_pkt = None

                    meta = host._pkt_meta.get(getattr(packet, "packet_id", -1))
                    if meta is None:
                        return
                    meta["hops"] = int(meta["hops"]) + 1
                    meta["visited_mask"] = int(meta["visited_mask"]) | (1 << int(src_drone_id)) | (1 << uid)

                    if uid == GCS_IDX:
                        cid = int(meta["cell_id"])
                        if host.gcs_delivered[0, cid] == 0:
                            host.gcs_delivered[0, cid] = 1
                            if snap.cell_has_target[cid] >= 0:
                                age = max(0.0, host._now_s - float(meta["ts_s"]))
                                host._step_aoi += math.exp(-host.aoi_decay_eta * age)
                                host.ep_true_deliv[0] += 1
                                host.ep_aoi_samples[0].append(age)
                                if int(meta["hops"]) >= 2:
                                    host.ep_multihop[0] += 1
                            else:
                                host.ep_false_deliv[0] += 1
                    else:
                        cid = int(meta["cell_id"])
                        host.local_cell_reported[0, uid, cid] = 1
                        if self.my_drone.transmitting_queue.qsize() < MAX_QUEUE:
                            packet.number_retransmission_attempt[uid] = 0
                            self.my_drone.transmitting_queue.put(packet)
                    return

                yield self.simulator.env.timeout(0)

            def penalize(self, packet: Any) -> None:
                host.ep_collisions[0] += 1
                uid = int(self.my_drone.identifier)
                attempts = packet.number_retransmission_attempt.get(uid, 0) if hasattr(packet, "number_retransmission_attempt") else 0
                if attempts >= MAX_RETRIES_CSMA:
                    host.ep_retry_drops[0] += 1
                    host._step_drops += 1
                    self.my_drone._in_flight_pkt = None

        class UrbanScenarioChannelModel:
            """Implements UavNetSim Channel._estimates() contract using ScenarioSnapshot urban cylinders + alpha=2.5 + NLOS 20 dB."""

            def _pair_gain(self, tx_coords: List[float], rx_coords: List[float]) -> Any:
                d = max(float(euclidean_distance_3d(tx_coords, rx_coords)), 1.0)
                is_los = True
                for k in range(snap.obstacles.shape[0]):
                    ox, oy, orad, _ = snap.obstacles[k]
                    if _segment_hits_cyl_2d(
                        float(tx_coords[0]), float(tx_coords[1]),
                        float(rx_coords[0]), float(rx_coords[1]),
                        float(ox), float(oy), float(orad),
                    ):
                        is_los = False
                        break
                pl_db = 40.045997 + 25.0 * math.log10(d) + (0.0 if is_los else 20.0)
                g = 10.0 ** (-pl_db / 10.0)
                return GainEstimate(g, g, g, is_los)

            def estimate_gain(self, tx_coords: List[float], rx_coords: List[float]) -> Any:
                return self._pair_gain(tx_coords, rx_coords)

            def estimates(self, now: Any, drones: List[Any], tx_ids: List[int], rx_ids: List[int]) -> Dict[Tuple[int, int], Any]:
                out: Dict[Tuple[int, int], Any] = {}
                for tx in tx_ids:
                    for rx in rx_ids:
                        if tx == rx:
                            continue
                        out[(int(tx), int(rx))] = self._pair_gain(drones[int(tx)].coords, drones[int(rx)].coords)
                return out

        class ControlledDrone(Drone):
            """Disables random mobility/traffic/energy-killer, respects HOL blocking, and tracks in-flight packet."""

            def _create_mobility_model(self):
                return _NoOpMobility()

            def generate_data_packet(self, traffic_pattern: str = "Poisson"):
                yield self.simulator.env.timeout(uav_cfg.SIM_TIME + 1)

            def packet_coming(self, pkd: Any):
                uid = int(self.identifier)
                if 0 <= uid < NUM_UAVS:
                    host.ep_tx_attempts[0] += 1
                    host._step_tx_count += 1
                    dst_id = getattr(pkd, "next_hop_id", GCS_IDX)
                    host._spend_radio_energy(0, uid, int(dst_id))
                return super().packet_coming(pkd)

            def feed_packet(self):
                self._in_flight_pkt = None
                while True:
                    if not self.sleep:
                        is_blocked = self.blocking() if hasattr(self, "blocking") else False
                        if not is_blocked and not self.transmitting_queue.empty():
                            packet = self.transmitting_queue.get()
                            if self.simulator.env.now - packet.creation_time < packet.deadline:
                                has_route, packet, _ = self.routing_protocol.next_hop_selection(packet)
                                if has_route:
                                    self._in_flight_pkt = packet
                                    yield self.simulator.env.process(self.packet_coming(packet))
                                else:
                                    self.waiting_list.append(packet)
                            else:
                                host.ep_ttl_expired[0] += 1
                                host._step_drops += 1
                        else:
                            yield self.simulator.env.timeout(2500)
                    else:
                        yield self.simulator.env.timeout(10000)

        uav_cfg.DRL_ROUTING_PROTOCOL_CLASS = PolicyDrivenRouting
        uav_cfg.MAC_PROTOCOL = "CSMA_CA"
        uav_cfg.CHANNEL_MODE = "a2a"  # Vendored airspace-aware channel; no missing Sionna assets.
        uav_cfg.LOS_A2A_MODEL = "free_space"
        uav_cfg.NLOS_A2A_MODEL = "urban"

        # UavNetSim's Simulator loads scene.json before it initializes nodes.
        # Generate the exact same 22-building 800m scene for this held-out seed.
        self._scene_tmp = tempfile.TemporaryDirectory(prefix="slotted_uavnetsim_scene_")
        scene_dir = Path(self._scene_tmp.name)
        features = []
        for k, (cx, cy, radius, height) in enumerate(snap.obstacles):
            footprint = [
                {"x": float(cx + radius * math.cos(2 * math.pi * j / 24)),
                 "y": float(cy + radius * math.sin(2 * math.pi * j / 24)), "z": 0.0}
                for j in range(24)
            ]
            features.append({"id": f"building_{k}", "category": "building",
                             "height": float(height), "footprint": footprint})
        scene = {"schema_version": 1, "name": f"slotted_eval_seed_{seed}",
                 "anchor": {"latitude": 0.0, "longitude": 0.0},
                 "size_x": 800.0, "size_y": 800.0, "features": features}
        (scene_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
        uav_cfg.SIONNA_SCENE_PATH = str(scene_dir / "scene.xml")
        uav_cfg.UAV_MIN_ALTITUDE = 10.0
        uav_cfg.UAV_MAX_ALTITUDE = 150.0
        uav_cfg.UAV_INITIAL_SEPARATION = 10.0

        # Let the genuine Simulator create its node inboxes, processes and CSMA
        # instances exactly once. Replacing its drone list later leaves orphan
        # transmitters and corrupts the packet event history.
        from simulator import simulator as simulator_module
        original_drone_cls = simulator_module.Drone
        simulator_module.Drone = ControlledDrone
        try:
            simpy_env = simpy.Environment()
            sim = Simulator(seed=seed, env=simpy_env, n_drones=NUM_UAVS + 1)
        finally:
            simulator_module.Drone = original_drone_cls
        for idx, d in enumerate(sim.drones):
            if idx == GCS_IDX:
                d._coords = [float(snap.gcs_pos[0]), float(snap.gcs_pos[1]), float(snap.gcs_pos[2])]
            else:
                d._coords = [float(self.uav_pos[0, idx, 0]), float(self.uav_pos[0, idx, 1]), 50.0]

        # Apply the exact urban channel model to the real vendored PHY.
        if hasattr(sim, "channel") and sim.channel is not None:
            sim.channel.channel_model = UrbanScenarioChannelModel()
        # Avoid redundant unbounded event snapshots during headless evaluation.
        if hasattr(sim, "publish_state"):
            sim.publish_state = lambda *args, **kwargs: None
        self._sim = sim
        self._simpy_env = simpy_env
        return obs, state, mask

    def _mac_slotted_step_env(
        self, e: int, dest_act: np.ndarray, now_s: float, rng: np.random.Generator
    ) -> Tuple[float, int]:
        self._dest_act[:] = dest_act
        self._now_s = float(now_s)
        self._step_aoi = 0.0
        self._step_drops = 0
        self._step_tx_count = 0
        self.last_ack[e, :] = 0.0

        for u in range(NUM_UAVS):
            d = self._sim.drones[u]
            d._coords = [float(self.uav_pos[e, u, 0]), float(self.uav_pos[e, u, 1]), 50.0]
            d.sleep = bool(self.uav_alive[e, u] == 0)
            if getattr(d, "waiting_list", None):
                for pkt in d.waiting_list:
                    if d.transmitting_queue.qsize() < MAX_QUEUE:
                        d.transmitting_queue.put(pkt)
                d.waiting_list.clear()

            for qi in range(int(self.q_count[e, u])):
                cid = int(self.q_cell_id[e, u, qi])
                ts = float(self.q_timestamp[e, u, qi])
                key = (u, cid, round(ts, 2))
                if key not in self._known_q_keys and abs(ts - now_s) < 1e-3:
                    self._known_q_keys.add(key)
                    pid = self._next_pkt_id
                    self._next_pkt_id += 1
                    pkt = self._data_packet_cls(
                        src_drone=d,
                        dst_drone=self._sim.drones[GCS_IDX],
                        creation_time=self._simpy_env.now,
                        data_packet_id=pid,
                        data_packet_length=8656,
                        simulator=self._sim,
                        channel_id=0,
                    )
                    pkt.deadline = self._uav_cfg.PACKET_LIFETIME
                    pkt.transmission_mode = 0
                    self._pkt_meta[pid] = {
                        "cell_id": cid,
                        "ts_s": ts,
                        "hops": 0,
                        "visited_mask": 1 << u,
                    }
                    if d.transmitting_queue.qsize() < MAX_QUEUE:
                        d.transmitting_queue.put(pkt)

        self._simpy_env.run(until=int(round(now_s * 1_000_000)))

        # Mirror SimPy queues (including any in-flight packet at index 0!) back to host_env.q_*
        self.q_count[e, :] = 0
        for u in range(NUM_UAVS):
            d = self._sim.drones[u]
            items = []
            if getattr(d, "_in_flight_pkt", None) is not None:
                items.append(d._in_flight_pkt)
            items.extend(list(d.transmitting_queue.queue))
            items.extend(list(getattr(d, "waiting_list", [])))
            cnt = min(len(items), MAX_QUEUE)
            self.q_count[e, u] = cnt
            for qi in range(cnt):
                meta = self._pkt_meta.get(getattr(items[qi], "packet_id", -1))
                if meta is not None:
                    self.q_cell_id[e, u, qi] = int(meta["cell_id"])
                    self.q_timestamp[e, u, qi] = float(meta["ts_s"])
                    self.q_hops[e, u, qi] = int(meta["hops"])
                    self.q_visited_mask[e, u, qi] = int(meta["visited_mask"])
                    self.q_retries[e, u, qi] = int(items[qi].number_retransmission_attempt.get(u, 0))

        busy_fraction = min(1.0, self._step_tx_count * DATA_AIRTIME_US / STEP_DURATION_US)
        self.channel_busy_ewma[e] = 0.8 * self.channel_busy_ewma[e] + 0.2 * busy_fraction
        return float(self._step_aoi), int(self._step_drops)


def _eval_uavnetsim_worker(args: Tuple[Dict[str, Any], int, int, str, float, Tuple[float, ...]]) -> EpisodeKPI:
    """ProcessPool worker running one seed on real vendored UavNetSim using CPU actor weights."""
    policy_spec, seed, max_steps, belief_mode, aoi_eta, weights = args
    import torch
    from uav_network.experiments.train_slotted_vanilla import build_agent, make_deterministic_policy_fn

    torch.set_num_threads(1)
    agent = build_agent(policy_spec["cfg"], torch.device("cpu"))
    agent.actor.load_state_dict(policy_spec["actor_state_dict"])
    agent.actor.eval()
    policy_fn = make_deterministic_policy_fn(agent, torch.device("cpu"))

    env = UavNetSimCSMA2DUavEnv(
        max_steps=max_steps,
        belief_sharing_mode=belief_mode,
        aoi_decay_eta=aoi_eta,
        reward_weights=weights,
    )
    try:
        obs, _, mask = env.reset(seeds=[int(seed)])
        for _ in range(max_steps):
            act_cont, act_disc = policy_fn(obs, mask)
            obs, _, mask, _, dones, _ = env.step(act_cont, act_disc)
            if float(dones[0]) > 0.5:
                break
        return _extract_episode_kpi(env, 0)
    finally:
        env.close_sim()


def evaluate_on_full_csma_uavnetsim(
    policy_fn: PolicyFn,
    eval_seeds: List[int],
    max_steps: int = 550,
    belief_sharing_mode: str = "local_decpomdp",
    aoi_decay_eta: float = 0.02,
    reward_weights: Optional[Tuple[float, float, float, float, float]] = None,
    policy_spec: Optional[Dict[str, Any]] = None,
    max_workers: Optional[int] = None,
) -> List[EpisodeKPI]:
    """Evaluate deterministic policy on REAL vendored `third_party/UavNetSim` SimPy CSMA/CA.

    When `policy_spec={"cfg": cfg, "actor_state_dict": cpu_state_dict}` is provided and
    `len(eval_seeds) >= 4`, runs seeds in parallel across CPU cores via `ProcessPoolExecutor`.
    """
    w = reward_weights if reward_weights is not None else (0.25, 10.0, 0.03, 20.0, 1.0)
    n_workers = max_workers if max_workers is not None else min(os.cpu_count() or 1, 8)
    if policy_spec is not None and len(eval_seeds) >= 4 and n_workers > 1:
        tasks = [(policy_spec, int(s), max_steps, belief_sharing_mode, aoi_decay_eta, w) for s in eval_seeds]
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            return list(ex.map(_eval_uavnetsim_worker, tasks))

    results: List[EpisodeKPI] = []
    for s in eval_seeds:
        env = UavNetSimCSMA2DUavEnv(
            max_steps=max_steps,
            belief_sharing_mode=belief_sharing_mode,
            aoi_decay_eta=aoi_decay_eta,
            reward_weights=w,
        )
        try:
            obs, _, mask = env.reset(seeds=[int(s)])
            for _ in range(max_steps):
                act_cont, act_disc = policy_fn(obs, mask)
                obs, _, mask, _, dones, _ = env.step(act_cont, act_disc)
                if float(dones[0]) > 0.5:
                    break
            results.append(_extract_episode_kpi(env, 0))
        finally:
            env.close_sim()
    return results


# Backward-compatibility alias
FullCSMA2DUavEnv = ReferenceCSMA2DUavEnv
