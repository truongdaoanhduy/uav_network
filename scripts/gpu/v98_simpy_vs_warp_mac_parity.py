#!/usr/bin/env python3
"""V100 Gate S12: Bug-Free Clean-Room Continuous-Time DES vs True-Saturation Warp GPU Parity.

Fixes BOTH root causes uncovered in V98/V99 forensic audit:
1. Replaces V98's blind 50us DIFS-polling DES (which knocked 84.3% of N=6 nodes off the
   20us slot grid and collapsed p_des to 4.76%) with an exact 7-state continuous-time DES
   (`ST_IDLE..ST_TX_ACK`) matching Warp `events_v67.py` + `mac_v67.py` semantics.
2. Fixes V99's `_refill_saturated` mid-flight `q_pid` overwrite bug (which mismatched
   `ack_pid` vs `tx_pkt_id` across 10ms chunk boundaries and pulled N=6 Warp down from
   20.49% to 16.84%) by keeping `q_pid` invariant and only refilling unused slots.
3. Explicitly embeds `UPSTREAM_UAVNETSIM_DELTA` against pinned UavNetSim-v1 commit
   `c2703f24eb58cdb557a5f67ded5c7b0368a964ed` (Zhou et al., arXiv:2507.09852).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src"), str(ROOT / "scripts/gpu")]

from v94_mac_analytic_validation import solve_finite_retry_bianchi

ST_IDLE, ST_DIFS, ST_BACKOFF, ST_TX, ST_WAIT_ACK, ST_SIFS, ST_TX_ACK = range(7)
NEVER_US = 2_000_000_000
DATA_US = 312

UPSTREAM_UAVNETSIM_DELTA = {
    "pinned_commit": "c2703f24eb58cdb557a5f67ded5c7b0368a964ed",
    "cw_upper_bound": "Warp default [0,CW] (cw_inclusive_upper=1) vs upstream [0,CW-1] (set cw_inclusive_upper=0)",
    "data_frame_us": "Warp 280+32*entries (312us) vs upstream 8656b/2Mbps (4328us)",
    "idle_and_listen": "Warp exact event-driven + ready_mask vs upstream 20us wait_idle poll + 1us listen() lag",
    "ack_channel_hold": "Warp physical SIFS(10us) gap + RX ST_TX_ACK vs upstream TX holds channel_states for SIFS+ACK",
    "retry_drop_side_effect": "Warp purges all routes via bad_nh at ret>=5 vs upstream penalize(pkd) per ACK timeout",
}


def run_cleanroom_continuous_des(
    n_nodes: int,
    *,
    cw_inclusive_upper: int = 1,
    duration_us: int = 3_000_000,
    seed: int = 44,
    hidden: bool = False,
    data_us: int = DATA_US,
    difs_us: int = 50,
    sifs_us: int = 10,
    slot_us: int = 20,
    ack_us: int = 120,
    ack_to_us: int = 180,
    cw_min: int = 31,
    max_retx: int = 5,
    inval_route_on_drop: bool = False,
    chunk_refill_us: int = 0,
) -> dict:
    """Clean-room continuous-time DES matching Warp events_v67.py + mac_v67.py exact semantics."""
    rng = random.Random(seed)
    n_radio, gcs = n_nodes + 1, n_nodes
    audible = [[(i != j) for j in range(n_radio)] for i in range(n_radio)]
    if hidden and n_nodes == 2:
        audible[0][1] = audible[1][0] = False

    state = [ST_IDLE] * n_radio
    deadline = [NEVER_US] * n_radio
    bo_rem = [0] * n_radio
    frozen = [False] * n_radio
    retries = [0] * n_radio
    has_route = [r < n_nodes for r in range(n_radio)]
    ack_dst, ack_pid = [-1] * n_radio, [-1] * n_radio
    tx_pid, next_hop = [-1] * n_radio, [-1] * n_radio
    rx_lock, rx_corrupt, rx_interf = [-1] * n_radio, [False] * n_radio, [0] * n_radio

    now = next_pid = acks = retx = drops = 0
    next_refill = chunk_refill_us if chunk_refill_us > 0 else NEVER_US

    def is_busy(r: int) -> bool:
        if state[r] in (ST_TX, ST_TX_ACK):
            return True
        return any(o != r and audible[o][r] and state[o] in (ST_TX, ST_TX_ACK) for o in range(n_radio))

    def sum_interf(rx: int, excl: int) -> int:
        return sum(1 for o in range(n_radio) if o != excl and o != rx and audible[o][rx] and state[o] in (ST_TX, ST_TX_ACK))

    def rand_slot(r: int) -> int:
        cw = min(1023, ((cw_min + 1) << min(retries[r], max_retx)) - 1)
        return rng.randrange((cw + 1) if cw_inclusive_upper else max(1, cw))

    def arm_idle(r: int, t: int) -> None:
        if r != gcs and state[r] == ST_IDLE and has_route[r] and not is_busy(r):
            state[r], deadline[r] = ST_DIFS, t + difs_us

    def finish_tx(tx: int) -> None:
        nonlocal acks
        st = state[tx]
        state[tx], deadline[tx] = ST_IDLE, NEVER_US
        nh = next_hop[tx]
        if st == ST_TX_ACK:
            for rx in range(n_radio):
                if rx_lock[rx] != tx:
                    continue
                ok = (not rx_corrupt[rx]) and (rx_interf[rx] == 0) and audible[tx][rx]
                rx_lock[rx], rx_corrupt[rx] = -1, False
                if rx == nh and ok and state[rx] == ST_WAIT_ACK and tx_pid[rx] == ack_pid[tx]:
                    state[nh], deadline[nh], retries[nh] = ST_IDLE, NEVER_US, 0
                    acks += 1
            return
        for rx in range(n_radio):
            if rx == tx or rx_lock[rx] != tx:
                continue
            ok = (not rx_corrupt[rx]) and (rx_interf[rx] == 0) and audible[tx][rx]
            rx_lock[rx], rx_corrupt[rx] = -1, False
            if not ok or rx != nh:
                continue
            if state[rx] in (ST_IDLE, ST_DIFS, ST_BACKOFF):
                if state[rx] == ST_BACKOFF:
                    bo_rem[rx], frozen[rx] = max(0, deadline[rx] - now), True
                ack_dst[rx], ack_pid[rx] = tx, tx_pid[tx]
                state[rx], deadline[rx] = ST_SIFS, now + sifs_us
        state[tx], deadline[tx] = ST_WAIT_ACK, now + ack_to_us

    def start_tx(tx: int, dur: int, nh: int, is_ack: bool, pid: int, ready: set[int]) -> None:
        if rx_lock[tx] >= 0:
            rx_corrupt[tx] = True
        state[tx], deadline[tx], next_hop[tx], tx_pid[tx] = (ST_TX_ACK if is_ack else ST_TX), now + dur, nh, pid
        for rx in range(n_radio):
            if rx == tx:
                continue
            if (rx not in ready) and is_busy(rx):
                if state[rx] == ST_DIFS:
                    state[rx], deadline[rx] = ST_IDLE, NEVER_US
                elif state[rx] == ST_BACKOFF:
                    bo_rem[rx], frozen[rx] = max(0, deadline[rx] - now), True
                    state[rx], deadline[rx] = ST_IDLE, NEVER_US
            if state[rx] in (ST_TX, ST_TX_ACK):
                continue
            if rx_lock[rx] < 0 and audible[tx][rx]:
                rx_lock[rx], rx_corrupt[rx], rx_interf[rx] = tx, False, sum_interf(rx, tx)
            elif rx_lock[rx] >= 0:
                rx_interf[rx] = max(rx_interf[rx], sum_interf(rx, rx_lock[rx]))

    for r in range(n_radio):
        arm_idle(r, 0)

    while True:
        t_min = min(duration_us + 1, next_refill, min(deadline))
        if t_min > duration_us:
            break
        now = t_min
        if now == next_refill:
            for r in range(n_nodes):
                has_route[r] = True
                arm_idle(r, now)
            next_refill += chunk_refill_us
        for r in range(n_radio):
            if deadline[r] == now and state[r] in (ST_TX, ST_TX_ACK):
                finish_tx(r)
        for r in range(n_radio):
            if state[r] == ST_DIFS and deadline[r] == now:
                if is_busy(r):
                    state[r], deadline[r] = ST_IDLE, NEVER_US
                else:
                    rem = bo_rem[r] if frozen[r] else (rand_slot(r) * slot_us)
                    frozen[r], state[r], deadline[r] = False, ST_BACKOFF, now + rem
        ready = {
            r for r in range(n_radio)
            if deadline[r] == now and (state[r] == ST_SIFS or (state[r] == ST_BACKOFF and has_route[r] and not is_busy(r)))
        }
        for r in range(n_radio):
            st = state[r]
            if st == ST_BACKOFF and deadline[r] == now:
                if r not in ready:
                    state[r], deadline[r] = ST_IDLE, NEVER_US
                else:
                    next_pid += 1
                    start_tx(r, data_us, gcs, False, next_pid, ready)
            elif st == ST_SIFS and deadline[r] == now:
                start_tx(r, ack_us, ack_dst[r], True, ack_pid[r], ready)
            elif st == ST_WAIT_ACK and deadline[r] == now:
                retx += 1
                retries[r] += 1
                if retries[r] >= max_retx:
                    drops += 1
                    retries[r] = 0
                    if inval_route_on_drop:
                        has_route[r] = False
                state[r], deadline[r] = ST_IDLE, NEVER_US
            arm_idle(r, now)

    att = acks + retx
    p_des = retx / max(1, att)
    return {
        "n_nodes": n_nodes,
        "cw_inclusive_upper": cw_inclusive_upper,
        "hidden": hidden,
        "p_des": float(p_des),
        "se_p": float(math.sqrt(p_des * (1.0 - p_des) / max(1, att))),
        "throughput_des": float((acks * float(data_us)) / max(1.0, float(duration_us))),
        "attempts": att,
        "acks": acks,
        "retries": retx,
        "drops": drops,
    }


def run_warp_true_saturation(
    n_uavs: int,
    *,
    cw_inclusive_upper: int = 1,
    num_envs: int = 32,
    measure_chunks: int = 40,
    chunk_us: int = 10_000,
    device: str = "cuda:0",
    seed: int = 20261127,
    hidden: bool = False,
) -> dict[str, float]:
    """Run WarpCompressedRadioNetworkV67 with true 64-frame PKT_DATA saturation & invariant q_pid."""
    import warp as wp
    from zhang_xue.gpu.runtime_v67 import WarpCompressedRadioNetworkV67
    from zhang_xue.gpu.state_v67 import NEVER_US as W_NEVER, PKT_DATA

    net = WarpCompressedRadioNetworkV67(
        num_envs=num_envs,
        num_uavs=n_uavs,
        grid_w=2,
        enable_data_mule=False,
        enable_peer=False,
        enable_peer_telemetry=False,
        q_cap=64,
        cw_inclusive_upper=cw_inclusive_upper,
        device=device,
        seed=seed,
    )
    gcs = n_uavs

    def _put(name: str, arr: np.ndarray) -> None:
        a = net.data[name]
        wp.copy(a, wp.array(np.asarray(arr), dtype=a.dtype, device=device))

    gain = np.full((num_envs, n_uavs + 1, n_uavs + 1), 1.0e-4, dtype=np.float32)
    gain[:, np.arange(n_uavs + 1), np.arange(n_uavs + 1)] = 0.0
    if hidden and n_uavs == 2:
        gain[:, 0, 1] = 0.0
        gain[:, 1, 0] = 0.0
    net.set_channel_gain(gain)
    _put("hello_due", np.full((num_envs, n_uavs + 1), W_NEVER, dtype=np.int32))
    _put("hello_pending", np.zeros((num_envs, n_uavs + 1), dtype=np.uint8))

    rh = np.full((num_envs, n_uavs + 1, n_uavs + 1), -1, dtype=np.int8)
    rd = np.full((num_envs, n_uavs + 1, n_uavs + 1), 127, dtype=np.int8)
    re = np.zeros((num_envs, n_uavs + 1, n_uavs + 1), dtype=np.int32)
    rh[:, :n_uavs, gcs] = gcs
    rd[:, :n_uavs, gcs] = 1
    re[:, :n_uavs, gcs] = W_NEVER

    # Initialize PKT_DATA (0) and invariant slot PIDs ONCE so in-flight active_slot PIDs are NEVER overwritten!
    q_type_0 = np.full((num_envs, n_uavs + 1, 64), PKT_DATA, dtype=np.uint8)
    q_src_0 = np.broadcast_to(np.arange(n_uavs + 1, dtype=np.int8)[None, :, None], (num_envs, n_uavs + 1, 64)).copy()
    q_dst_0 = np.full((num_envs, n_uavs + 1, 64), gcs, dtype=np.int8)
    q_exp_0 = np.full((num_envs, n_uavs + 1, 64), W_NEVER, dtype=np.int32)
    q_ent_0 = np.ones((num_envs, n_uavs + 1, 64), dtype=np.uint8)
    q_pid_0 = (np.arange(64, dtype=np.int32)[None, None, :] + 1000 * np.arange(n_uavs + 1, dtype=np.int32)[None, :, None] + 1)
    q_pid_0 = np.broadcast_to(q_pid_0, (num_envs, n_uavs + 1, 64)).copy()
    _put("q_type", q_type_0)
    _put("q_src", q_src_0)
    _put("q_dst", q_dst_0)
    _put("q_expiry", q_exp_0)
    _put("q_entries", q_ent_0)
    _put("q_pid", q_pid_0)

    def _refill_unused_only() -> None:
        _put("route_hop", rh)
        _put("route_dist", rd)
        _put("route_expiry", re)
        # Clear GCS reverse E2E-ACK queue so GCS never overflows stats_drop;
        # for UAVs, append ONLY unused slots (q_used==0) without touching active_slot!
        qu = net.data["q_used"].numpy()
        qo = net.data["q_order"].numpy()
        qc = net.data["q_count"].numpy()
        qu[:, gcs, :] = 0
        qc[:, gcs] = 0
        for e in range(num_envs):
            for u in range(n_uavs):
                cnt = int(qc[e, u])
                if cnt < 48:
                    for slot in range(64):
                        if qu[e, u, slot] == 0:
                            qu[e, u, slot] = 1
                            qo[e, u, cnt] = slot
                            cnt += 1
                            if cnt >= 64:
                                break
                    qc[e, u] = cnt
        _put("q_used", qu)
        _put("q_order", qo)
        _put("q_count", qc)

    for _ in range(2):
        _refill_unused_only()
        net.step(duration_us=chunk_us, max_events=8192)

    ack0 = net.data["stats_mac_ack"].numpy().astype(np.int64).copy()
    ret0 = net.data["stats_retry"].numpy().astype(np.int64).copy()
    drp0 = net.data["stats_drop"].numpy().astype(np.int64).copy()
    now0 = net.data["now"].numpy().astype(np.int64).copy()

    for _ in range(measure_chunks):
        _refill_unused_only()
        net.step(duration_us=chunk_us, max_events=8192)

    ack1 = net.data["stats_mac_ack"].numpy().astype(np.int64)
    ret1 = net.data["stats_retry"].numpy().astype(np.int64)
    drp1 = net.data["stats_drop"].numpy().astype(np.int64)
    now1 = net.data["now"].numpy().astype(np.int64)

    d_ack = int(np.sum(ack1 - ack0))
    d_ret = int(np.sum(ret1 - ret0))
    d_drp = int(np.sum(drp1 - drp0))
    d_us = float(np.sum(now1 - now0))
    att = d_ack + d_ret
    p_w = d_ret / max(1, att)
    s_w = (d_ack * float(DATA_US)) / max(1.0, d_us)
    return {
        "n_uavs": n_uavs,
        "cw_inclusive_upper": cw_inclusive_upper,
        "hidden": hidden,
        "p_warp": float(p_w),
        "se_p": float(math.sqrt(p_w * (1.0 - p_w) / max(1, att))),
        "throughput_warp": float(s_w),
        "attempts": att,
        "acks": d_ack,
        "retries": d_ret,
        "drops": d_drp,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", default="2,4,6")
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--des-only", action="store_true")
    ap.add_argument("--des-us", "--des-duration-us", dest="des_duration_us", type=int, default=24_000_000,
                    help="DES observation window in us; alias for --des-duration-us")
    ap.add_argument("--hidden-des-duration-us", type=int, default=None,
                    help="Separate hidden-terminal window; defaults to --des-us")
    ap.add_argument("--output", "--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.des_duration_us <= 0 or (args.hidden_des_duration_us is not None and args.hidden_des_duration_us <= 0):
        ap.error("DES windows must be positive microseconds")
    if args.hidden_des_duration_us is None:
        args.hidden_des_duration_us = args.des_duration_us

    rows = []
    all_ok = True
    for n in [int(x) for x in args.nodes.split(",") if x.strip()]:
        an1 = solve_finite_retry_bianchi(n, inclusive=True, total_attempts=5)
        an0 = solve_finite_retry_bianchi(n, inclusive=False, total_attempts=5)
        des1 = run_cleanroom_continuous_des(n, cw_inclusive_upper=1, duration_us=args.des_duration_us, seed=44 + n)
        des0 = run_cleanroom_continuous_des(n, cw_inclusive_upper=0, duration_us=args.des_duration_us, seed=144 + n)
        row = {
            "n_nodes": n,
            "bianchi_att5_ieee": an1,
            "bianchi_att5_uavnetsim": an0,
            "cleanroom_des_ieee": des1,
            "cleanroom_des_uavnetsim": des0,
        }
        if not args.des_only:
            w1 = run_warp_true_saturation(n, cw_inclusive_upper=1, num_envs=args.envs, measure_chunks=args.chunks, device=args.device)
            w0 = run_warp_true_saturation(n, cw_inclusive_upper=0, num_envs=args.envs, measure_chunks=args.chunks, device=args.device)
            rel_p1 = abs(w1["p_warp"] - des1["p_des"]) / max(1e-9, des1["p_des"])
            rel_s1 = abs(w1["throughput_warp"] - des1["throughput_des"]) / max(1e-9, des1["throughput_des"])
            rel_p0 = abs(w0["p_warp"] - des0["p_des"]) / max(1e-9, des0["p_des"])
            rel_s0 = abs(w0["throughput_warp"] - des0["throughput_des"]) / max(1e-9, des0["throughput_des"])
            row.update({
                "warp_ieee": w1,
                "warp_uavnetsim": w0,
                "warp_vs_des_rel_err": {"p_ieee": rel_p1, "s_ieee": rel_s1, "p_uavnetsim": rel_p0, "s_uavnetsim": rel_s0},
                "directional_cw_ok": bool(w0["p_warp"] > w1["p_warp"]),
            })
            if not (w1["attempts"] > 5000 and w0["attempts"] > 5000 and rel_p1 <= 0.05 and rel_s1 <= 0.03 and rel_p0 <= 0.05 and rel_s0 <= 0.03 and row["directional_cw_ok"]):
                all_ok = False
        rows.append(row)

    des_ht = run_cleanroom_continuous_des(2, cw_inclusive_upper=1, duration_us=args.hidden_des_duration_us, seed=99, hidden=True, inval_route_on_drop=True, chunk_refill_us=10_000)
    ht_block = {"cleanroom_des_hidden": des_ht}
    if not args.des_only:
        w_ht = run_warp_true_saturation(2, cw_inclusive_upper=1, num_envs=args.envs, measure_chunks=20, device=args.device, hidden=True)
        ht_block["warp_hidden"] = w_ht
        ht_block["hidden_rel_err_p"] = abs(w_ht["p_warp"] - des_ht["p_des"]) / max(1e-9, des_ht["p_des"])
        ht_block["hidden_degrades_p"] = bool(w_ht["p_warp"] > rows[0]["warp_ieee"]["p_warp"])
        if not (ht_block["hidden_degrades_p"] and ht_block["hidden_rel_err_p"] <= 0.05):
            all_ok = False

    out = {
        "status": ("DES_ONLY_REFERENCE" if args.des_only else ("PASS" if all_ok else "FAIL")),
        "upstream_packet_level_simpy_parity": "NOT_TESTED",
        "scenario_frozen": False,
        "des_duration_us": args.des_duration_us,
        "hidden_des_duration_us": args.hidden_des_duration_us,
        "mac_pass_scope": "self-written DES research surrogate only; NOT upstream UavNetSim-v1 SimPy",
        "mode": "des_only" if args.des_only else f"warp_{args.device}_vs_cleanroom_des_and_bianchi",
        "scope": "Bug-free 7-state continuous-time DES vs true-saturation Warp GPU (invariant q_pid, PKT_DATA=0) + documented upstream UavNetSim-v1 delta",
        "upstream_uavnetsim_delta": UPSTREAM_UAVNETSIM_DELTA,
        "results": rows,
        "hidden_terminal": ht_block,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
