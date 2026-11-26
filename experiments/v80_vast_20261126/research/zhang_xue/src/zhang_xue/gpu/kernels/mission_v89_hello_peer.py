"""V89 packet-sourced neighbor telemetry, opt-in. No peer truth enters the actor.

The sender's 4 float32 fields are snapshotted as an active HELLO frame
at the actual MAC TX start; only successful PHY reception populates the
receiver-local cache. The observable peer age uses the original TX timestamp.
"""
from __future__ import annotations

from zhang_xue.gpu.state_v67 import HAS_WARP, NetworkBuffersV67, ROUTE_TTL_US, wp

if HAS_WARP:
    @wp.kernel
    def capture_hello_offer_v89(
        state: NetworkBuffersV67,
        positions: wp.array3d(dtype=wp.float32),
        velocities: wp.array3d(dtype=wp.float32),
    ):
        e, tx = wp.tid()
        state.hello_offer_pose[e, tx, 0] = positions[e, tx, 0]
        state.hello_offer_pose[e, tx, 1] = positions[e, tx, 1]
        if tx < 6:
            state.hello_offer_pose[e, tx, 2] = velocities[e, tx, 0]
            state.hello_offer_pose[e, tx, 3] = velocities[e, tx, 1]
        else:
            state.hello_offer_pose[e, tx, 2] = 0.0
            state.hello_offer_pose[e, tx, 3] = 0.0

    @wp.kernel
    def observe_hello_peers_v89(
        state: NetworkBuffersV67,
        positions: wp.array3d(dtype=wp.float32),
        velocities: wp.array3d(dtype=wp.float32),
        obs: wp.array3d(dtype=wp.float32),
        v_max: float,
    ):
        e, u = wp.tid()
        x = positions[e, u, 0]
        y = positions[e, u, 1]
        vx = velocities[e, u, 0]
        vy = velocities[e, u, 1]
        now = state.now[e]
        used = int(0)
        for rank in range(5):
            best = int(-1)
            best_d2 = float(1.0e30)
            for peer in range(6):
                if peer == u or (used & (1 << peer)) != 0:
                    continue
                sent_at = int(state.hello_rx_at_us[e, u, peer])
                if sent_at < 0 or now - sent_at >= ROUTE_TTL_US:
                    continue
                dx = state.hello_rx_pose[e, u, peer, 0] - x
                dy = state.hello_rx_pose[e, u, peer, 1] - y
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best = peer
                    best_d2 = d2
            offset = 514 + rank * 7
            for k in range(7):
                obs[e, u, offset + k] = 0.0
            if best >= 0:
                used = used | (1 << best)
                obs[e, u, offset + 0] = (state.hello_rx_pose[e, u, best, 0] - x) / 1000.0
                obs[e, u, offset + 1] = (state.hello_rx_pose[e, u, best, 1] - y) / 1000.0
                obs[e, u, offset + 2] = (state.hello_rx_pose[e, u, best, 2] - vx) / v_max
                obs[e, u, offset + 3] = (state.hello_rx_pose[e, u, best, 3] - vy) / v_max
                obs[e, u, offset + 4] = 1.0
                age = wp.max(0, now - int(state.hello_rx_at_us[e, u, best]))
                obs[e, u, offset + 5] = wp.clamp(float(age) / float(ROUTE_TTL_US), 0.0, 1.0)
                obs[e, u, offset + 6] = wp.sqrt(best_d2) / 1000.0
