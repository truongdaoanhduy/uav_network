"""Opt-in NVIDIA Warp Dempster–Shafer evidence update kernels.

Each thread handles one (world, UAV, ground-plane cell). `observations` only
contains noisy sensor events: -1 not seen, 0 negative, 1 positive.
No target IDs/true target coordinates are visible to these functions.

These are tested building blocks, not yet wired into the production simulator:
verification, candidate report IDs and packet-aware map fusion need a separate
explicit design. Existing Bayesian training is unaffected.
"""
import warp as wp


@wp.kernel
def ds_reset(m_f: wp.array3d(dtype=wp.float32),
             m_e: wp.array3d(dtype=wp.float32),
             m_u: wp.array3d(dtype=wp.float32)):
    world, agent, cell = wp.tid()
    m_f[world, agent, cell] = wp.float32(0.0)
    m_e[world, agent, cell] = wp.float32(0.0)
    m_u[world, agent, cell] = wp.float32(1.0)


@wp.kernel
def ds_update_observations(
    m_f: wp.array3d(dtype=wp.float32),
    m_e: wp.array3d(dtype=wp.float32),
    m_u: wp.array3d(dtype=wp.float32),
    observations: wp.array3d(dtype=wp.int32),
    positive_confidences: wp.array3d(dtype=wp.float32),
    negative_confidences: wp.array3d(dtype=wp.float32),
):
    world, agent, cell = wp.tid()
    status = observations[world, agent, cell]
    if status != 0 and status != 1:
        return
    old_f = m_f[world, agent, cell]
    old_e = m_e[world, agent, cell]
    old_u = m_u[world, agent, cell]
    evidence_f = wp.float32(0.0)
    evidence_e = wp.float32(0.0)
    if status == 1:
        evidence_f = positive_confidences[world, agent, cell]
    else:
        evidence_e = negative_confidences[world, agent, cell]
    if evidence_f < wp.float32(0.0) or evidence_e < wp.float32(0.0):
        return
    if evidence_f >= wp.float32(1.0) or evidence_e >= wp.float32(1.0):
        return
    evidence_u = wp.float32(1.0) - evidence_f - evidence_e
    denominator = wp.float32(1.0) - old_e * evidence_f - old_f * evidence_e
    if denominator <= wp.float32(1e-8):
        return
    new_f = (old_f * evidence_f + old_f * evidence_u + old_u * evidence_f) / denominator
    new_e = (old_e * evidence_e + old_e * evidence_u + old_u * evidence_e) / denominator
    new_u = old_u * evidence_u / denominator
    m_f[world, agent, cell] = wp.clamp(new_f, wp.float32(0.0), wp.float32(1.0))
    m_e[world, agent, cell] = wp.clamp(new_e, wp.float32(0.0), wp.float32(1.0))
    m_u[world, agent, cell] = wp.clamp(new_u, wp.float32(0.0), wp.float32(1.0))
