"""GPU/CPU Warp projection for M2 correlated target DS (F/E/U) masses.

INPUT: *unique, conditionally correlated* evidence-event LLR positive/negative
support totals. Event provenance and ordering, SPRT and UavNetSim network are
computed in the CPU Native reference. This is only the numerically stable
parallel 2D map mass projection; DO NOT treat it as a full Warp M2 rollout.

Each thread owns one (environment, node, cell). For 1000m/25m cells there are
1600 cells; 7 nodes consist of 6 UAV local maps and the GCS map. Float32
requires only 20 bytes / cell across two input and three output arrays.
"""
from __future__ import annotations
import warp as wp


@wp.func
def _log_expm1_positive(x: wp.float32) -> wp.float32:
    # The DS closed form is softmax(log(expm1(A)),log(expm1(B)),0).
    # Direct expm1 overflows for long evidence and exp(x)-1 cancels for tiny x.
    if x <= wp.float32(0.0):
        return wp.float32(-1.0e30)
    if x < wp.float32(0.001):
        return wp.log(x) + wp.log(wp.float32(1.0) + x * wp.float32(0.5)
                                    + x * x * wp.float32(0.1666666667))
    if x > wp.float32(18.0):
        return x
    return wp.log(wp.exp(x) - wp.float32(1.0))


@wp.kernel
def m2_project_mass(
    positive_support: wp.array3d(dtype=wp.float32),
    negative_support: wp.array3d(dtype=wp.float32),
    m_f: wp.array3d(dtype=wp.float32),
    m_e: wp.array3d(dtype=wp.float32),
    m_u: wp.array3d(dtype=wp.float32),
):
    world,node,cell=wp.tid()
    a=positive_support[world,node,cell]
    b=negative_support[world,node,cell]
    lf=_log_expm1_positive(a)
    le=_log_expm1_positive(b)
    biggest=wp.max(wp.float32(0.0),wp.max(lf,le))
    sf=wp.exp(lf-biggest)
    se=wp.exp(le-biggest)
    su=wp.exp(-biggest)
    norm=sf+se+su
    m_f[world,node,cell]=sf/norm
    m_e[world,node,cell]=se/norm
    m_u[world,node,cell]=su/norm
