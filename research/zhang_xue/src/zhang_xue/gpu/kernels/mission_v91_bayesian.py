"""Independent-cell Bayesian target belief from origin-indexed sensor counts.

IMPORTANT: Exact *per-cell* conditional log-odds update under the assumed
Bernoulli Pd/Pf observation model and fixed marginal prior, not an exact joint
posterior over 15-without-replacement targets. Unknown fraction is a separately
defined epistemic coverage score, NOT a DS mass; the three channels need not
sum to one in Bayesian mode.
"""
from __future__ import annotations

from zhang_xue.gpu.state_v67 import HAS_WARP, wp

if HAS_WARP:
    @wp.func
    def _bayes_from_counts_v91(
        p: float, n: float, log_prior: float, llr_pos: float, llr_neg: float,
    ) -> wp.vec3:
        logit = wp.clamp(log_prior + p * llr_pos + n * llr_neg, -40.0, 40.0)
        posterior = 1.0 / (1.0 + wp.exp(-logit))
        unknown = wp.exp(-0.6931471805599453 * (p + n))
        return wp.vec3(posterior, 1.0 - posterior, unknown)
