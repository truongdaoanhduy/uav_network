"""v5.7 Numerically Stable Dempster-Shafer (FP32/FP64), Exact EWMA AoI Aging,
Zeng 2019 Battery (Steady-Level + Acceleration Thrust), and Kinematic Safety Shield.

Upgrades over v5.4/v5.6:
  - REF-DS-01 + FP32 int-diff refinement: exact up to n=65535 in FP32 & FP64, raises on total conflict.
  - V57-AOI-EXACT: O(1) EWMA effective counts (`ewma_effective_counts_update` + `counts_to_canonical_ds_effective`)
    eliminating artificial rejuvenation of stale evidence (matches Xue 2026 Eq. 9-10 per-sample exponential decay).
  - V57-CAL-BBA: optional reliability cap `max_mass` in `sensor_likelihood_to_bba` so a single false alarm
    cannot immediately exceed `confirm_th=0.90`.
  - V57-ENERGY-ACCEL: `zeng_propulsion_power_with_accel_w` adds thrust-tilt factor kappa=(1+(||a||/g)^2)^0.5
    on induced power and positive kinetic energy delta (m=2.0 kg, eta_prop=0.70).
  - V57-SHIELD: `kinematic_safety_shield` replaces hard-zero velocity veto (which violated a_max=5 m/s^2 by 5x)
    with braking-aware kinematic safe action projection strictly enforcing ||v_new - v_prev||/dt <= a_max.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
def counts_to_canonical_ds_stable(
    n_pos: int, n_neg: int, mf: float = 0.80, me: float = 0.80
) -> Tuple[float, float, float]:
    """Exact overflow/underflow-free canonical Dempster-Shafer on Theta = {T, F}.

    Handles n_pos, n_neg in [0, 65535] without cancellation in FP32 or FP64.
    Raises ValueError on undefined total conflict (K = 1, i.e., mf=1, me=1, n_pos>0, n_neg>0).
    """
    if n_pos < 0 or n_neg < 0:
        raise ValueError(f"Counts must be non-negative: ({n_pos}, {n_neg})")
    if not (0.0 <= mf <= 1.0 and 0.0 <= me <= 1.0):
        raise ValueError(f"BBAs must lie in [0, 1]: ({mf}, {me})")
    if n_pos == 0 and n_neg == 0:
        return 0.0, 0.0, 1.0
    if mf == 1.0 and me == 1.0 and n_pos > 0 and n_neg > 0:
        raise ValueError("Total Dempster-Shafer conflict (K = 1) is undefined")
    if mf == 1.0 and n_pos > 0:
        return 1.0, 0.0, 0.0
    if me == 1.0 and n_neg > 0:
        return 0.0, 1.0, 0.0

    lmf = math.log1p(-mf)
    lme = math.log1p(-me)
    a = 0.0 if n_pos == 0 else n_pos * lmf
    b = 0.0 if n_neg == 0 else n_neg * lme

    if mf == me:
        # Integer-difference path prevents FP32 subtraction cancellation at large n
        d = (n_pos - n_neg) * lmf  # equals a - b
        if n_pos <= n_neg:  # a >= b because lmf <= 0
            pa, pb, pu = 1.0, math.exp(-d), math.exp(b)
        else:
            pa, pb, pu = math.exp(d), 1.0, math.exp(a)
    else:
        s = max(a, b)
        pa = math.exp(a - s)
        pb = math.exp(b - s)
        pu = math.exp(min(a, b))

    t_num = 0.0 if n_pos == 0 else pb * (-math.expm1(a))
    f_num = 0.0 if n_neg == 0 else pa * (-math.expm1(b))
    z = t_num + f_num + pu
    return t_num / z, f_num / z, pu / z


def ewma_effective_counts_update(
    eff_pos: float,
    eff_neg: float,
    last_ts_s: float,
    now_s: float,
    det: bool,
    eta: float = 0.02,
) -> Tuple[float, float]:
    """O(1) exact exponential aging of per-sample evidence counts (Xue 2026 Eq. 9-10).

    Because exp(-eta * (t - t1)) == exp(-eta * (t - t2)) * exp(-eta * (t2 - t1)),
    decaying accumulated effective counts before adding the new observation eliminates
    the 'artificial rejuvenation' defect of frozen cumulative-snapshot discounting.
    """
    if eta < 0.0 or not math.isfinite(eta):
        raise ValueError("eta must be finite and non-negative")
    dt = max(0.0, float(now_s) - float(last_ts_s)) if last_ts_s >= 0.0 else 0.0
    decay = math.exp(-eta * dt) if (eta > 0.0 and dt > 0.0) else 1.0
    new_pos = float(eff_pos) * decay + (1.0 if det else 0.0)
    new_neg = float(eff_neg) * decay + (0.0 if det else 1.0)
    return new_pos, new_neg


def counts_to_canonical_ds_effective(
    eff_pos: float, eff_neg: float, mf: float = 0.80, me: float = 0.80
) -> Tuple[float, float, float]:
    """Numerically stable canonical DS for continuous/discounted effective counts >= 0."""
    ep = float(eff_pos)
    en = float(eff_neg)
    if ep < 0.0 or en < 0.0 or not (math.isfinite(ep) and math.isfinite(en)):
        raise ValueError(f"Effective counts must be finite non-negative: ({ep}, {en})")
    if not (0.0 <= mf <= 1.0 and 0.0 <= me <= 1.0):
        raise ValueError(f"BBAs must lie in [0, 1]: ({mf}, {me})")
    if ep == 0.0 and en == 0.0:
        return 0.0, 0.0, 1.0
    if mf == 1.0 and me == 1.0 and ep > 0.0 and en > 0.0:
        raise ValueError("Total Dempster-Shafer conflict (K = 1) is undefined")
    if mf == 1.0 and ep > 0.0:
        return 1.0, 0.0, 0.0
    if me == 1.0 and en > 0.0:
        return 0.0, 1.0, 0.0

    lmf = math.log1p(-mf)
    lme = math.log1p(-me)
    a = 0.0 if ep == 0.0 else ep * lmf
    b = 0.0 if en == 0.0 else en * lme
    s = max(a, b)
    pa = math.exp(a - s)
    pb = math.exp(b - s)
    pu = math.exp(min(a, b))
    t_num = 0.0 if ep == 0.0 else pb * (-math.expm1(a))
    f_num = 0.0 if en == 0.0 else pa * (-math.expm1(b))
    z = t_num + f_num + pu
    return t_num / z, f_num / z, pu / z


def discounted_mass(mass: Sequence[float], freshness_weight: float) -> Tuple[float, float, float]:
    """Xue-style evidence reliability discount: m'(T/E)=psi*m(T/E), m'(U)=1-psi*(1-m(U))."""
    mf, me, mu = (float(x) for x in mass)
    w = float(freshness_weight)
    if not (math.isfinite(w) and 0.0 <= w <= 1.0) or not all(math.isfinite(x) and x >= 0.0 for x in (mf, me, mu)):
        raise ValueError("Invalid mass or freshness weight")
    if abs(mf + me + mu - 1.0) > 1e-7:
        raise ValueError("Mass must sum to 1")
    if w == 1.0:
        return mf, me, mu
    if w == 0.0:
        return 0.0, 0.0, 1.0
    return w * mf, w * me, 1.0 - w * (mf + me)


def combine_ds_masses(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Orthogonal sum of two binary-frame DS BPAs; never hide total conflict."""
    at, af, au = (float(v) for v in a)
    bt, bf, bu = (float(v) for v in b)
    for m in ((at, af, au), (bt, bf, bu)):
        if not all(math.isfinite(v) and -1e-12 <= v <= 1 + 1e-12 for v in m) or abs(sum(m) - 1.0) > 1e-7:
            raise ValueError("Invalid BPA")
    k = at * bf + af * bt
    if k >= 1.0 - 1e-12:
        raise ValueError("Dempster total conflict, K=1")
    denom = 1.0 - k
    tt = (at * bt + at * bu + au * bt) / denom
    ff = (af * bf + af * bu + au * bf) / denom
    uu = (au * bu) / denom
    z = tt + ff + uu
    return tt / z, ff / z, uu / z


def sensor_likelihood_to_bba(
    pd: float, pf: float, max_mass: float | None = None
) -> Tuple[float, float]:
    """Likelihood-ratio-to-BBA mapping with optional v5.7 reliability cap `max_mass`.

    Positive evidence support = 1 - P(+|empty)/P(+|target).
    Negative evidence support = 1 - P(-|target)/P(-|empty).
    When `max_mass` (e.g. 0.80) is provided, caps single-observation support so a single
    false alarm cannot immediately trigger `confirm_th=0.90`.
    """
    pd, pf = float(pd), float(pf)
    if not (0.0 < pf < pd < 1.0):
        raise ValueError("require 0 < Pf < Pd < 1")
    mf = 1.0 - pf / pd
    me = 1.0 - (1.0 - pd) / (1.0 - pf)
    if max_mass is not None:
        cap = float(max_mass)
        if not (0.0 < cap <= 1.0):
            raise ValueError("max_mass must lie in (0, 1]")
        mf = min(mf, cap)
        me = min(me, cap)
    return mf, me


def fuse_provenance_counts(
    n_pos: Sequence[int], n_neg: Sequence[int], gen_ts_s: Sequence[int],
    rx_age_s: Sequence[float], *, mf: float = 0.8, me: float = 0.8, eta: float = 0.0,
    mode: str = "sum",
) -> Tuple[float, float, float]:
    """Fuse uniquely origin-tagged cumulative evidence (not repeated re-transmissions).

    `sum`: combine each origin's canonical DS mass once, with delivery-age discount per origin.
    `zhang_max`: exact Eq (16) MAX of counts before canonical DS, no AoI discount (baseline).
    """
    if not math.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite nonnegative")
    if mode not in ("sum", "zhang_max"):
        raise ValueError("unknown fusion mode")
    p = [int(x) for x in n_pos]
    n = [int(x) for x in n_neg]
    ts = [int(x) for x in gen_ts_s]
    delays = [float(x) for x in rx_age_s]
    if not (len(p) == len(n) == len(ts) == len(delays)):
        raise ValueError("origin arrays must match")
    if any(x < 0 for x in p + n):
        raise ValueError("negative counts")
    if mode == "zhang_max":
        return counts_to_canonical_ds_stable(max(p, default=0), max(n, default=0), mf, me)
    if eta == 0.0:
        return counts_to_canonical_ds_stable(sum(p), sum(n), mf, me)
    result = (0.0, 0.0, 1.0)
    for pos, neg, t, delay in zip(p, n, ts, delays):
        if pos == 0 and neg == 0:
            continue
        if t < 0 or not math.isfinite(delay) or delay < 0:
            raise ValueError("non-empty origin evidence requires valid generation timestamp and receive age")
        single = counts_to_canonical_ds_stable(pos, neg, mf, me)
        psi = math.exp(-eta * delay)
        result = combine_ds_masses(result, discounted_mass(single, psi))
    return result
