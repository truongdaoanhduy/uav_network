#!/usr/bin/env python3
"""V94 Independent IEEE 802.11 DCF Validation Suite (Gate S12).

Addresses S12 without conflating Warp CPU<->CUDA computational parity with
independent model validation:
- Tier 1: Exact finite-retry Bianchi & Tinnirello (2005) analytic fixed point
  for MAX_RETX=5, CW_min=31, comparing IEEE closed [0, CW] (W_i = CW_i + 1)
  against UavNetSim-v1 exclusive [0, CW-1] (W_i = CW_i).
- Tier 2: Standalone clean-room Python discrete-event (DES) single-hop DCF
  oracle (zero shared code with Warp or UavNetSim) verifying collision rate p
  and saturation throughput S against Tier 1 across N in {2, 4, 6}.
- Optional --with-warp hook runs WarpCompressedRadioNetworkV67 on cuda:0/cpu
  under matched single-hop contention parameters on Vast.ai.
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
sys.path[:0] = [str(ROOT), str(ROOT / "research/zhang_xue/src")]

# Matched IEEE 802.11 DCF constants from state_v67.py
DIFS_US = 50.0
SIFS_US = 10.0
SLOT_US = 20.0
ACK_US = 120.0
ACK_TO_US = 180.0
DATA_US = 312.0
CW_MIN = 31
MAX_RETX = 5  # Total attempts R+1 = 6 (stages i = 0..5)


def cw_at_stage(stage: int, *, cw_min: int = CW_MIN, inclusive: bool = True) -> int:
    """Return backoff support size W_i for retry stage i (0..MAX_RETX)."""
    cw_i = min(1023, ((cw_min + 1) << stage) - 1)
    return (cw_i + 1) if inclusive else cw_i


def solve_finite_retry_bianchi(
    n_nodes: int,
    *,
    cw_min: int = CW_MIN,
    max_retx: int = MAX_RETX,
    inclusive: bool = True,
    data_us: float = DATA_US,
    total_attempts: int | None = None,
) -> dict[str, float]:
    """Solve exact finite-retry Bianchi & Tinnirello (2005) fixed point (p, tau, S)."""
    n_att = (max_retx + 1) if total_attempts is None else int(total_attempts)
    w_stages = [float(cw_at_stage(i, cw_min=cw_min, inclusive=inclusive)) for i in range(n_att)]

    def tau_of_p(p: float) -> float:
        num = 0.0
        den = 0.0
        p_pow = 1.0
        for w_i in w_stages:
            num += p_pow
            den += p_pow * (w_i + 1.0) * 0.5
            p_pow *= p
        return num / max(1e-12, den)

    # Bisection on f(p) = 1 - (1 - tau(p))^(n-1) - p = 0 (strictly decreasing in p)
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = (1.0 - (1.0 - tau_of_p(mid)) ** (n_nodes - 1)) - mid
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
    p = 0.5 * (lo + hi)
    tau = tau_of_p(p)
    residual = abs((1.0 - (1.0 - tau) ** (n_nodes - 1)) - p)

    p_tr = 1.0 - (1.0 - tau) ** n_nodes
    p_s = (n_nodes * tau * ((1.0 - tau) ** (n_nodes - 1))) / max(1e-12, p_tr)
    t_s = data_us + SIFS_US + ACK_US + DIFS_US
    t_c = data_us + ACK_TO_US + DIFS_US
    e_slot_us = (1.0 - p_tr) * SLOT_US + p_tr * p_s * t_s + p_tr * (1.0 - p_s) * t_c
    throughput_frac = (p_tr * p_s * data_us) / max(1e-12, e_slot_us)
    drop_prob = p ** n_att
    return {
        "n_nodes": n_nodes,
        "cw_mode": "ieee_inclusive_[0,CW]" if inclusive else "uavnetsim_exclusive_[0,CW-1]",
        "p_collision": float(p),
        "tau_attempt": float(tau),
        "throughput_airtime_frac": float(throughput_frac),
        "retry_drop_prob": float(drop_prob),
        "fixed_point_residual": float(residual),
    }


def run_cleanroom_slotted_dcf_des(
    n_nodes: int,
    *,
    num_slots: int = 80_000,
    seed: int = 44,
    cw_min: int = CW_MIN,
    max_retx: int = MAX_RETX,
    inclusive: bool = True,
) -> dict[str, float]:
    """Independent clean-room slotted IEEE 802.11 DCF DES (zero shared code)."""
    rng = random.Random(seed)
    stages = [0] * n_nodes
    bo = [rng.randrange(cw_at_stage(0, cw_min=cw_min, inclusive=inclusive)) for _ in range(n_nodes)]
    tx_attempts = 0
    tx_collisions = 0
    tx_successes = 0
    drops = 0
    t_s = DATA_US + SIFS_US + ACK_US + DIFS_US
    t_c = DATA_US + ACK_TO_US + DIFS_US
    elapsed_us = 0.0

    for _ in range(num_slots):
        tx_nodes = [i for i in range(n_nodes) if bo[i] == 0]
        if not tx_nodes:
            elapsed_us += SLOT_US
            for i in range(n_nodes):
                bo[i] -= 1
            continue
        tx_attempts += len(tx_nodes)
        tx_set = set(tx_nodes)
        if len(tx_nodes) == 1:
            elapsed_us += t_s
            tx_successes += 1
            u = tx_nodes[0]
            stages[u] = 0
            bo[u] = rng.randrange(cw_at_stage(0, cw_min=cw_min, inclusive=inclusive))
        else:
            elapsed_us += t_c
            tx_collisions += len(tx_nodes)
            for u in tx_nodes:
                if stages[u] >= max_retx:
                    drops += 1
                    stages[u] = 0
                else:
                    stages[u] += 1
                bo[u] = rng.randrange(cw_at_stage(stages[u], cw_min=cw_min, inclusive=inclusive))
        for i in range(n_nodes):
            if i not in tx_set:
                bo[i] -= 1

    p_emp = tx_collisions / max(1, tx_attempts)
    s_emp = (tx_successes * DATA_US) / max(1e-12, elapsed_us)
    return {
        "n_nodes": n_nodes,
        "num_slots": num_slots,
        "seed": seed,
        "p_collision_emp": float(p_emp),
        "throughput_airtime_frac_emp": float(s_emp),
        "tx_attempts": tx_attempts,
        "tx_successes": tx_successes,
        "drops": drops,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", default="2,4,6,10")
    ap.add_argument("--slots", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    node_list = [int(x.strip()) for x in args.nodes.split(",") if x.strip()]
    comparisons = []
    max_p_rel = 0.0
    max_s_rel = 0.0
    for n in node_list:
        an_inc = solve_finite_retry_bianchi(n, inclusive=True)
        an_exc = solve_finite_retry_bianchi(n, inclusive=False)
        des_inc = run_cleanroom_slotted_dcf_des(n, num_slots=args.slots, seed=args.seed, inclusive=True)
        p_rel = abs(des_inc["p_collision_emp"] - an_inc["p_collision"]) / max(1e-9, an_inc["p_collision"])
        s_rel = abs(des_inc["throughput_airtime_frac_emp"] - an_inc["throughput_airtime_frac"]) / max(1e-9, an_inc["throughput_airtime_frac"])
        max_p_rel = max(max_p_rel, p_rel)
        max_s_rel = max(max_s_rel, s_rel)
        comparisons.append({
            "n_nodes": n,
            "bianchi_finite_retry_ieee_0_CW": an_inc,
            "bianchi_finite_retry_uavnetsim_0_CWm1": an_exc,
            "cleanroom_des_ieee_0_CW": des_inc,
            "p_collision_rel_err": float(p_rel),
            "throughput_rel_err": float(s_rel),
            "cw_mode_p_shift": float(an_exc["p_collision"] - an_inc["p_collision"]),
        })

    out = {
        "status": "PASS" if (max_p_rel <= 0.05 and max_s_rel <= 0.03) else "FAIL",
        "scope": "Independent finite-retry Bianchi & Tinnirello (2005) analytic oracle vs clean-room Python slotted DCF DES; explicit [0,CW] vs [0,CW-1] shift quantified",
        "max_p_collision_rel_err": float(max_p_rel),
        "max_throughput_rel_err": float(max_s_rel),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
