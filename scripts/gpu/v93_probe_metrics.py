"""Terminal snapshots and paired, visited-cell sensor reliability diagnostics.

Adaptive ECE here is a binary positive-class, count-weighted, quantile-bin
diagnostic with ties kept together. It is not an implementation of multiclass
ACE. DS probability scoring uses BetP(T)=m(T)+m(Theta)/2, not m(T) alone.
"""
from __future__ import annotations
import math
import numpy as np


class TerminalSnapshots:
    """Save each world's first terminal values before any subsequent step/reset."""
    def __init__(self, envs: int):
        self.seen = np.zeros(envs, dtype=bool)
        self.values = {}

    def capture(self, selected, values):
        chosen = np.asarray(selected, dtype=bool) & ~self.seen
        if not chosen.any():
            return
        for key, value in values.items():
            a = np.asarray(value)
            if a.shape[0] != len(self.seen):
                raise ValueError("snapshot values must have one row per world")
            if key not in self.values:
                self.values[key] = np.zeros_like(a)
            self.values[key][chosen] = a[chosen]
        self.seen |= chosen


def binary_reliability(probabilities, labels, bins=15):
    p, y = np.asarray(probabilities, dtype=np.float64).ravel(), np.asarray(labels).ravel()
    if p.shape != y.shape or type(bins) is not int or bins < 1:
        raise ValueError("aligned probability/label vectors and positive bins required")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError("finite probabilities in [0,1] and binary labels required")
    if not len(p):
        return {"n": 0, "brier": None, "ece_equal_width": None,
                "adaptive_ece_positive_class": None, "prevalence": None,
                "adaptive_bins": []}

    def grouped(ids):
        rows = []
        for k in np.unique(ids):
            sel = ids == k
            rows.append({"n": int(sel.sum()), "mean_probability": float(p[sel].mean()),
                         "positive_frequency": float(y[sel].mean())})
        error = sum(r["n"] * abs(r["mean_probability"] - r["positive_frequency"])
                    for r in rows) / len(p)
        return float(error), rows

    fixed_ids = np.minimum((p * bins).astype(int), bins - 1)
    cuts = np.unique(np.quantile(p, np.linspace(0, 1, min(bins, len(p)) + 1)[1:-1]))
    adaptive_ids = np.searchsorted(cuts, p, side="right")
    fixed, _ = grouped(fixed_ids)
    adaptive, rows = grouped(adaptive_ids)
    return {"n": int(len(p)), "brier": float(np.mean((p - y) ** 2)),
            "ece_equal_width": fixed, "adaptive_ece_positive_class": adaptive,
            "prevalence": float(y.mean()), "adaptive_bins": rows}


def compare_counts(pos, neg, truth, *, prior, pd, pf, mf, me):
    """Score both estimators on exactly the same independent origin counts."""
    from zhang_xue.belief.dempster_shafer import counts_to_canonical_ds_stable
    p, n = np.asarray(pos, dtype=np.int64).ravel(), np.asarray(neg, dtype=np.int64).ravel()
    y = np.asarray(truth, dtype=np.uint8).ravel()
    if p.shape != n.shape or p.shape != y.shape or np.any(p < 0) or np.any(n < 0):
        raise ValueError("aligned nonnegative count arrays required")
    observed = (p + n) > 0
    p, n, y = p[observed], n[observed], y[observed]
    logits = math.log(prior / (1-prior)) + p * math.log(pd/pf) + n * math.log((1-pd)/(1-pf))
    bayes = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    ds = np.asarray([counts_to_canonical_ds_stable(int(a), int(b), mf, me)
                     for a,b in zip(p,n)], dtype=np.float64).reshape(-1,3)
    betp = ds[:,0] + .5 * ds[:,2]
    return {"visited_cells": int(len(p)),
            "visits_min": int((p+n).min()) if len(p) else None,
            "visits_max": int((p+n).max()) if len(p) else None,
            "bayes": binary_reliability(bayes, y),
            "ds_pignistic": binary_reliability(betp, y),
            "ds_target_mass_diagnostic": binary_reliability(ds[:,0], y),
            "bayes_posterior": bayes.tolist()}

