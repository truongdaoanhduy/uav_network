"""Correlated, altitude/visibility-calibrated binary sensor observations.

A stationary per-cell *shared* nuisance sign is latent to all UAVs. It is
sampled independently of target occupancy and remains fixed during a mission.
Conditioned on the nuisance, independent Bernoulli draws are used at each
observation; marginal p(z=1 | occupied)=p1 and p(z=1 | empty)=p0.
This induces temporal AND cross-UAV dependence without oracle access at nodes.
It is a declared scientific model, not a claim about a real optical camera.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def _sensor_params(p1: float, p0: float, correlation_strength: float) -> None:
    vals = (p1, p0, correlation_strength)
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("sensor parameters must be finite")
    if not (0 < p0 < p1 < 1):
        raise ValueError("sensor must be informative: 0<p0<p1<1")
    if not (0 <= correlation_strength <= 1):
        raise ValueError("correlation strength must be in [0,1]")


@dataclass(frozen=True)
class Event:
    """Noisy observation only. No oracle target identity or ground truth field."""
    event_id: str
    origin: int
    tick: int
    cell: int
    layer: str
    positive: bool
    p1: float  # P(z=1 | occupied), marginalized over shared nuisance
    p0: float  # P(z=1 | empty), marginalized over shared nuisance
    correlation_strength: float = 0.0

    def __post_init__(self):
        if not self.event_id or self.origin < 0 or self.tick < 0 or self.cell < 0:
            raise ValueError("event identity, node, tick and cell must be valid")
        if self.layer not in ("target", "obstacle"):
            raise ValueError("unknown observation layer")
        _sensor_params(self.p1, self.p0, self.correlation_strength)

    @property
    def sort_key(self):
        return (self.tick, self.origin, self.event_id)


@dataclass(frozen=True)
class CellEvidence:
    """Effective occupancy likelihood for partial FoV under a cell mixture.

    The target location is assumed uniformly distributed within the cell for
    the occupancy H1 likelihood; actual target coordinates never enter this
    function. Occluded cells are not measured.
    """
    p1: float
    p0: float
    visible_fraction: float

    @classmethod
    def from_footprint(cls, pd: float, pf: float, q: float):
        if not (0 < pf < pd < 1) or not (0 <= q <= 1):
            raise ValueError("invalid Pd, Pf or visible fraction")
        if q == 0:
            return None  # unseen is never a negative observation
        p0 = 1 - (1 - pf) ** q
        p1 = q * pd + (1 - q) * p0
        return cls(p1=p1, p0=p0, visible_fraction=q)


def _component_positive(p: float, sign: int, rho: float) -> float:
    # Valid probability, with E[sign]=0 and exactly preserved marginal p.
    return p + sign * rho * min(p, 1 - p)


class CorrelatedBinarySensor:
    """Ground-truth-bearing simulator ONLY; never pass this object to Actor."""
    def __init__(self, num_cells: int, seed: int, correlation_strength: float):
        if num_cells <= 0 or not (0 <= correlation_strength <= 1):
            raise ValueError("invalid grid or correlation")
        self.rng = np.random.default_rng(seed)
        self.shared_sign = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=num_cells)
        self.rho = float(correlation_strength)

    def sample(self, cell: int, p1: float, p0: float, occupied: bool) -> bool:
        _sensor_params(p1, p0, self.rho)
        if not (0 <= cell < len(self.shared_sign)):
            raise IndexError("cell outside sensor map")
        p = p1 if occupied else p0
        conditional_p = _component_positive(p, int(self.shared_sign[cell]), self.rho)
        return bool(self.rng.random() < conditional_p)


class SharedBiasLikelihood:
    """Exact sequential likelihood ratio for the shared-bias mixture model.

    Maintains two conditional posteriors over the unknown environmental sign,
    one assuming the cell occupied and the other assuming it is empty. The
    log-likelihood increment conditions on ALL previously processed events,
    regardless of which UAV recorded them. Events must be replayed by sort_key.
    """
    def __init__(self):
        self.weights1 = [.5, .5]  # posterior P(shared_sign=-1,+1 | H1, history)
        self.weights0 = [.5, .5]  # posterior P(shared_sign=-1,+1 | H0, history)
        self.observation_count = 0
        self.last_probabilities = None

    def observe(self, event: Event) -> float:
        _sensor_params(event.p1, event.p0, event.correlation_strength)
        ps1 = [_component_positive(event.p1, sign, event.correlation_strength) for sign in (-1, 1)]
        ps0 = [_component_positive(event.p0, sign, event.correlation_strength) for sign in (-1, 1)]
        conditional_p1 = sum(w*p for w, p in zip(self.weights1, ps1))
        conditional_p0 = sum(w*p for w, p in zip(self.weights0, ps0))
        # Log likelihood for the actually observed outcome, not only positives.
        l1 = conditional_p1 if event.positive else 1 - conditional_p1
        l0 = conditional_p0 if event.positive else 1 - conditional_p0
        if l1 <= 0 or l0 <= 0:
            raise ValueError("degenerate conditional sensor likelihood")
        llr = math.log(l1 / l0)
        for prior, component in ((self.weights1, ps1), (self.weights0, ps0)):
            evidence = [w*(p if event.positive else 1-p) for w,p in zip(prior,component)]
            total = sum(evidence)
            prior[:] = [v/total for v in evidence]
        self.observation_count += 1
        self.last_probabilities = (conditional_p1, conditional_p0)
        return llr
