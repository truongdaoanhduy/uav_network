"""Opt-in XYZ flight and Bayesian camera settings; legacy defaults stay XY."""
from dataclasses import dataclass, field, asdict
import math


@dataclass
class FlightSettings:
    motion_dimensions: int = 2
    altitude_min_m: float = 0.0
    altitude_max_m: float = 150.0
    obstacle_height_min_m: float = 30.0
    obstacle_height_max_m: float = 120.0


@dataclass
class BayesianSettings:
    model: str = "exact"
    grid_cell_m: float = 25.0
    belief_prior: float = 0.5
    altitude_anchors: list = field(default_factory=lambda: [0.,50.,100.,150.])
    pd: list = field(default_factory=lambda: [.99,.90,.80,.70])
    pf: list = field(default_factory=lambda: [.01,.10,.20,.30])
    confirmation_threshold: float = .99
    fine_altitude_m: float = 50.0
    belief_patch_cells: int = 5


def validate_xyz(config):
    w,s = config.world,config.sensor
    if type(w.motion_dimensions) is not int or w.motion_dimensions not in (2,3):
        raise ValueError("motion_dimensions must be 2 or 3")
    if s.model not in ("exact","bayesian") or (w.motion_dimensions==3)!=(s.model=="bayesian"):
        raise ValueError("XYZ motion requires Bayesian sensing; legacy XY uses exact sensing")
    numbers = [w.altitude_min_m,w.altitude_max_m,w.obstacle_height_min_m,
               w.obstacle_height_max_m,s.grid_cell_m,s.belief_prior,
               s.confirmation_threshold,s.fine_altitude_m,*s.altitude_anchors,*s.pd,*s.pf]
    if any(type(v) not in (float,int) or not math.isfinite(v) for v in numbers):
        raise ValueError("XYZ camera settings must be finite numbers")
    if not (0<=w.altitude_min_m<w.altitude_max_m and
            0<w.obstacle_height_min_m<=w.obstacle_height_max_m and
            s.grid_cell_m>0 and 0<s.belief_prior<1 and 0<s.confirmation_threshold<1):
        raise ValueError("Invalid XYZ bounds, grid or probabilities")
    if (len(s.altitude_anchors)!=4 or len(s.pd)!=4 or len(s.pf)!=4
            or any(a>=b for a,b in zip(s.altitude_anchors,s.altitude_anchors[1:]))
            or any(not 0<pf<pd<1 for pd,pf in zip(s.pd,s.pf))):
        raise ValueError("Camera needs four ordered altitude anchors and 0<Pf<Pd<1")
    if w.motion_dimensions==3 and not (w.altitude_min_m<=s.fine_altitude_m<=w.altitude_max_m
            and w.altitude_min_m<=s.altitude_m<=w.altitude_max_m):
        raise ValueError("Initial/fine altitude outside flight bounds")
    if type(s.belief_patch_cells) is not int or not 1<=s.belief_patch_cells<=13 or s.belief_patch_cells%2!=1:
        raise ValueError("belief_patch_cells must be an odd integer in [1,13]")
    if math.ceil(w.map_size_m/s.grid_cell_m)>128:
        raise ValueError("Bayesian grid exceeds 128 cells per axis")


def legacy_xyz_defaults():
    return {"world":asdict(FlightSettings()),"sensor":asdict(BayesianSettings())}


def strip_legacy_xyz_defaults(values):
    """Normalize only unused defaults, so existing XY scientific identities survive."""
    if values.get("world",{}).get("motion_dimensions",2)==2:
        for group,defaults in legacy_xyz_defaults().items():
            for key,value in defaults.items():
                if values[group].get(key)==value:
                    values[group].pop(key,None)
    return values
