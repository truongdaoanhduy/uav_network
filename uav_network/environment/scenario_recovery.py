"""Certified complete scenes bound random placement failure for a fixed protocol."""
from functools import lru_cache
import json
from types import SimpleNamespace

import numpy as np

SCENE_ATTEMPTS = 4


@lru_cache(maxsize=64)
def _certificates(identity):
    from .world import ScenePlacementError, _sample_scenario
    values = json.loads(identity)
    world = SimpleNamespace(**values["world"])
    # Certification has a fixed, bounded search budget independent of a reset's budget.
    world.placement_attempts = 5000
    config = SimpleNamespace(world=world,
                             sensor=SimpleNamespace(footprint_radius_m=values["footprint"]))
    scenes = []
    for seed in range(32):
        try:
            scene = _sample_scenario(config, np.random.default_rng(seed))
        except ScenePlacementError:
            continue
        for name in ("positions", "targets", "obstacles", "gcs"):
            getattr(scene, name).setflags(write=False)
        scenes.append(scene)
        if len(scenes) == 4:
            break
    if not scenes:
        raise ValueError("Scene generation has no certified complete fallback; "
                         "check obstacle packing, launch spacing and target exclusion")
    return tuple(scenes)


def certified_scenes(config):
    """Build once before random collection; cached arrays cannot be mutated by a reset."""
    identity = json.dumps({"world": vars(config.world),
                           "footprint": config.sensor.footprint_radius_m}, sort_keys=True)
    return _certificates(identity)
