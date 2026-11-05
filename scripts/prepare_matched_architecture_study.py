#!/usr/bin/env python3
"""Prepare ten matched run configs, without submitting cloud jobs."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import yaml

from uav_network.configuration import load_config

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "configs/comparison-reference-masac-matd3-xyz-sensing.yaml"
DESTINATION = ROOT / "outputs/matched-architecture-seed44-20261104"

# (algorithm, PQL, shared actor/critic, vectorized critic)
CASES = [
    ("masac", False, False, False),
    ("masac", False, True, False),
    ("masac", True, False, False),
    ("masac", True, True, False),
    ("matd3", False, False, False),
    ("matd3", False, True, False),
    ("matd3", True, False, False),
    ("matd3", True, True, False),
    ("masac", False, True, True),
    ("masac", True, True, False),  # exact scientific replica of case 04
]


def scientific_settings(config):
    data = deepcopy(config.to_dict())
    data["runtime"].pop("output_dir")
    return data


def prepare(destination=DESTINATION):
    destination = Path(destination)
    base = load_config(REFERENCE)
    assert base.runtime.seed == 44
    assert base.runtime.deterministic and base.runtime.critic_chunk_size == 2048
    assert not base.runtime.vectorize_critics
    assert base.runtime.amp_mode == "off" and not base.runtime.compile_models
    assert not base.matd3.output_normalization
    base.runtime.collector_overlap = False  # fixed-order BASE collection, matching deterministic PQL
    common = base.to_dict()
    results = []
    for number, (algorithm, pql, shared, vectorized) in enumerate(CASES, 1):
        config = deepcopy(base)
        config.runtime.algorithm = algorithm
        config.runtime.vectorize_critics = vectorized
        config.pql.enabled = pql
        group = getattr(config, algorithm)
        group.share_actor_parameters = shared
        group.share_critic_parameters = shared
        name = f"{number:02d}-{algorithm}-{'pql' if pql else 'base'}-{'shared' if shared else 'independent'}"
        if vectorized:
            name += "-vmap"
        if number == 10:
            name += "-replica"
        config.runtime.output_dir = str(Path("outputs/matched-architecture-seed44-20261104/training") / name)
        config.tracking.group = "matched-architecture-seed44-20261104"
        config.tracking.tags = list(base.tracking.tags) + ["matched-architecture-study"]
        config.validate()
        data = config.to_dict()
        # Scientific conditions identical except four experimental axes:
        # algorithm, BASE/PQL, parameter sharing, and vectorization.
        for group_name in ("world", "sensor", "network", "energy", "reward"):
            assert data[group_name] == common[group_name]
        assert all(data["runtime"][key] == common["runtime"][key]
                   for key in ("seed", "eval_seed", "num_envs", "episodes", "deterministic",
                               "critic_chunk_size", "collector_overlap", "amp_mode", "compile_models",
                               "fused_optimizers", "replay_device", "replay_cpu_fraction"))
        assert group.share_actor_parameters == group.share_critic_parameters == shared
        config_path = destination / "configs" / f"{name}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(data, sort_keys=False))
        assert scientific_settings(load_config(config_path)) == scientific_settings(config)
        results.append({"id": number, "name": name, "algorithm": algorithm,
                        "pql": pql, "shared": shared, "vectorize_critics": vectorized,
                        "config": str(config_path), "output_dir": config.runtime.output_dir})
    first = load_config(results[3]["config"])
    replica = load_config(results[9]["config"])
    assert scientific_settings(first) == scientific_settings(replica)
    # The BASE vectorization pair has exactly one changed scientific flag.
    plain = load_config(results[1]["config"])
    vmap = load_config(results[8]["config"])
    vmap.runtime.vectorize_critics = False
    assert scientific_settings(plain) == scientific_settings(vmap)
    manifest = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "reference": str(REFERENCE), "seed": 44, "launched": False,
                "notes": ["No Kaggle jobs submitted", "Case 04 and 10 are scientific replicas",
                          "MASAC shared critic does not invoke independent-critic torch.vmap"],
                "cases": results}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    manifest = prepare()
    print(json.dumps({"manifest": str(DESTINATION / "manifest.json"),
                      "cases": len(manifest["cases"]), "replica_checked": True,
                      "launched": False}, indent=2))
