"""Repeat native-network smoke training from scratch with one root seed."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from uav_network.configuration import load_config
from uav_network.experiments.training import train


def check(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base = load_config("configs/smoke-parallel.yaml")
    base.world.motion_safety = "dynamic_window"
    base.network.receiver_admission_mask = True
    results = {}
    for algorithm in ("masac", "matd3"):
        runs = []
        for repeat in (1, 2):
            config = deepcopy(base)
            config.runtime.algorithm = algorithm
            config.runtime.output_dir = str(output / f"{algorithm}-{repeat}")
            runs.append(train(config))
        fields = ("model_hash", "episode_hash", "settings_hash")
        exact = all(runs[0]["reproducibility"][k] == runs[1]["reproducibility"][k] for k in fields)
        results[algorithm] = {"repeat_exact": exact,
            "episodes": runs[0]["episodes"], "global_step": runs[0]["global_step"],
            "updates": runs[0]["updates"], "reproducibility": runs[0]["reproducibility"]}
        if not exact:
            raise AssertionError(f"{algorithm} did not repeat exactly")
    (output / "validation.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    check(parser.parse_args().output)
