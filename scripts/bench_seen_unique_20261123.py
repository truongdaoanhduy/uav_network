"""Isolated sampled-row marking microbenchmark. Does NOT benchmark full MARL."""
import argparse
import json
import time
import torch


def measure(capacity=1_000_000, batch=8192, repeats=300, device="cpu"):
    target = torch.device(device)
    generator = torch.Generator(device=target).manual_seed(44)
    indices = [
        torch.randint(capacity, (batch,), generator=generator, device=target)
        for _ in range(repeats)
    ]

    def bench(use_unique):
        seen = torch.zeros(capacity, dtype=torch.bool, device=target)
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        started = time.perf_counter()
        for batch_index in indices:
            if use_unique:
                seen[batch_index.unique()] = True
            else:
                seen[batch_index] = True
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        elapsed = time.perf_counter() - started
        return seen, elapsed

    # Separate preliminary runs for warm caches, not part of measurement.
    bench(True)
    bench(False)
    old, old_time = bench(True)
    new, new_time = bench(False)
    if not torch.equal(old, new):
        raise AssertionError("Regression: direct marking differs from unique reference")
    return {
        "device": str(target), "capacity": capacity, "batch": batch, "repeats": repeats,
        "unique_s": old_time, "direct_s": new_time,
        "marking_speedup": old_time / new_time,
        "semantic_parity": True,
        "scope": "isolated replay sampling metadata only; no training speed claim",
    }


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    parser.add_argument("--repeats", type=int, default=300)
    args=parser.parse_args()
    print(json.dumps(measure(device=args.device, repeats=args.repeats),indent=2))
