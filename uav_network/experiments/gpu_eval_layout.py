"""Deterministic, collision-free Warp allocation for held-out RL evaluation.

Each checkpoint consumes exactly its configured number of independent worlds.
A checkpoint is never split across two Warp batches, and every batch is
aligned on checkpoint boundaries; optional unused capacity is not an episode.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationBatch:
    checkpoint_indices: tuple[int, ...]
    worlds: int
    checkpoint_ranges: tuple[tuple[int, int, int], ...]


def policies_per_validation_batch(capacity: int, episodes_per_checkpoint: int) -> int:
    """floor(C/E), where C is physical worlds/GPU and E is Validation episodes."""
    if not isinstance(capacity, int) or not isinstance(episodes_per_checkpoint, int):
        raise ValueError("Warp capacity and Validation episode count must be integers")
    if capacity < 1 or episodes_per_checkpoint < 1:
        raise ValueError("Warp capacity and Validation episode count must be positive")
    if episodes_per_checkpoint > capacity:
        raise ValueError("Cannot keep a whole checkpoint in one Warp batch: E > capacity")
    return capacity // episodes_per_checkpoint


def plan_validation(checkpoints: int, episodes: int, capacity: int,
                    gpu_index: int = 0, gpu_count: int = 1) -> tuple[ValidationBatch, ...]:
    """Plan GPU-strided checkpoints, preserving every checkpoint in exactly one batch."""
    if (not all(isinstance(x, int) for x in (checkpoints, gpu_index, gpu_count))
            or checkpoints < 0 or gpu_count < 1 or not 0 <= gpu_index < gpu_count):
        raise ValueError("Invalid checkpoint/GPU allocation")
    k = policies_per_validation_batch(capacity, episodes)
    indices = tuple(range(gpu_index, checkpoints, gpu_count))
    result = []
    for first in range(0, len(indices), k):
        selected = indices[first:first+k]
        offsets = tuple((idx, slot*episodes, (slot+1)*episodes)
                        for slot, idx in enumerate(selected))
        result.append(ValidationBatch(selected, len(selected)*episodes, offsets))
    if sum(len(item.checkpoint_indices) for item in result) != len(indices):
        raise AssertionError("Checkpoint missing from Warp layout")
    if any(batch.worlds > capacity for batch in result):
        raise AssertionError("Warp pool over-allocated")
    return tuple(result)


def plan_test(total_episodes: int, gpus: int, capacity: int) -> tuple[tuple[tuple[int,int], ...], ...]:
    """Partition held-out episodes per GPU, never duplicating seeds.

    The requested 4,000 episodes / 2 GPUs / >=2,000 capacity results in
    ONE 2,000-world Warp rollout per GPU and simultaneous GPU processes.
    Larger test banks are split only when physical capacity demands it.
    """
    if (not all(isinstance(x, int) for x in (total_episodes, gpus, capacity))
            or total_episodes < 1 or gpus not in (1, 2) or capacity < 1):
        raise ValueError("Invalid Final Test count/GPU capacity")
    workers = min(total_episodes, gpus)
    partitions = []
    for gpu in range(workers):
        first = (total_episodes*gpu)//workers
        stop = (total_episodes*(gpu+1))//workers
        partitions.append(tuple((offset, min(offset+capacity, stop))
                                for offset in range(first, stop, capacity)))
    observed = [i for part in partitions for a,b in part for i in range(a,b)]
    if observed != list(range(total_episodes)):
        raise AssertionError("Final Test seed ranges not contiguous/non-overlapping")
    return tuple(partitions)
