"""Derived frame availability bitmaps for the experimental framefree scheduler."""
import warp as wp

from ...environment.gpu.state import WorldParams, WorldState

wp.set_module_options({"enable_backward": False})


def bitmap_words(frame_capacity: int) -> int:
    if type(frame_capacity) is not int or frame_capacity < 0:
        raise ValueError("frame_capacity must be a nonnegative integer")
    return (frame_capacity + 63) // 64


@wp.func
def _bit(frame: int) -> wp.uint64:
    return wp.uint64(1) << wp.uint64(frame % 64)


@wp.func
def set_free_bit(free_words: wp.array2d(dtype=wp.uint64), e: int, frame: int):
    word = frame // 64
    free_words[e, word] = free_words[e, word] | _bit(frame)


@wp.func
def clear_free_bit(free_words: wp.array2d(dtype=wp.uint64), e: int, frame: int):
    word = frame // 64
    free_words[e, word] = free_words[e, word] & ~_bit(frame)


@wp.func
def set_reclaim_bit(reclaim_words: wp.array2d(dtype=wp.uint64), e: int, frame: int):
    word = frame // 64
    reclaim_words[e, word] = reclaim_words[e, word] | _bit(frame)


@wp.func
def clear_reclaim_bit(reclaim_words: wp.array2d(dtype=wp.uint64), e: int, frame: int):
    word = frame // 64
    reclaim_words[e, word] = reclaim_words[e, word] & ~_bit(frame)


@wp.func
def rebuild_frame_bitmaps(s: WorldState, p: WorldParams, e: int,
                          free_words: wp.array2d(dtype=wp.uint64),
                          reclaim_words: wp.array2d(dtype=wp.uint64)):
    words = (p.frame_capacity + 63) // 64
    for word in range(words):
        free_words[e, word] = wp.uint64(0)
        reclaim_words[e, word] = wp.uint64(0)
    for frame in range(p.frame_capacity):
        if s.tx_active[e, frame] == 0:
            set_free_bit(free_words, e, frame)
        elif s.tx_active[e, frame] == 2 and s.tx_receive[e, frame] == 3:
            set_reclaim_bit(reclaim_words, e, frame)


@wp.func
def mark_reclaim_if_ready(s: WorldState, e: int, frame: int,
                          reclaim_words: wp.array2d(dtype=wp.uint64)):
    if frame >= 0 and s.tx_active[e, frame] == 2 and s.tx_receive[e, frame] == 3:
        set_reclaim_bit(reclaim_words, e, frame)


@wp.func
def apply_reclaims_and_take_smallest(s: WorldState, p: WorldParams, e: int,
                                     free_words: wp.array2d(dtype=wp.uint64),
                                     reclaim_words: wp.array2d(dtype=wp.uint64)) -> int:
    words = (p.frame_capacity + 63) // 64
    for word in range(words):
        pending = reclaim_words[e, word]
        if pending != wp.uint64(0):
            for bit_index in range(64):
                frame = word * 64 + bit_index
                bit = wp.uint64(1) << wp.uint64(bit_index)
                if frame < p.frame_capacity and (pending & bit) != wp.uint64(0):
                    s.tx_active[e, frame] = 0
                    free_words[e, word] = free_words[e, word] | bit
            reclaim_words[e, word] = wp.uint64(0)
    selected = int(-1)
    for word in range(words):
        available = free_words[e, word]
        if selected < 0 and available != wp.uint64(0):
            for bit_index in range(64):
                frame = word * 64 + bit_index
                bit = wp.uint64(1) << wp.uint64(bit_index)
                if (selected < 0 and frame < p.frame_capacity
                        and (available & bit) != wp.uint64(0)):
                    selected = frame
                    free_words[e, word] = free_words[e, word] & ~bit
    return selected


@wp.kernel
def rebuild_frame_bitmaps_kernel(s: WorldState, p: WorldParams,
                                 free_words: wp.array2d(dtype=wp.uint64),
                                 reclaim_words: wp.array2d(dtype=wp.uint64)):
    rebuild_frame_bitmaps(s, p, wp.tid(), free_words, reclaim_words)


@wp.kernel
def mark_reclaim_if_ready_kernel(s: WorldState, frame: int,
                                 reclaim_words: wp.array2d(dtype=wp.uint64)):
    mark_reclaim_if_ready(s, wp.tid(), frame, reclaim_words)


@wp.kernel
def apply_reclaims_and_take_smallest_kernel(s: WorldState, p: WorldParams,
                                            free_words: wp.array2d(dtype=wp.uint64),
                                            reclaim_words: wp.array2d(dtype=wp.uint64),
                                            selected: wp.array(dtype=wp.int32)):
    e = wp.tid()
    selected[e] = apply_reclaims_and_take_smallest(s, p, e, free_words, reclaim_words)

@wp.kernel
def rebuild_frame_bitmaps_masked_kernel(s: WorldState, p: WorldParams,
                                        mask: wp.array(dtype=wp.int32),
                                        free_words: wp.array2d(dtype=wp.uint64),
                                        reclaim_words: wp.array2d(dtype=wp.uint64)):
    e = wp.tid()
    if mask[e] != 0:
        rebuild_frame_bitmaps(s, p, e, free_words, reclaim_words)
