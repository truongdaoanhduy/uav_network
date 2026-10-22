"""Derived per-receiver pending-frame heap for the experimental rxindex scheduler."""
import warp as wp

from ...environment.gpu.state import WorldParams, WorldState
from .events import push


@wp.func
def receiver_before(s: WorldState, e: int, left: int, right: int) -> bool:
    left_order = s.tx_receive_order[e,left]
    right_order = s.tx_receive_order[e,right]
    return left_order < right_order or (left_order == right_order and left < right)


@wp.func
def receiver_push(s: WorldState, p: WorldParams, e: int, receiver: int,
                  frames: wp.array3d(dtype=wp.int32),
                  sizes: wp.array2d(dtype=wp.int32), frame: int):
    i = sizes[e,receiver]
    frames[e,receiver,i] = frame
    sizes[e,receiver] = i + 1
    while i > 0:
        parent = (i-1)//2
        parent_frame = frames[e,receiver,parent]
        current = frames[e,receiver,i]
        if not receiver_before(s,e,current,parent_frame):
            break
        frames[e,receiver,parent] = current
        frames[e,receiver,i] = parent_frame
        i = parent


@wp.func
def receiver_pop_min(s: WorldState, p: WorldParams, e: int, receiver: int,
                     frames: wp.array3d(dtype=wp.int32),
                     sizes: wp.array2d(dtype=wp.int32)) -> int:
    count = sizes[e,receiver]
    if count <= 0:
        return -1
    result = frames[e,receiver,0]
    count -= 1
    sizes[e,receiver] = count
    if count > 0:
        frames[e,receiver,0] = frames[e,receiver,count]
        i = int(0)
        while i*2+1 < count:
            child = i*2+1
            if child+1 < count:
                left = frames[e,receiver,child]
                right = frames[e,receiver,child+1]
                if receiver_before(s,e,right,left):
                    child += 1
            child_frame = frames[e,receiver,child]
            current = frames[e,receiver,i]
            if not receiver_before(s,e,child_frame,current):
                break
            frames[e,receiver,i] = child_frame
            frames[e,receiver,child] = current
            i = child
    return result


@wp.func
def next_receive_indexed(s: WorldState, p: WorldParams, e: int, receiver: int,
                         now: wp.float64, receive_kind: int,
                         frames: wp.array3d(dtype=wp.int32),
                         sizes: wp.array2d(dtype=wp.int32)):
    if s.node_receiving[e,receiver] == 0:
        frame = receiver_pop_min(s,p,e,receiver,frames,sizes)
        if frame >= 0:
            s.node_receiving[e,receiver] = 1
            s.tx_receive[e,frame] = 2
            push(s,p,e,now,receive_kind,frame,0,0)


@wp.kernel
def reset_receiver_index(p: WorldParams, mask: wp.array(dtype=wp.int32),
                         frames: wp.array3d(dtype=wp.int32),
                         sizes: wp.array2d(dtype=wp.int32)):
    e = wp.tid()
    if mask[e] != 0:
        for receiver in range(p.n+1):
            sizes[e,receiver] = 0


@wp.kernel
def rebuild_receiver_index(s: WorldState, p: WorldParams,
                           frames: wp.array3d(dtype=wp.int32),
                           sizes: wp.array2d(dtype=wp.int32)):
    e = wp.tid()
    for receiver in range(p.n+1):
        sizes[e,receiver] = 0
    for frame in range(p.frame_capacity):
        if (s.tx_active[e,frame] != 0 and s.tx_receive[e,frame] == 1):
            receiver = s.tx_destination[e,frame]
            if receiver >= 0 and receiver <= p.n:
                receiver_push(s,p,e,receiver,frames,sizes,frame)
