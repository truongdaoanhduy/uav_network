"""Bounded timestamp/priority/insertion-order event heap per world."""
import warp as wp
from ...environment.gpu.state import WorldParams, WorldState


@wp.func
def key_before(time_a: wp.float64, priority_a: int, order_a: wp.int64,
               time_b: wp.float64, priority_b: int, order_b: wp.int64) -> bool:
    return (time_a < time_b
        or (time_a == time_b
            and (priority_a < priority_b
                 or (priority_a == priority_b and order_a < order_b))))


@wp.func
def candidate_before_root(s: WorldState, e: int, time: wp.float64,
                          priority: int, order: wp.int64) -> bool:
    return (s.event_count[e] == 0
            or key_before(time, priority, order,
                          s.event_time[e,0], s.event_priority[e,0], s.event_order[e,0]))


@wp.func
def earlier(s: WorldState, e: int, a: int, b: int) -> bool:
    return key_before(s.event_time[e,a], s.event_priority[e,a], s.event_order[e,a],
                      s.event_time[e,b], s.event_priority[e,b], s.event_order[e,b])


@wp.func
def swap(s: WorldState, e: int, a: int, b: int):
    time = s.event_time[e,a]
    kind = s.event_kind[e,a]
    arg = s.event_arg[e,a]
    priority = s.event_priority[e,a]
    generation = s.event_generation[e,a]
    order = s.event_order[e,a]
    s.event_time[e,a] = s.event_time[e,b]
    s.event_kind[e,a] = s.event_kind[e,b]
    s.event_arg[e,a] = s.event_arg[e,b]
    s.event_priority[e,a] = s.event_priority[e,b]
    s.event_generation[e,a] = s.event_generation[e,b]
    s.event_order[e,a] = s.event_order[e,b]
    s.event_time[e,b] = time
    s.event_kind[e,b] = kind
    s.event_arg[e,b] = arg
    s.event_priority[e,b] = priority
    s.event_generation[e,b] = generation
    s.event_order[e,b] = order


@wp.func
def push(s: WorldState, p: WorldParams, e: int, time: wp.float64, kind: int,
         arg: int, generation: int, priority: int):
    i = s.event_count[e]
    if i >= p.event_capacity:
        s.errors[e] = 1
    else:
        s.event_count[e] += 1
        s.event_sequence[e] += wp.int64(1)
        s.event_time[e,i] = time
        s.event_kind[e,i] = kind
        s.event_arg[e,i] = arg
        s.event_generation[e,i] = generation
        s.event_priority[e,i] = priority
        s.event_order[e,i] = s.event_sequence[e]
        while i > 0:
            parent = (i-1)//2
            if not earlier(s,e,i,parent):
                break
            swap(s,e,i,parent)
            i = parent


@wp.func
def pop(s: WorldState, e: int):
    count = s.event_count[e]-1
    s.event_count[e] = count
    if count > 0:
        swap(s,e,0,count)
        i = int(0)
        while i*2+1 < count:
            child = i*2+1
            if child+1 < count and earlier(s,e,child+1,child):
                child += 1
            if not earlier(s,e,child,i):
                break
            swap(s,e,i,child)
            i = child


@wp.func
def next_time_after_root(s: WorldState, e: int, fallback: wp.float64) -> wp.float64:
    result = fallback
    count = s.event_count[e]
    if count > 1:
        child = int(1)
        if count > 2 and earlier(s,e,2,1):
            child = 2
        result = wp.min(result,s.event_time[e,child])
    return result

@wp.func
def replace_root(s: WorldState, e: int, time: wp.float64, kind: int, arg: int,
                 generation: int, priority: int):
    """Replace the current minimum with one newly inserted event.

    This is exactly the heap state transition of pop()+push() when the popped
    event only reschedules itself: event_count is unchanged and insertion order
    still advances by one.
    """
    s.event_sequence[e] += wp.int64(1)
    s.event_time[e,0] = time
    s.event_kind[e,0] = kind
    s.event_arg[e,0] = arg
    s.event_generation[e,0] = generation
    s.event_priority[e,0] = priority
    s.event_order[e,0] = s.event_sequence[e]
    count = s.event_count[e]
    i = int(0)
    while i*2+1 < count:
        child = i*2+1
        if child+1 < count and earlier(s,e,child+1,child):
            child += 1
        if not earlier(s,e,child,i):
            break
        swap(s,e,i,child)
        i = child
