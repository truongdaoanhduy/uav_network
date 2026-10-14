"""Bounded timestamp/priority/insertion-order event heap per world."""
import warp as wp
from ...environment.gpu.state import WorldParams, WorldState


@wp.func
def earlier(s: WorldState, e: int, a: int, b: int) -> bool:
    return (s.event_time[e,a] < s.event_time[e,b]
        or (s.event_time[e,a] == s.event_time[e,b]
            and (s.event_priority[e,a] < s.event_priority[e,b]
                 or (s.event_priority[e,a] == s.event_priority[e,b]
                     and s.event_order[e,a] < s.event_order[e,b]))))


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
