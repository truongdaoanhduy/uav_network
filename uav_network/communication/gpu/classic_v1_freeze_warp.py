"""Reproduce upstream classic UavNetSim-v1's interrupted CSMA backoff.

This is a bounded externally specified busy-carrier schedule, NOT a full
MAC network scheduler. The upstream CsmaCa.mac_send waits for idle,
counting DIFS+backoff. If an occupied channel interrupts DIFS, the whole
DIFS+backoff restarts; if interrupted during backoff, the remaining
waiting time becomes frozen backoff, preceded by another full DIFS.
Carrier polling after interruption is quantized to source SLOT_DURATION.

Inputs are actual busy resource intervals from a separate MAC trace,
not inferred from an oracle route or network success. No random draws are
created here; the caller provides source-equivalent randint slots.
"""
from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def _freeze_mac(
    slots:wp.array(dtype=wp.int32,ndim=2),
    busy_start:wp.array(dtype=wp.int32,ndim=3),
    busy_end:wp.array(dtype=wp.int32,ndim=3),
    num_busy:wp.array(dtype=wp.int32,ndim=2),
    first_tx:wp.array(dtype=wp.int32,ndim=2),
    num_interruptions:wp.array(dtype=wp.int32,ndim=2),
    slot_duration_us:int,
    difs_duration_us:int,
):
    e,u=wp.tid()
    backoff=slots[e,u]*slot_duration_us
    remaining=difs_duration_us+backoff
    now=wp.int32(0)
    interruptions=wp.int32(0)
    for k in range(4):
        if k>=num_busy[e,u]:
            break
        begin=busy_start[e,u,k]
        finish=busy_end[e,u,k]
        if begin>=now+remaining:
            break  # MAC packet sends before this external carrier starts.
        if finish<=now:
            continue
        if begin<=now:
            # Initially busy: the source has not started any timer,
            # so the backoff is unchanged; wait_idle_channel polls
            # every SLOT_DURATION from this point.
            wait=finish-now
            now+=((wait+slot_duration_us-1)//slot_duration_us)*slot_duration_us
            continue
        # The native listen() coroutine checks channel every 1µs.
        # On busy-carrier start, it interrupts mac_send. In upstream,
        # elapsed "to_wait" is removed from current countdown.
        elapsed=begin-now
        leftover=remaining-elapsed
        if leftover>backoff:
            # In DIFS, the old backoff remains intact.
            remaining=difs_duration_us+backoff
        else:
            # In backoff, upstream stores remaining total to_wait as
            # the new frozen backoff, and adds DIFS after idle again.
            backoff=leftover
            remaining=difs_duration_us+backoff
        interruptions+=1
        now=begin
        # The exact upstream wait_idle_channel coroutine polls in
        # intervals of SLOT_DURATION even though listen polls at1µs.
        wait=finish-now
        now+=((wait+slot_duration_us-1)//slot_duration_us)*slot_duration_us
    first_tx[e,u]=now+remaining
    num_interruptions[e,u]=interruptions


def compute_backoff_freeze_schedule(
    backoff_draw_slots,
    busy_start_us,
    busy_end_us,
    busy_interval_count,
    *,
    transmission_attempts=None,
    slot_duration_us:int=20,
    difs_duration_us:int=50,
    device="cpu",
):
    """Warp per-sender first-transmission clock with explicit busy events.

    Time is in integer microseconds, four non-overlapping sorted busy
    intervals max. Each independent [E,N] sender follows its actual
    upstream backoff semantics, and provided attempt index validates
    the correct native CW=32*2**(attempt-1)-1 randint bound.
    Results are [E,N] INT32.
    """
    draws=np.asarray(backoff_draw_slots)
    starts=np.asarray(busy_start_us)
    ends=np.asarray(busy_end_us)
    counts=np.asarray(busy_interval_count)
    if draws.ndim!=2 or draws.size==0:
        raise ValueError("backoff_draw_slots must be nonempty [E,N]")
    if starts.shape!=(*draws.shape,4) or ends.shape!=starts.shape:
        raise ValueError("busy start/end must be [E,N,4]")
    if counts.shape!=draws.shape:
        raise ValueError("busy_interval_count must be [E,N]")
    if any(not np.issubdtype(a.dtype,np.integer)
           for a in (draws,starts,ends,counts)):
        raise ValueError("all timing fields must be integers")
    if int(slot_duration_us)<=0 or int(difs_duration_us)<0:
        raise ValueError("invalid MAC timing constants")
    if transmission_attempts is None:
        attempts=np.ones(draws.shape,dtype=np.int32)
    else:
        attempts=np.asarray(transmission_attempts)
        if (attempts.shape!=draws.shape or
            not np.issubdtype(attempts.dtype,np.integer)):
            raise ValueError("transmission_attempts must be integer [E,N]")
        if np.any((attempts<1)|(attempts>5)):
            raise ValueError("native retransmission attempts must be 1..5")
    contention_window=32*(2**(attempts-1))-1
    if np.any((draws<0)|(draws>=contention_window)):
        raise ValueError("draw outside native exponential contention window")
    if np.any((counts<0)|(counts>4)):
        raise ValueError("busy interval count must be 0..4")
    valid=np.arange(4)[None,None,:]<counts[:,:,None]
    if np.any((starts[valid]<0)|(ends[valid]<=starts[valid])):
        raise ValueError("busy intervals need start>=0 and end>start")
    if np.any(
        valid[:,:,1:]&
        (starts[:,:,1:]<ends[:,:,:-1])
    ):
        raise ValueError("busy intervals must be sorted non-overlapping")
    if np.any(ends[valid]>1_000_000_000):
        raise ValueError("outside supported bounded 32-bit time horizon")
    wp.init()
    def arr(a):
        return wp.array(a.astype(np.int32),dtype=wp.int32,device=device)
    first=wp.zeros(draws.shape,dtype=wp.int32,device=device)
    interrupted=wp.zeros(draws.shape,dtype=wp.int32,device=device)
    wp.launch(_freeze_mac,dim=draws.shape,inputs=[
        arr(draws),arr(starts),arr(ends),arr(counts),first,interrupted,
        int(slot_duration_us),int(difs_duration_us),
    ],device=device)
    return {
        "first_tx_us":first.numpy(),
        "interruption_count":interrupted.numpy(),
        "scope":"single_source_external_busy_schedule_not_full_mac",
    }
