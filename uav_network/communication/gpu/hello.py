"""Opt-in contention-aware one-hop HELLO: real PHY reception, no remote-state oracle.

Single broadcast frame per source enters the SAME event heap and channel
interference/CCA record as data and ACK. Every peer decodes independently;
undecoded/dropped frames NEVER update the actor's neighbor cache.
"""
import warp as wp
from ...environment.gpu.state import WorldParams, WorldState
from .reports import active, bytes_used
from .phy import (busy, gain, arrival_start, arrival_end,
                  latest_arrival_end, link_distance, C)
from .events import push
from .rng import randbelow

wp.set_module_options({"enable_fp_fma": False, "enable_backward": False})

@wp.func
def hello_fresh(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int,i:int,j:int):
    age=wp.float64(s.steps[e])*p.dt-s.hello_rx_time[e,i,j]
    return (p.hello_enabled!=0 and s.hello_rx_time[e,i,j]>=wp.float64(0)
            and age>=wp.float64(0) and age<=p.hello_freshness)

@wp.func
def schedule_hello(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int,source:int,now:wp.float64):
    # Fresh contention window each period, independently of the data queue.
    push(s,p,e,now+p.difs+wp.float64(randbelow(s,e,source,p.cw_min+1))*p.slot,
         18,source,int(now),1)
    push(s,p,e,now+p.hello_period,17,source,0,1)

@wp.func
def emit_hello(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int,source:int,now:wp.float64):
    """Reserve ONE real RF frame; scheduled RX events share this waveform."""
    if not active(s,p,e,source):
        return
    power=p.tx_power_max
    duration=wp.float64(p.hello_bits)/p.bit_rate*wp.float64(1e6)
    tx_cost=duration/wp.float64(1e6)*(p.tx_circuit_power+power/p.pa_efficiency)
    if p.energy_enabled!=0 and tx_cost>s.battery[e,source]:
        return
    s.communication_energy[e,source]+=tx_cost
    s.tx_energy[e,source]+=tx_cost
    if p.energy_enabled!=0:
        s.battery[e,source]=wp.max(wp.float64(0),s.battery[e,source]-tx_cost)
    retained=int(0)
    for k in range(s.frame_count[e]):
        f=s.frame_order[e,k]
        if latest_arrival_end(s,p,e,f)>=now-p.retention:
            s.frame_order[e,retained]=f
            retained+=1
        else:
            s.tx_active[e,f]=0
            if s.tx_receive[e,f]==1 or s.tx_receive[e,f]==2:
                s.tx_active[e,f]=2
    s.frame_count[e]=retained
    frame=int(-1)
    for f in range(p.frame_capacity):
        if s.tx_active[e,f]==2 and s.tx_receive[e,f]==3:
            s.tx_active[e,f]=0
        if frame<0 and s.tx_active[e,f]==0:
            frame=f
    if frame<0:
        s.errors[e]=3
        return
    s.frame_order[e,s.frame_count[e]]=frame
    s.frame_count[e]+=1
    s.tx_active[e,frame]=1
    s.tx_kind[e,frame]=2
    s.tx_receive[e,frame]=3  # no ACK/inbox; per-peer PHY decode events
    s.tx_source[e,frame]=source
    s.tx_destination[e,frame]=p.n  # broadcast marker, not a destination oracle
    s.tx_power[e,frame]=power
    s.tx_start[e,frame]=now
    s.tx_end[e,frame]=now+duration
    if p.propagation_delay_enabled!=0:
        for receiver in range(p.n+1):
            s.tx_propagation_delay[e,frame,receiver]=link_distance(s,p,e,source,receiver)/C*wp.float64(1e6)
    # Frame payload is sampled only when the source transmits.
    s.tx_hello_position[e,source]=s.positions[e,source]
    s.tx_hello_velocity[e,source]=s.velocities[e,source]
    s.tx_hello_altitude[e,source]=s.altitudes[e,source]
    s.tx_hello_vertical_velocity[e,source]=s.vertical_velocity[e,source]
    s.tx_hello_battery[e,source]=s.battery[e,source]/p.capacity
    s.tx_hello_free_buffer[e,source]=wp.clamp(
        (wp.float64(p.buffer_bytes)-wp.float64(bytes_used(s,p,e,source)))/wp.float64(p.buffer_bytes),
        wp.float64(0),wp.float64(1))
    s.hello_tx_count[e]+=wp.int64(1)
    for receiver in range(p.n):
        if receiver!=source:
            push(s,p,e,arrival_end(s,p,e,frame,receiver),19,frame*p.n+receiver,0,1)

@wp.func
def receive_hello(s:wp.ref[WorldState],p:wp.ref[WorldParams],e:int,arg:int,now:wp.float64):
    frame=arg//p.n
    receiver=arg%p.n
    source=s.tx_source[e,frame]
    # RF/queue truth is used by the physical channel ONLY, never by the actor.
    if source==receiver or not active(s,p,e,receiver):
        return
    if link_distance(s,p,e,source,receiver)>p.peer_range:
        s.hello_rx_fail_count[e]+=wp.int64(1)
        return
    start=arrival_start(s,p,e,frame,receiver)
    end=arrival_end(s,p,e,frame,receiver)
    duration=wp.max(end-start,wp.float64(1e-9))
    signal=s.tx_power[e,frame]*gain(s,p,e,source,receiver,now)
    interference=wp.float64(0)
    transmitting=bool(False)
    for k in range(s.frame_count[e]):
        other=s.frame_order[e,k]
        if other!=frame:
            overlap=wp.max(wp.float64(0),wp.min(end,arrival_end(s,p,e,other,receiver))
                           -wp.max(start,arrival_start(s,p,e,other,receiver)))
            if overlap>wp.float64(0):
                interference+=s.tx_power[e,other]*gain(s,p,e,s.tx_source[e,other],receiver,now)*(overlap/duration)
                if s.tx_source[e,other]==receiver:
                    transmitting=True
    good=(not transmitting and signal>wp.float64(0)
          and wp.float64(10)*wp.log10(signal/(p.noise_power+interference))>=p.sinr_threshold)
    rx_cost=duration/wp.float64(1e6)*p.rx_circuit_power
    if p.energy_enabled!=0:
        if rx_cost>s.battery[e,receiver]:
            good=False
            rx_cost=s.battery[e,receiver]
        s.battery[e,receiver]=wp.max(wp.float64(0),s.battery[e,receiver]-rx_cost)
    s.communication_energy[e,receiver]+=rx_cost
    s.rx_energy[e,receiver]+=rx_cost
    if good:
        # Copy ONLY the payload as actually transmitted; no live remote read.
        s.hello_position[e,receiver,source]=s.tx_hello_position[e,source]
        s.hello_velocity[e,receiver,source]=s.tx_hello_velocity[e,source]
        s.hello_altitude[e,receiver,source]=s.tx_hello_altitude[e,source]
        s.hello_vertical_velocity[e,receiver,source]=s.tx_hello_vertical_velocity[e,source]
        s.hello_battery[e,receiver,source]=s.tx_hello_battery[e,source]
        s.hello_free_buffer[e,receiver,source]=s.tx_hello_free_buffer[e,source]
        s.hello_rx_time[e,receiver,source]=now/wp.float64(1e6)
        s.hello_rx_count[e]+=wp.int64(1)
    else:
        s.hello_rx_fail_count[e]+=wp.int64(1)
