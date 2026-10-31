"""Device state shared by motion, sensing, packet and collector kernels."""
import numpy as np
import warp as wp

from ..observations import observation_dim, state_dim
from ..scenario_recovery import SCENE_ATTEMPTS, certified_scenes
from ...communication.diagnostics import COUNTERS, STATS
from ...communication.link_budget import report_reference_tx_energy_j


@wp.struct
class WorldParams:
    xyz: wp.int32
    continuous_dim: wp.int32
    grid_n: wp.int32
    patch_n: wp.int32
    coarse_n: wp.int32
    altitude_min: wp.float64
    altitude_max: wp.float64
    obstacle_min_height: wp.float64
    obstacle_max_height: wp.float64
    cell_width: wp.float64
    fov_tangent: wp.float64
    belief_prior: wp.float64
    fine_altitude: wp.float64
    confirmation_threshold: wp.float64
    verified_empty_belief: wp.float64
    investigation_threshold: wp.float64
    sensor_anchors: wp.vec4d
    sensor_pd: wp.vec4d
    sensor_pf: wp.vec4d
    n: wp.int32
    targets_count: wp.int32
    obstacles_count: wp.int32
    max_steps: wp.int32
    motion_safety: wp.int32
    occlusion: wp.int32
    target_slots: wp.int32
    obstacle_slots: wp.int32
    coverage_grid_cells: wp.int32
    report_bytes: wp.int32
    buffer_bytes: wp.int32
    admission_mask: wp.int32
    recovery_enabled: wp.int32
    energy_enabled: wp.int32
    placement_attempts: wp.int32
    scene_attempts: wp.int32
    fallback_count: wp.int32
    polygon_sides: wp.int32
    network_enabled: wp.int32
    max_queue_packets: wp.int32
    payload_bytes: wp.int32
    max_retries: wp.int32
    ack_uses_data_power: wp.int32
    propagation_delay_enabled: wp.int32
    tx_power_mapping_log: wp.int32
    guided_power: wp.int32
    power_limited_admission: wp.int32
    link_state_observation: wp.int32
    cw_min: wp.int32
    event_capacity: wp.int32
    packet_capacity: wp.int32
    frame_capacity: wp.int32
    fragments: wp.int32
    session_capacity: wp.int32
    trace_capacity: wp.int32
    trace_enabled: wp.int32
    coalesce_idle_polls: wp.int32
    map_size: wp.float64
    dt: wp.float64
    max_speed: wp.float64
    max_accel: wp.float64
    safety_distance: wp.float64
    obstacle_clearance: wp.float64
    altitude: wp.float64
    footprint: wp.float64
    peer_range: wp.float64
    gcs_range: wp.float64
    report_ttl: wp.float64
    capacity: wp.float64
    idle_power: wp.float64
    hover_power: wp.float64
    speed_coefficient: wp.float64
    accel_coefficient: wp.float64
    launch_radius: wp.float64
    launch_spacing: wp.float64
    target_exclusion: wp.float64
    obstacle_min_radius: wp.float64
    obstacle_max_radius: wp.float64
    gcs_x: wp.float64
    gcs_y: wp.float64
    tx_power_min: wp.float64
    tx_power_max: wp.float64
    power_margin_factor: wp.float64
    pa_efficiency: wp.float64
    tx_circuit_power: wp.float64
    rx_circuit_power: wp.float64
    packet_lifetime: wp.float64
    queue_poll: wp.float64
    bit_rate: wp.float64
    frequency: wp.float64
    noise_power: wp.float64
    sinr_threshold: wp.float64
    cca_threshold: wp.float64
    nlos_exponent: wp.float64
    nlos_excess_loss_db: wp.float64
    nlos_scale: wp.float64
    slot: wp.float64
    sifs: wp.float64
    difs: wp.float64
    ack_bits: wp.float64
    header_bits: wp.float64
    ack_timeout: wp.float64
    retention: wp.float64
    reward_discovery: wp.float64
    reward_delivery: wp.float64
    reward_blocked: wp.float64
    reward_expired: wp.float64
    reward_dropped: wp.float64
    reward_step: wp.float64
    reward_communication: wp.float64
    reward_propulsion: wp.float64
    reward_communication_efficiency: wp.float64
    reward_shield_correction: wp.float64
    reward_information_gain: wp.float64
    reward_investigation: wp.float64
    reward_shaping_gamma: wp.float64
    report_reference_tx_energy: wp.float64


@wp.struct
class WorldState:
    altitudes: wp.array2d(dtype=wp.float64)
    vertical_velocity: wp.array2d(dtype=wp.float64)
    obstacle_heights: wp.array2d(dtype=wp.float64)
    belief: wp.array3d(dtype=wp.float64)
    belief_coarse: wp.array3d(dtype=wp.float64)
    belief_potential: wp.array(dtype=wp.float64)
    visited: wp.array3d(dtype=wp.int32)
    positive: wp.array3d(dtype=wp.int32)
    sensor_tick: wp.array(dtype=wp.int32)
    sensor_counts: wp.array2d(dtype=wp.int64)
    xyz_work: wp.array3d(dtype=wp.vec3d)
    fallback_positions: wp.array2d(dtype=wp.vec2d)
    fallback_targets: wp.array2d(dtype=wp.vec2d)
    fallback_obstacles: wp.array2d(dtype=wp.vec3d)
    generation_fallback: wp.array(dtype=wp.int32)
    generation_attempts: wp.array(dtype=wp.int32)
    diagnostic_counts: wp.array2d(dtype=wp.int64)
    diagnostic_stats: wp.array3d(dtype=wp.float64)
    power_histogram: wp.array3d(dtype=wp.int64)
    delivery_hop_histogram: wp.array2d(dtype=wp.int64)
    first_discovery: wp.array2d(dtype=wp.float64)
    first_delivery: wp.array2d(dtype=wp.float64)
    buffer_peak: wp.array2d(dtype=wp.int64)
    positions: wp.array2d(dtype=wp.vec2d)
    velocities: wp.array2d(dtype=wp.vec2d)
    targets: wp.array2d(dtype=wp.vec2d)
    obstacles: wp.array2d(dtype=wp.vec3d)
    gcs: wp.array(dtype=wp.vec2d)
    battery: wp.array2d(dtype=wp.float64)
    flight_energy: wp.array2d(dtype=wp.float64)
    inflight: wp.array2d(dtype=wp.float64)
    communication_energy: wp.array2d(dtype=wp.float64)
    tx_energy: wp.array2d(dtype=wp.float64)
    rx_energy: wp.array2d(dtype=wp.float64)
    visible: wp.array3d(dtype=wp.int32)
    known: wp.array3d(dtype=wp.int32)
    coverage: wp.array3d(dtype=wp.int32)
    recovery: wp.array3d(dtype=wp.int32)
    report_present: wp.array3d(dtype=wp.int32)
    report_origin: wp.array3d(dtype=wp.int32)
    report_hops: wp.array3d(dtype=wp.int32)
    report_created: wp.array3d(dtype=wp.float64)
    report_order: wp.array3d(dtype=wp.int64)
    delivered: wp.array2d(dtype=wp.int32)
    motion_counts: wp.array2d(dtype=wp.int32)
    report_counts: wp.array2d(dtype=wp.int32)
    steps: wp.array(dtype=wp.int32)
    terminated: wp.array(dtype=wp.int32)
    blocked: wp.array(dtype=wp.int32)
    clipped: wp.array(dtype=wp.int32)
    errors: wp.array(dtype=wp.int32)
    observation: wp.array3d(dtype=wp.float32)
    state: wp.array2d(dtype=wp.float32)
    reward: wp.array(dtype=wp.float32)
    pcg_state: wp.array2d(dtype=wp.uint64)
    seeds: wp.array(dtype=wp.int64)
    motion_work: wp.array3d(dtype=wp.vec2d)
    motion_flags: wp.array3d(dtype=wp.int32)
    coalesce_step: wp.array(dtype=wp.int32)
    node_queue_count: wp.array2d(dtype=wp.int32)
    event_count: wp.array(dtype=wp.int32)
    frame_count: wp.array(dtype=wp.int32)
    trace_count: wp.array(dtype=wp.int32)
    report_delivery_count: wp.array(dtype=wp.int32)
    event_sequence: wp.array(dtype=wp.int64)
    packet_sequence: wp.array(dtype=wp.int64)
    report_sequence: wp.array(dtype=wp.int64)
    packet_time: wp.array(dtype=wp.float64)
    mission_time: wp.array(dtype=wp.float64)
    packet_delay_sum: wp.array(dtype=wp.float64)
    report_delay_sum: wp.array(dtype=wp.float64)
    report_hop_sum: wp.array(dtype=wp.float64)
    before: wp.array2d(dtype=wp.float64)
    reward_components: wp.array2d(dtype=wp.float64)
    shield_correction_step: wp.array(dtype=wp.float64)
    shield_correction: wp.array(dtype=wp.float64)
    event_time: wp.array2d(dtype=wp.float64)
    pkt_created: wp.array2d(dtype=wp.float64)
    pkt_deadline: wp.array2d(dtype=wp.float64)
    pkt_power: wp.array2d(dtype=wp.float64)
    pkt_report_created: wp.array2d(dtype=wp.float64)
    pkt_backoff: wp.array2d(dtype=wp.float64)
    pkt_wait_start: wp.array2d(dtype=wp.float64)
    pkt_first_attempt: wp.array2d(dtype=wp.float64)
    tx_start: wp.array2d(dtype=wp.float64)
    tx_end: wp.array2d(dtype=wp.float64)
    tx_power: wp.array2d(dtype=wp.float64)
    tx_created: wp.array2d(dtype=wp.float64)
    tx_report_created: wp.array2d(dtype=wp.float64)
    tx_propagation_delay: wp.array3d(dtype=wp.float64)
    pair_gain: wp.array2d(dtype=wp.float64)
    pair_time: wp.array2d(dtype=wp.float64)
    trace_time: wp.array2d(dtype=wp.float64)
    trace_report: wp.array3d(dtype=wp.float64)  # target, origin, created, hops, power, source/dest XY
    packet_counts: wp.array2d(dtype=wp.int32)
    selection_counts: wp.array2d(dtype=wp.int32)
    event_kind: wp.array2d(dtype=wp.int32)
    event_arg: wp.array2d(dtype=wp.int32)
    event_priority: wp.array2d(dtype=wp.int32)
    event_generation: wp.array2d(dtype=wp.int32)
    pkt_status: wp.array2d(dtype=wp.int32)
    pkt_source: wp.array2d(dtype=wp.int32)
    pkt_destination: wp.array2d(dtype=wp.int32)
    pkt_target: wp.array2d(dtype=wp.int32)
    pkt_fragment: wp.array2d(dtype=wp.int32)
    pkt_attempt: wp.array2d(dtype=wp.int32)
    pkt_origin: wp.array2d(dtype=wp.int32)
    pkt_hops: wp.array2d(dtype=wp.int32)
    pkt_feed: wp.array2d(dtype=wp.int32)
    pkt_ttl: wp.array2d(dtype=wp.int32)
    pkt_generation: wp.array2d(dtype=wp.int32)
    pkt_backoff_version: wp.array2d(dtype=wp.int32)
    pkt_delivered: wp.array2d(dtype=wp.int32)
    node_current: wp.array2d(dtype=wp.int32)
    node_feed_alive: wp.array2d(dtype=wp.int32)
    node_wait_finish: wp.array2d(dtype=wp.int32)
    node_wait_packet: wp.array2d(dtype=wp.int32)
    node_receiving: wp.array2d(dtype=wp.int32)
    frame_order: wp.array2d(dtype=wp.int32)
    tx_active: wp.array2d(dtype=wp.int32)
    tx_kind: wp.array2d(dtype=wp.int32)
    tx_packet: wp.array2d(dtype=wp.int32)
    tx_generation: wp.array2d(dtype=wp.int32)
    tx_source: wp.array2d(dtype=wp.int32)
    tx_destination: wp.array2d(dtype=wp.int32)
    tx_target: wp.array2d(dtype=wp.int32)
    tx_fragment: wp.array2d(dtype=wp.int32)
    tx_payload: wp.array2d(dtype=wp.int32)
    tx_ttl: wp.array2d(dtype=wp.int32)
    tx_origin: wp.array2d(dtype=wp.int32)
    tx_hops: wp.array2d(dtype=wp.int32)
    tx_receive: wp.array2d(dtype=wp.int32)
    session_active: wp.array2d(dtype=wp.int32)
    session_origin: wp.array2d(dtype=wp.int32)
    session_hops: wp.array2d(dtype=wp.int32)
    trace_source: wp.array2d(dtype=wp.int32)
    trace_destination: wp.array2d(dtype=wp.int32)
    trace_kind: wp.array2d(dtype=wp.int32)
    action_counts: wp.array2d(dtype=wp.int64)
    event_order: wp.array2d(dtype=wp.int64)
    pkt_id: wp.array2d(dtype=wp.int64)
    pkt_order: wp.array2d(dtype=wp.int64)
    pkt_session_id: wp.array2d(dtype=wp.int64)
    pkt_session: wp.array2d(dtype=wp.int64)
    tx_id: wp.array2d(dtype=wp.int64)
    tx_session: wp.array2d(dtype=wp.int64)
    tx_receive_order: wp.array2d(dtype=wp.int64)
    session_id: wp.array2d(dtype=wp.int64)
    session_window_start: wp.array2d(dtype=wp.int32)
    session_arrived_mask: wp.array2d(dtype=wp.uint64)
    session_pending_mask: wp.array2d(dtype=wp.uint64)
    session_created: wp.array2d(dtype=wp.float64)
    session_power: wp.array2d(dtype=wp.float64)
    pair_transmitter: wp.array2d(dtype=wp.vec2d)
    pair_receiver: wp.array2d(dtype=wp.vec2d)
    mt: wp.array3d(dtype=wp.uint32)
    mt_index: wp.array2d(dtype=wp.int32)


def params_from_config(config):
    config.validate()
    w, c, n, en, r = config.world, config.sensor, config.network, config.energy, config.reward
    if n.max_retries > 26:
        raise NotImplementedError("warp_packet supports max_retries<=26; larger retries need multiword getrandbits")
    if n.max_queue_packets > 64:
        raise NotImplementedError("warp_packet streaming reports support max_queue_packets<=64")
    nlos_exponent, nlos_excess_db = {"urban": (3.0, 12.0), "suburban": (2.6, 8.0)}[n.nlos_model]
    p = WorldParams()
    p.xyz = int(config.motion_dim == 3); p.continuous_dim = config.continuous_dim
    p.grid_n = int(np.ceil(w.map_size_m/c.grid_cell_m)); p.patch_n = c.belief_patch_cells; p.coarse_n = c.belief_coarse_cells
    p.altitude_min, p.altitude_max = w.altitude_min_m, w.altitude_max_m
    p.obstacle_min_height, p.obstacle_max_height = w.obstacle_height_min_m, w.obstacle_height_max_m
    p.cell_width, p.fov_tangent = c.grid_cell_m, np.tan(np.deg2rad(c.full_fov_deg/2))
    p.belief_prior, p.fine_altitude, p.confirmation_threshold = c.belief_prior, c.fine_altitude_m, c.confirmation_threshold
    p.verified_empty_belief, p.investigation_threshold = c.verified_empty_belief, r.investigation_belief_threshold
    p.sensor_anchors, p.sensor_pd, p.sensor_pf = wp.vec4d(*c.altitude_anchors), wp.vec4d(*c.pd), wp.vec4d(*c.pf)
    values = dict(n=w.num_uavs, targets_count=w.num_targets, obstacles_count=w.num_obstacles,
                  max_steps=w.max_steps, motion_safety=int(w.motion_safety == "dynamic_window"),
                  occlusion=int(c.obstacle_occlusion), target_slots=c.visible_target_slots,
                  obstacle_slots=c.visible_obstacle_slots, report_bytes=n.report_bytes,
                  coverage_grid_cells=c.coverage_grid_cells,
                  buffer_bytes=n.buffer_bytes, admission_mask=int(n.receiver_admission_mask),
                  recovery_enabled=int(n.rediscovery_recovery), energy_enabled=int(en.enabled),
                  network_enabled=int(n.enabled), placement_attempts=w.placement_attempts,
                  scene_attempts=SCENE_ATTEMPTS,
                  polygon_sides=n.obstacle_polygon_sides, max_queue_packets=n.max_queue_packets,
                  payload_bytes=n.payload_bytes, max_retries=n.max_retries,
                  ack_uses_data_power=int(n.ack_uses_data_power),
                  propagation_delay_enabled=int(n.propagation_delay_enabled),
                  tx_power_mapping_log=int(n.tx_power_mapping == "log"),
                  guided_power=int(n.power_control == "link_budget"),
                  power_margin_factor=10.0 ** (n.link_budget_margin_db / 10.0),
                  power_limited_admission=int(n.power_limited_admission),
                  link_state_observation=int(n.link_state_observation), cw_min=31,
                  coalesce_idle_polls=int(config.runtime.coalesce_idle_polls),
                  map_size=w.map_size_m, dt=w.dt_s, max_speed=w.max_speed_mps,
                  max_accel=w.max_accel_mps2, safety_distance=w.safety_distance_m,
                  obstacle_clearance=w.obstacle_clearance_m, altitude=c.altitude_m,
                  footprint=c.footprint_radius_m, peer_range=n.peer_range_m, gcs_range=n.gcs_range_m,
                  report_ttl=n.report_ttl_s, capacity=en.battery_capacity_j,
                  idle_power=en.idle_power_w, hover_power=en.hover_power_w,
                  speed_coefficient=en.speed_squared_coefficient,
                  accel_coefficient=en.acceleration_squared_coefficient,
                  launch_radius=w.launch_radius_m, launch_spacing=w.launch_min_spacing_m,
                  target_exclusion=w.target_exclusion_radius_m,
                  obstacle_min_radius=w.obstacle_radius_min_m, obstacle_max_radius=w.obstacle_radius_max_m,
                  tx_power_min=n.tx_power_min_w, tx_power_max=n.tx_power_max_w,
                  pa_efficiency=n.pa_efficiency, tx_circuit_power=n.tx_circuit_power_w,
                  rx_circuit_power=n.rx_circuit_power_w,
                  packet_lifetime=n.packet_lifetime_s, queue_poll=n.queue_poll_us,
                  bit_rate=n.bit_rate_bps, frequency=n.carrier_frequency_hz,
                  noise_power=10 ** ((n.noise_density_dbm_hz + 10*np.log10(n.bandwidth_hz)
                                     + n.noise_figure_db - 30)/10),
                  sinr_threshold=n.sinr_threshold_db, cca_threshold=n.cca_threshold_dbm,
                  nlos_exponent=nlos_exponent, nlos_excess_loss_db=nlos_excess_db,
                  nlos_scale=10 ** (-(nlos_excess_db + n.obstacle_extra_loss_db)/10),
                  slot=20., sifs=10., difs=50., ack_bits=240., header_bits=464.,
                  reward_discovery=r.discovery, reward_delivery=r.delivery,
                  reward_blocked=r.blocked_motion, reward_expired=r.expired_report,
                  reward_dropped=r.dropped_report, reward_step=r.step_cost,
                  reward_communication=r.communication_energy, reward_propulsion=r.propulsion_energy_per_kj,
                  reward_communication_efficiency=r.communication_efficiency,
                  reward_shield_correction=r.shield_correction,
                  reward_information_gain=r.information_gain, reward_investigation=r.investigation_shaping,
                  reward_shaping_gamma=r.shaping_gamma,
                  report_reference_tx_energy=report_reference_tx_energy_j(n))
    gcs = w.gcs_position_m or [w.map_size_m/2, 0.]
    ack_timeout = 240/n.bit_rate_bps*1e6+60
    if n.propagation_delay_enabled:
        max_distance = np.sqrt(2*w.map_size_m**2+(w.altitude_max_m if config.motion_dim==3 else c.altitude_m)**2)
        ack_timeout += 2*max_distance/299_792_458.0*1e6
    values.update(gcs_x=gcs[0], gcs_y=gcs[1], ack_timeout=ack_timeout,
                  retention=2*(n.payload_bytes*8+192)/n.bit_rate_bps*1e6,
                  fragments=(n.report_bytes+n.payload_bytes-1)//n.payload_bytes)
    values.update(packet_capacity=w.num_uavs*(n.max_queue_packets+2),
                  event_capacity=w.num_uavs*(n.max_queue_packets+8)*8,
                  frame_capacity=w.num_uavs*(n.max_queue_packets+4)*4,
                  session_capacity=w.num_uavs*w.num_targets*(w.num_uavs+1),
                  trace_capacity=w.num_uavs*(n.max_queue_packets+2)*(n.max_retries*2+1),
                  trace_enabled=1)
    for name, value in values.items():
        setattr(p, name, value)
    return p


def seed_rows(seeds):
    """NumPy PCG64's 128-bit state for the geometry kernel."""
    rows = []
    for seed in seeds:
        d = np.random.PCG64(int(seed)).state["state"]
        rows.append([d["state"] & ((1<<64)-1), d["state"] >> 64,
                     d["inc"] & ((1<<64)-1), d["inc"] >> 64])
    return np.asarray(rows, dtype=np.uint64)


def allocate_state(config, num_envs=1, seeds=None, device="cpu", trace=False, certify=False):
    """Allocate one independent world per row; no native simulator or worker."""
    p = params_from_config(config)
    bank = certified_scenes(config) if certify else ()
    p.fallback_count = len(bank)
    if not trace:
        p.trace_enabled = 0
        p.trace_capacity = 1
    e, n, t, o = num_envs, p.n, p.targets_count, max(1, p.obstacles_count)
    if seeds is None:
        seeds = np.arange(config.runtime.seed, config.runtime.seed+e, dtype=np.int64)
    seeds = np.asarray(seeds, dtype=np.int64)
    if seeds.shape != (e,):
        raise ValueError("seeds must contain one integer per world")
    s = WorldState()
    for name, dtype in (("positions", wp.vec2d), ("targets", wp.vec2d), ("obstacles", wp.vec3d)):
        count = {"positions": n, "targets": t, "obstacles": o}[name]
        width = 3 if name == "obstacles" else 2
        values = (np.stack([getattr(scene, name) for scene in bank]) if bank
                  else np.zeros((1, count, width)))
        if name == "obstacles" and p.obstacles_count == 0:
            values = np.zeros((max(1, len(bank)), 1, 3))
        setattr(s, "fallback_" + name, wp.array(values, dtype=dtype, device=device))
    shapes = {
        "coverage": ((e,n,config.sensor.coverage_grid_cells**2), wp.int32),
        "generation_fallback": ((e,), wp.int32),
        "generation_attempts": ((e,), wp.int32),
        "diagnostic_counts": ((e,len(COUNTERS)), wp.int64),
        "diagnostic_stats": ((e,len(STATS),4), wp.float64),
        "power_histogram": ((e,2,6), wp.int64),
        "delivery_hop_histogram": ((e,8), wp.int64),
        "first_discovery": ((e,t), wp.float64), "first_delivery": ((e,t), wp.float64),
        "buffer_peak": ((e,n), wp.int64),
        "positions": ((e,n), wp.vec2d), "velocities": ((e,n), wp.vec2d),
        "targets": ((e,t), wp.vec2d), "obstacles": ((e,o), wp.vec3d),
        "gcs": ((e,), wp.vec2d), "battery": ((e,n), wp.float64),
        "flight_energy": ((e,n), wp.float64), "delivered": ((e,t), wp.int32),
        "motion_counts": ((e,9), wp.int32), "report_counts": ((e,3), wp.int32),
        "inflight": ((e,n), wp.float64), "communication_energy": ((e,n+1), wp.float64),
        "tx_energy": ((e,n+1), wp.float64), "rx_energy": ((e,n+1), wp.float64),
        "observation": ((e,n,observation_dim(config)), wp.float32),
        "state": ((e,state_dim(config)), wp.float32), "reward": ((e,), wp.float32),
        "pcg_state": ((e,4), wp.uint64), "seeds": ((e,), wp.int64),
        "motion_work": ((e,n,3), wp.vec2d), "motion_flags": ((e,n,5), wp.int32),
        "before": ((e,8), wp.float64), "reward_components": ((e,12), wp.float64),
        "shield_correction_step": ((e,), wp.float64), "shield_correction": ((e,), wp.float64),
        "packet_counts": ((e,8), wp.int32), "selection_counts": ((e,5), wp.int32),
        "action_counts": ((e,4), wp.int64), "mt": ((e,n+1,624), wp.uint32),
        "mt_index": ((e,n+1), wp.int32)}
    xn, xo, xg = (n,o,p.grid_n*p.grid_n) if p.xyz else (0,0,0)
    shapes.update(altitudes=((e,xn),wp.float64), vertical_velocity=((e,xn),wp.float64),
        obstacle_heights=((e,xo),wp.float64), belief=((e,xn,xg),wp.float64),
        belief_coarse=((e,xn,config.sensor.belief_coarse_cells**2),wp.float64),
        belief_potential=((e,),wp.float64), visited=((e,xn,xg),wp.int32), positive=((e,xn,xg),wp.int32),
        sensor_tick=((e,),wp.int32), sensor_counts=((e,6 if p.xyz else 0),wp.int64),
        xyz_work=((e,xn,3),wp.vec3d))
    for name in ("visible", "known", "recovery", "report_present", "report_origin", "report_hops"):
        shapes[name] = ((e,n,t), wp.int32)
    shapes["report_created"] = ((e,n,t), wp.float64)
    shapes["trace_report"] = ((e,p.trace_capacity if trace else 0,9), wp.float64)
    shapes["tx_propagation_delay"] = ((e,p.frame_capacity,n+1) if p.propagation_delay_enabled else (e,0,n+1), wp.float64)
    shapes["report_order"] = ((e,n,t), wp.int64)
    for name in ("steps","terminated","blocked","clipped","errors","event_count","frame_count",
                 "trace_count","report_delivery_count","coalesce_step"):
        shapes[name] = ((e,), wp.int32)
    for name in ("event_sequence","packet_sequence","report_sequence"):
        shapes[name] = ((e,), wp.int64)
    for name in ("packet_time","mission_time","packet_delay_sum","report_delay_sum","report_hop_sum"):
        shapes[name] = ((e,), wp.float64)
    groups = (
        (p.event_capacity, wp.float64, "event_time"),
        (p.event_capacity, wp.int32, "event_kind event_arg event_priority event_generation"),
        (p.event_capacity, wp.int64, "event_order"),
        (p.packet_capacity, wp.float64, "pkt_created pkt_deadline pkt_power pkt_report_created pkt_backoff pkt_wait_start pkt_first_attempt"),
        (p.packet_capacity, wp.int32, "pkt_status pkt_source pkt_destination pkt_target pkt_fragment pkt_attempt pkt_origin pkt_hops pkt_feed pkt_ttl pkt_generation pkt_backoff_version pkt_delivered"),
        (p.packet_capacity, wp.int64, "pkt_id pkt_order pkt_session pkt_session_id"),
        (n+1, wp.int32, "node_queue_count node_current node_feed_alive node_wait_finish node_wait_packet node_receiving"),
        (p.frame_capacity, wp.int32, "frame_order tx_active tx_kind tx_packet tx_generation tx_source tx_destination tx_target tx_fragment tx_payload tx_ttl tx_origin tx_hops tx_receive"),
        (p.frame_capacity, wp.float64, "tx_start tx_end tx_power tx_created tx_report_created"),
        (p.frame_capacity, wp.int64, "tx_id tx_session tx_receive_order"),
        ((n+1)*(n+1), wp.float64, "pair_gain pair_time"),
        ((n+1)*(n+1), wp.vec2d, "pair_transmitter pair_receiver"),
        (p.session_capacity, wp.int32, "session_active session_origin session_hops session_window_start"),
        (p.session_capacity, wp.float64, "session_created session_power"),
        (p.session_capacity, wp.int64, "session_id"),
        (p.session_capacity, wp.uint64, "session_arrived_mask session_pending_mask"),
        (p.trace_capacity, wp.float64, "trace_time"),
        (p.trace_capacity, wp.int32, "trace_source trace_destination trace_kind"))
    for length, dtype, names in groups:
        for name in names.split():
            shapes[name] = ((e,length), dtype)
    for name, (shape, dtype) in shapes.items():
        setattr(s, name, wp.zeros(shape, dtype=dtype, device=device))
    s.seeds.assign(seeds)
    s.pcg_state.assign(seed_rows(seeds))
    s.gcs.assign(np.tile(gcs := [p.gcs_x,p.gcs_y], (e,1)))
    s.battery.fill_(p.capacity)
    return s, p


def allocated_bytes(s):
    return sum(value.capacity for value in vars(s).values() if isinstance(value, wp.array))
