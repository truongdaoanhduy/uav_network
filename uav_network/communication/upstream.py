"""Small compatibility boundary for the pinned external packet simulator."""
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
import json
import logging
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sys

import numpy as np

from .routing import SelectedNextHopRouting

REVISION = "04daafb815eb377409b40b285574eeb62b9a8d58"
ROOT = Path(__file__).resolve().parents[2]


def load_upstream(config):
    root = Path(config.network.upstream_path).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    if not (root / "simulator/simulator.py").is_file():
        try:
            installed = distribution("uavnetsim")
            origin = json.loads(installed.read_text("direct_url.json") or "{}")
            if origin.get("vcs_info", {}).get("commit_id") != REVISION:
                raise FileNotFoundError("Installed UavNetSim is not the pinned revision")
            root = Path(installed.locate_file(""))
        except PackageNotFoundError as error:
            raise FileNotFoundError("UavNetSim is missing. Run bash setup.sh first.") from error
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    logger = logging.getLogger()
    handler = logging.NullHandler() if not logger.handlers else None
    if handler:
        logger.addHandler(handler)
    try:
        module = import_module("simulator.simulator")
        native_config = import_module("utils.config")
        packets = import_module("entities.packet")
        simpy = import_module("simpy")
    finally:
        if handler:
            logger.removeHandler(handler)
    if root not in Path(module.__file__).resolve().parents:
        raise RuntimeError("A different top-level simulator package is already imported")
    return module, native_config, packets.DataPacket, simpy


def scene_payload(scene, config):
    sides = config.network.obstacle_polygon_sides
    features = []
    for i, (x, y, radius) in enumerate(scene.obstacles):
        features.append({"id": f"obstacle-{i}", "category": "building",
                         "height": config.sensor.altitude_m + 20,
                         "material": "itu_concrete", "source": "uav_network",
                         "footprint": [{"x": x + radius * np.cos(a),
                                        "y": y + radius * np.sin(a), "z": 0.0}
                                       for a in np.linspace(0, 2 * np.pi, sides, endpoint=False)]})
    return {"schema_version": 1, "name": "uav-swarm-2d", "anchor": {"latitude": 0, "longitude": 0},
            "size_x": config.world.map_size_m, "size_y": config.world.map_size_m,
            "features": features}


def configure_native(native, config, scene_path):
    w, n = config.world, config.network
    values = dict(NUMBER_OF_DRONES=w.num_uavs + 1, MAX_TTL=w.num_uavs + 2,
                  ROUTING_PROTOCOL="DRL", DRL_ROUTING_PROTOCOL_CLASS=SelectedNextHopRouting,
                  MAC_PROTOCOL="CSMA_CA", STATIC_CASE=1, HETEROGENEOUS=0,
                  MOBILITY_MODEL="GaussMarkov3D", TRAFFIC_PATTERN="UNIFORM",
                  PACKET_ARRIVAL_RATE=1e-12, MAP_LENGTH=w.map_size_m, MAP_WIDTH=w.map_size_m,
                  MAP_HEIGHT=config.sensor.altitude_m + 20, UAV_MIN_ALTITUDE=0,
                  UAV_MAX_ALTITUDE=config.sensor.altitude_m + 20,
                  UAV_BUILDING_CLEARANCE=0, UAV_BOUNDARY_CLEARANCE=0,
                  SIONNA_SCENE_PATH=str(scene_path / "scene.xml"), CHANNEL_MODE="a2a",
                  LOS_A2A_MODEL="free_space", NLOS_A2A_MODEL=n.nlos_model,
                  MAX_QUEUE_SIZE=n.max_queue_packets, AVERAGE_PAYLOAD_LENGTH=n.payload_bytes * 8,
                  VARIABLE_PAYLOAD_LENGTH=0, PACKET_LIFETIME=n.packet_lifetime_s * 1e6,
                  INITIAL_ENERGY=1e9, ENERGY_THRESHOLD=0,
                  TRANSMITTING_POWER=n.tx_power_min_w, CARRIER_FREQUENCY=n.carrier_frequency_hz,
                  BANDWIDTH=n.bandwidth_hz, BIT_RATE=n.bit_rate_bps,
                  SINR_THRESHOLD_DB=n.sinr_threshold_db, CCA_THRESHOLD_DBM=n.cca_threshold_dbm,
                  THERMAL_NOISE_DENSITY_DBM_HZ=n.noise_density_dbm_hz,
                  RECEIVER_NOISE_FIGURE_DB=n.noise_figure_db, MAX_RETRANSMISSION_ATTEMPT=n.max_retries)
    values["BIT_TRANSMISSION_TIME"] = 1e6 / n.bit_rate_bps
    values["ACK_TIMEOUT"] = native.ACK_PACKET_LENGTH / n.bit_rate_bps * 1e6 + native.SIFS_DURATION + 50
    previous = {name: getattr(native, name) for name in values}
    for name, value in values.items():
        setattr(native, name, value)
    return previous


def mission_drone_class(original, native, config):
    """Keep upstream MAC/PHY, receiver and queue resource; own traffic/motion."""
    class MissionDrone(original):
        def _create_mobility_model(self):
            return SimpleNamespace()  # XY motion is owned by the mission environment.

        def generate_data_packet(self):
            yield self.env.timeout(self.simulator.total_simulation_time + 1)

        def feed_packet(self):
            while not self.sleep:
                interval = config.network.queue_poll_us
                delay = interval
                # Mission traffic enters only before each clock.run; native ACK
                # retries call packet_coming directly, never enqueue new data.
                # Skip empty ticks, keeping the final pre-boundary tick so the
                # next poll retains its phase and normal boundary scheduling.
                # Integral clocks avoid changing repeated-addition rounding.
                if (self.simulator.mission_bridge.coalesce_empty_step and self.transmitting_queue.empty()
                        and float(interval).is_integer() and float(self.env.now).is_integer()):
                    remaining = self.simulator.mission_bridge.step_end_us - self.env.now
                    delay *= max(1, math.ceil(remaining / interval) - 1)
                yield self.env.timeout(delay)
                if self.blocking() or self.transmitting_queue.empty():
                    continue
                packet = self.transmitting_queue.get()
                if (self.env.now >= packet.creation_time + packet.deadline
                        or packet.number_retransmission_attempt[self.identifier] >= native.MAX_RETRANSMISSION_ATTEMPT):
                    self.simulator.event_bus.publish("packet_dropped", self.env.now,
                                                     packet_id=packet.packet_id, node=self.identifier,
                                                     reason="deadline_or_retry_limit")
                    continue
                has_route, packet, _ = self.routing_protocol.next_hop_selection(packet)
                if has_route:
                    yield self.env.process(self.packet_coming(packet))
    return MissionDrone


def start_simulator(bridge, scene, seed):
    module, native, data_packet, simpy = load_upstream(bridge.config)
    directory = TemporaryDirectory(prefix="uav-network-scene-")
    path = Path(directory.name)
    (path / "scene.json").write_text(json.dumps(scene_payload(scene, bridge.config)))
    previous = configure_native(native, bridge.config, path)
    original = module.Drone
    module.Drone = mission_drone_class(original, native, bridge.config)
    try:
        clock = simpy.Environment()
        total = (bridge.config.world.max_steps + 2) * bridge.config.world.dt_s * 1e6
        simulator = module.Simulator(seed=seed, env=clock, n_drones=bridge.config.world.num_uavs + 1,
                                     total_simulation_time=total, drone_speed=0.0)
        simulator.mission_bridge = bridge
    except Exception:
        for name, value in previous.items():
            setattr(native, name, value)
        directory.cleanup()
        raise
    finally:
        module.Drone = original
    return simulator, clock, native, data_packet, directory, previous
