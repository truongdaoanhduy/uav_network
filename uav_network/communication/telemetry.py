"""Passive hooks into the pinned native simulator's actual decisions."""
import math
import numpy as np
from .diagnostics import COUNT, STAT, POWER_EDGES, add_sample
from .link_budget import nominal_required_power_w


def install_telemetry(bridge):
    channel = bridge.simulator.channel
    counts, stats = bridge.diagnostic_counts, bridge.diagnostic_stats
    original_transmit, original_evaluate, original_busy = channel.transmit, channel._evaluate, channel.is_busy_for

    def transmit(packet, source, receivers):
        data = isinstance(packet, bridge.DataPacket)
        kind = "data" if data else "ack"
        power = getattr(packet, "mission_power_w", bridge.config.network.tx_power_min_w)
        counts[COUNT[kind + "_attempts"]] += 1
        add_sample(stats, STAT[kind + "_tx_power_w"], power)
        bridge.power_histogram[int(not data), np.searchsorted(POWER_EDGES, power, side="right")] += 1
        if data:
            counts[COUNT["data_retries"]] += int(packet.number_retransmission_attempt[source] > 1)
            destination = packet.dst_drone.identifier
            link = "gcs" if destination == bridge.config.world.num_uavs else "peer"
            add_sample(stats, STAT["data_" + link + "_tx_power_w"], power)
            src_xy = bridge.simulator.drones[source].coords[:2]
            dst_xy = bridge.simulator.drones[destination].coords[:2]
            required = nominal_required_power_w(src_xy, dst_xy, bridge.scene.obstacles, bridge.config.network,
                                                bridge.config.sensor.altitude_m, destination == bridge.config.world.num_uavs)
            add_sample(stats, STAT["data_required_power_w"], required)
            margin = 10 * math.log10(power / required)
            add_sample(stats, STAT["data_power_margin_db"], margin)
            counts[COUNT["data_below_required_power_attempts"]] += int(power < required)
        return original_transmit(packet, source, receivers)

    def evaluate(target, receiver):
        result = original_evaluate(target, receiver)
        sinr, rssi, interferers = result
        data = isinstance(target.packet, bridge.DataPacket)
        success = sinr >= bridge.config.network.sinr_threshold_db
        kind = "data" if data else "ack"
        counts[COUNT[kind + ("_phy_successes" if success else "_phy_failures")]] += 1
        if data:
            snr = rssi - 10 * math.log10(bridge.native.noise_power_watt() * 1000)
            drones = bridge.simulator.drones
            pair = (target.transmitter_id, receiver)
            # The decode just populated this exact cached estimate; never query the PHY again.
            line_of_sight = channel.channel_model._links[pair].estimate.line_of_sight
            counts[COUNT["data_decode_count"]] += 1
            counts[COUNT["data_overlap_decodes"]] += int(bool(interferers))
            counts[COUNT["data_collision_failures"]] += int(not success and bool(interferers))
            for name in ("los" if line_of_sight else "nlos",
                         "gcs" if receiver == bridge.config.world.num_uavs else "peer"):
                counts[COUNT["data_" + name + "_decodes"]] += 1
                counts[COUNT["data_" + name + "_successes"]] += int(success)
            if not success:
                failure = "weak_signal" if snr < bridge.config.network.sinr_threshold_db else "interference"
                counts[COUNT["data_" + failure + "_failures"]] += 1
            distance = math.dist(drones[target.transmitter_id].coords, drones[receiver].coords)
            for name, value in (("data_snr_db", snr), ("data_sinr_db", sinr),
                                ("data_rssi_dbm", rssi), ("data_distance_m", distance)):
                add_sample(stats, STAT[name], value)
        return result

    def busy(drone, channel_id):
        value = original_busy(drone, channel_id)
        counts[COUNT["cca_checks"]] += 1
        counts[COUNT["cca_busy"]] += int(value)
        return value

    channel.transmit, channel._evaluate, channel.is_busy_for = transmit, evaluate, busy
    for drone in bridge.simulator.drones:
        mac = drone.mac_protocol
        original_send = mac.mac_send

        def send(packet, original_send=original_send):
            counts[COUNT["backoff_draws"]] += 1
            yield from original_send(packet)

        def listen(packet, drone=drone):
            key = f"mac_send{drone.identifier}_{packet.packet_id}"
            while drone.mac_process_finish[key] == 0:
                if channel.is_busy_for(drone, packet.channel_id):
                    process = drone.mac_process_dict[key]
                    if not process.triggered:
                        counts[COUNT["backoff_freezes"]] += 1
                        process.interrupt()
                        break
                yield bridge.clock.timeout(1)

        mac.mac_send, mac.listen = send, listen
