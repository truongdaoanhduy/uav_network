"""Apply battery budgets and packet-selected power at the real native PHY."""
from dataclasses import replace
from .diagnostics import COUNT


def transmit_energy_j(bits, rf_power_w, network_config):
    """Physical transmitter energy: electronics plus RF output through the PA."""
    airtime = float(bits) / float(network_config.bit_rate_bps)
    return airtime * (float(network_config.tx_circuit_power_w)
                      + float(rf_power_w) / float(network_config.pa_efficiency))


def receive_energy_j(bits, network_config):
    """Receiver electronics energy for listening to one full frame."""
    return float(bits) / float(network_config.bit_rate_bps) * float(network_config.rx_circuit_power_w)


def _debit_radio(bridge, node, requested, bucket):
    """Account radio joules and debit UAV battery; GCS has an external supply."""
    requested = max(0.0, float(requested))
    if requested == 0.0:
        return 0.0, True
    is_uav = node < bridge.config.world.num_uavs
    if is_uav and bridge.config.energy.enabled:
        available = max(0.0, float(bridge.battery_j[node]))
        consumed = min(requested, available)
        bridge.battery_j[node] = available - consumed
        bridge.simulator.drones[node].residual_energy = bridge.battery_j[node]
        bridge.simulator.drones[node].sleep = not bridge.node_active(node)
        affordable = consumed >= requested
    else:
        consumed, affordable = requested, True
    bridge.energy_j[node] += consumed
    bucket[node] += consumed
    return consumed, affordable


def install_phy_budget(bridge):
    channel = bridge.simulator.channel
    original_transmit = channel.transmit
    original_evaluate = channel._evaluate

    def transmit(packet, transmitter, receivers):
        saved = bridge.native.TRANSMITTING_POWER
        bridge.native.TRANSMITTING_POWER = getattr(packet, "mission_power_w", saved)
        try:
            return original_transmit(packet, transmitter, receivers)
        finally:
            bridge.native.TRANSMITTING_POWER = saved

    def evaluate(target, receiver):
        result = original_evaluate(target, receiver)
        # A frame presented to an active receiver consumes listening energy even
        # when SINR/interference later makes the decode fail.
        if bridge.node_active(receiver):
            cost = receive_energy_j(target.packet.packet_length, bridge.config.network)
            _, affordable = _debit_radio(bridge, receiver, cost, bridge.rx_energy_j)
            if not affordable:
                sinr, rssi, interferers = result
                result = (float("-inf"), rssi, interferers)
        return result

    channel.transmit = transmit
    channel._evaluate = evaluate
    for drone in bridge.simulator.drones:
        phy = drone.mac_protocol.phy

        def emit(packet, receivers, drone=drone):
            node = drone.identifier
            kind = "data" if isinstance(packet, bridge.DataPacket) else "ack"
            if not bridge.node_active(node):
                bridge.diagnostic_counts[COUNT["energy_denials"]] += 1
                bridge.diagnostic_counts[COUNT[kind + "_energy_denials"]] += 1
                bridge.energy_blocked_transmissions += 1
                return False
            power = getattr(packet, "mission_power_w", bridge.config.network.tx_power_min_w)
            cost = transmit_energy_j(packet.packet_length, power, bridge.config.network)
            is_uav = node < bridge.config.world.num_uavs
            budgeted = is_uav and bridge.config.energy.enabled
            if budgeted and cost > bridge.battery_j[node]:
                bridge.diagnostic_counts[COUNT["energy_denials"]] += 1
                bridge.diagnostic_counts[COUNT[kind + "_energy_denials"]] += 1
                bridge.energy_blocked_transmissions += 1
                return False
            _debit_radio(bridge, node, cost, bridge.tx_energy_j)
            channel.transmit(packet, node, receivers)
            return True

        # Guard before channel.transmit; changing only the native consume hook
        # would still let an unaffordable frame reach the channel.
        phy.unicast = lambda packet, receiver, emit=emit: emit(packet, [receiver])
        phy.multicast = lambda packet, receivers, emit=emit: emit(packet, receivers)
        phy.broadcast = lambda packet, emit=emit, drone=drone: emit(
            packet, [d.identifier for d in bridge.simulator.drones if d is not drone])


def install_obstacle_loss(bridge):
    model = bridge.simulator.channel.channel_model
    original = model._estimate
    scale = 10 ** (-bridge.config.network.obstacle_extra_loss_db / 10)

    def estimate(transmitter, receiver):
        gain = original(transmitter, receiver)
        if not gain.line_of_sight:
            gain = replace(gain, nominal=gain.nominal * scale,
                           lower=gain.lower * scale, upper=gain.upper * scale)
        return gain

    model._estimate = estimate
