"""Apply battery budgets and packet-selected power at the real native PHY."""
from dataclasses import replace


def install_phy_budget(bridge):
    channel = bridge.simulator.channel
    original_transmit = channel.transmit

    def transmit(packet, transmitter, receivers):
        saved = bridge.native.TRANSMITTING_POWER
        bridge.native.TRANSMITTING_POWER = getattr(packet, "mission_power_w", saved)
        try:
            return original_transmit(packet, transmitter, receivers)
        finally:
            bridge.native.TRANSMITTING_POWER = saved

    channel.transmit = transmit
    for drone in bridge.simulator.drones:
        phy = drone.mac_protocol.phy

        def emit(packet, receivers, drone=drone):
            node = drone.identifier
            if not bridge.node_active(node):
                bridge.energy_blocked_transmissions += 1
                return False
            power = getattr(packet, "mission_power_w", bridge.config.network.tx_power_min_w)
            cost = packet.packet_length / bridge.config.network.bit_rate_bps * power
            is_uav = node < bridge.config.world.num_uavs
            budgeted = is_uav and bridge.config.energy.enabled
            if budgeted and cost > bridge.battery_j[node]:
                bridge.energy_blocked_transmissions += 1
                return False
            bridge.energy_j[node] += cost
            if budgeted:
                bridge.battery_j[node] = max(0.0, bridge.battery_j[node] - cost)
                drone.residual_energy = bridge.battery_j[node]
                drone.sleep = not bridge.node_active(node)
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
