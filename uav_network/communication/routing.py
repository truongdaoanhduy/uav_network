"""Policy-selected single-hop routing, using upstream CSMA/CA ACK handling."""
import copy


class SelectedNextHopRouting:
    def __init__(self, simulator, my_drone):
        self.simulator = simulator
        self.my_drone = my_drone

    def next_hop_selection(self, packet):
        packet.next_hop_id = packet.dst_drone.identifier
        if self.my_drone.identifier not in packet.intermediate_drones:
            packet.intermediate_drones.append(self.my_drone.identifier)
        return True, packet, False

    def packet_reception(self, packet, src_drone_id):
        from entities.packet import AckPacket, DataPacket
        from utils import config

        if not self.simulator.mission_bridge.node_active(self.my_drone.identifier):
            return
        if isinstance(packet, DataPacket):
            received = copy.copy(packet)
            if received.packet_id not in self.simulator.metrics.datapacket_arrived:
                self.simulator.metrics.calculate_metrics(received)
                self.simulator.mission_bridge.packet_received(received, self.my_drone.identifier)
            config.GL_ID_ACK_PACKET += 1
            ack = AckPacket(src_drone=self.my_drone,
                            dst_drone=self.simulator.drones[src_drone_id],
                            ack_packet_id=config.GL_ID_ACK_PACKET,
                            ack_packet_length=config.ACK_PACKET_LENGTH,
                            ack_packet=received, simulator=self.simulator,
                            channel_id=received.channel_id)
            ack.mission_power_w = self.simulator.mission_bridge.config.network.tx_power_min_w
            yield self.simulator.env.timeout(config.SIFS_DURATION)
            ack.increase_ttl()
            self.my_drone.mac_protocol.phy.unicast(ack, src_drone_id)
            yield self.simulator.env.timeout(ack.packet_length / config.BIT_RATE * 1e6)
        elif isinstance(packet, AckPacket):
            data = packet.ack_packet
            if data.first_attempt_time is not None:
                self.simulator.metrics.mac_delay.append(
                    (self.simulator.env.now - data.first_attempt_time) / 1e3)
            self.my_drone.remove_from_queue(data)
            mac = self.my_drone.mac_protocol
            key = f"wait_ack{self.my_drone.identifier}_{data.packet_id}"
            process = mac.wait_ack_process_dict.get(key)
            if mac.wait_ack_process_finish.get(key, 1) == 0 and process and not process.triggered:
                mac.wait_ack_process_finish[key] = 1
                process.interrupt()

    def penalize(self, *args, **kwargs):
        """Routing is learned by MASAC, not an upstream routing learner."""
