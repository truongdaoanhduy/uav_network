"""Pure nominal A2A link-budget and normalized TX-power helpers."""
import math

C_M_S = 299_792_458.0
NLOS = {"urban": (3.0, 12.0), "suburban": (2.6, 8.0)}
DATA_HEADER_BITS = 464.0


def _unit_interval(value):
    return min(1.0, max(0.0, float(value)))


def power_from_action(action_value: float, network_config) -> float:
    u = _unit_interval((float(action_value) + 1.0) / 2.0)
    low, high = float(network_config.tx_power_min_w), float(network_config.tx_power_max_w)
    if network_config.tx_power_mapping == "linear":
        return low + u * (high - low)
    if network_config.tx_power_mapping == "log":
        return low * (high / low) ** u
    raise ValueError(f"Unsupported tx_power_mapping: {network_config.tx_power_mapping}")


def power_fraction(power_w: float, network_config) -> float:
    power = float(power_w)
    low, high = float(network_config.tx_power_min_w), float(network_config.tx_power_max_w)
    if network_config.tx_power_mapping == "linear":
        return _unit_interval((power - low) / (high - low)) if high > low else 0.0
    if network_config.tx_power_mapping == "log":
        if power <= 0:
            return 0.0
        return _unit_interval(math.log(power / low) / math.log(high / low)) if high > low else 0.0
    raise ValueError(f"Unsupported tx_power_mapping: {network_config.tx_power_mapping}")


def required_power_w(distance_m: float, frequency_hz: float, bandwidth_hz: float,
                     noise_density_dbm_hz: float, noise_figure_db: float,
                     sinr_threshold_db: float, nlos_model: str,
                     line_of_sight: bool, obstacle_extra_loss_db: float = 0.0) -> float:
    distance = max(1.0, float(distance_m))
    frequency = float(frequency_hz)
    if frequency <= 0 or bandwidth_hz <= 0:
        raise ValueError("frequency and bandwidth must be positive")
    reference_gain = (C_M_S / frequency / (4.0 * math.pi)) ** 2
    if line_of_sight:
        exponent, loss_db = 2.0, 0.0
    else:
        try:
            exponent, loss_db = NLOS[nlos_model]
        except KeyError as exc:
            raise ValueError(f"Unsupported NLOS model: {nlos_model}") from exc
        loss_db += float(obstacle_extra_loss_db)
    gain = reference_gain * distance ** (-exponent) * 10 ** (-loss_db / 10.0)
    noise_dbm = float(noise_density_dbm_hz) + 10.0 * math.log10(float(bandwidth_hz)) + float(noise_figure_db)
    noise_w = 10 ** ((noise_dbm - 30.0) / 10.0)
    threshold = 10 ** (float(sinr_threshold_db) / 10.0)
    return threshold * noise_w / gain


def polygon_line_of_sight(start_xy, end_xy, obstacles, sides: int) -> bool:
    """Match the regular-polygon obstacle footprint used by UavNetSim/Warp."""
    import numpy as np
    a = np.asarray(start_xy, dtype=float)
    b = np.asarray(end_xy, dtype=float)
    d = b - a
    for obstacle in np.asarray(obstacles, dtype=float).reshape(-1, 3):
        center, radius = obstacle[:2], float(obstacle[2])
        low, high, intersects = 0.0, 1.0, True
        for j in range(int(sides)):
            angle = j * 2.0 * math.pi / sides
            next_angle = (j + 1) * 2.0 * math.pi / sides
            first = center + radius * np.array([math.cos(angle), math.sin(angle)])
            second = center + radius * np.array([math.cos(next_angle), math.sin(next_angle)])
            edge = second - first
            f = edge[0] * (a - first)[1] - edge[1] * (a - first)[0]
            slope = edge[0] * d[1] - edge[1] * d[0]
            if slope == 0.0:
                if f < 0.0:
                    intersects = False
            elif slope > 0.0:
                low = max(low, -f / slope)
            else:
                high = min(high, -f / slope)
        if intersects and low <= high:
            return False
    return True


def nominal_required_power_w(source_xy, destination_xy, obstacles, network_config,
                             altitude_m: float = 0.0, destination_is_gcs: bool = False) -> float:
    import numpy as np
    source = np.asarray(source_xy, dtype=float)
    destination = np.asarray(destination_xy, dtype=float)
    horizontal = float(np.linalg.norm(destination - source))
    distance = math.hypot(horizontal, float(altitude_m) if destination_is_gcs else 0.0)
    line_of_sight = polygon_line_of_sight(source, destination, obstacles,
                                          network_config.obstacle_polygon_sides)
    return required_power_w(distance, network_config.carrier_frequency_hz,
                            network_config.bandwidth_hz, network_config.noise_density_dbm_hz,
                            network_config.noise_figure_db, network_config.sinr_threshold_db,
                            network_config.nlos_model, line_of_sight,
                            network_config.obstacle_extra_loss_db)


def report_reference_tx_energy_j(network_config) -> float:
    """One complete DATA report emitted once at Pmax; excludes ACKs/retries."""
    remaining = int(network_config.report_bytes)
    total_bits = 0.0
    payload = int(network_config.payload_bytes)
    while remaining > 0:
        fragment = min(payload, remaining)
        total_bits += fragment * 8.0 + DATA_HEADER_BITS
        remaining -= fragment
    return total_bits / float(network_config.bit_rate_bps) * float(network_config.tx_power_max_w)
