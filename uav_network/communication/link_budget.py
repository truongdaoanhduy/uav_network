"""Pure nominal A2A link-budget and normalized TX-power helpers."""
import math

C_M_S = 299_792_458.0
NLOS = {"urban": (3.0, 12.0), "suburban": (2.6, 8.0)}


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
