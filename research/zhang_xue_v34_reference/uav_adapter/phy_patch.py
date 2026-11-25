import math


def db_to_linear(db_val: float) -> float:
    return 10.0 ** (db_val / 10.0)


def free_space_path_loss(distance_m: float, fc_hz: float = 2.4e9, alpha: float = 2.0) -> float:
    c = 3.0e8
    d = max(float(distance_m), 1.0)
    return (c / (4.0 * math.pi * fc_hz * d)) ** alpha


def received_power_watt(
    tx_power_w: float,
    distance_m: float,
    tx_gain_dbi: float = 0.0,
    rx_gain_dbi: float = 0.0,
    fc_hz: float = 2.4e9,
    alpha: float = 2.0,
) -> float:
    return tx_power_w * db_to_linear(tx_gain_dbi) * db_to_linear(rx_gain_dbi) * free_space_path_loss(
        distance_m, fc_hz=fc_hz, alpha=alpha
    )


def sinr_db_patched(
    desired_tx_power_w: float,
    desired_distance_m: float,
    interferers,
    tx_gain_dbi: float = 0.0,
    rx_gain_dbi: float = 0.0,
    noise_power_w: float = 4.0e-11,
    fc_hz: float = 2.4e9,
    alpha: float = 2.0,
    rx_radio_enabled: bool = True,
    tx_radio_enabled: bool = True,
) -> float:
    if not (rx_radio_enabled and tx_radio_enabled):
        return -float("inf")
    pr_sig = received_power_watt(
        desired_tx_power_w, desired_distance_m, tx_gain_dbi, rx_gain_dbi, fc_hz, alpha
    )
    interf_total = sum(
        received_power_watt(i_pt, i_dist, i_gt, rx_gain_dbi, fc_hz, alpha)
        for i_pt, i_dist, i_gt in interferers
    )
    sinr_lin = pr_sig / (noise_power_w + interf_total)
    return 10.0 * math.log10(max(sinr_lin, 1e-30))
