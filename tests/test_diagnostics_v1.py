"""Pooled completed-outcome diagnostics must keep their actual denominators."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from uav_network.configuration import Config, load_config
from uav_network.experiments.tracking import episode_metrics
from uav_network.experiments.diagnostics import TrainingDiagnostics
from uav_network.experiments.comparison_metrics import evaluation_protocol


def outcome(step=8, value=1., generated=1, received=1, deliveries=1, samples=1):
    return {
        "episode": 1, "global_step": step, "return": value, "episode_length": 4,
        "num_uavs": 2, "search_rate_percent": 100., "delivery_rate_percent": 50.,
        "success": False, "blocked_moves": 0, "boundary_clips": 0,
        "expired_reports": 0, "dropped_reports": 0,
        "energy": {"propulsion_energy_j": 100., "battery_fraction": [.8, .7],
                   "battery_j": [80., 70.], "active_uavs": 2,
                   "per_uav_propulsion_energy_j": [40., 60.],
                   "per_uav_rf_energy_j": [1., 2.],
                   "per_uav_tx_radio_energy_j": [.7, 1.3],
                   "per_uav_rx_radio_energy_j": [.3, .7],
                   "per_uav_total_energy_j": [41., 62.]},
        "network": {"enabled": True, "generated_packets": generated, "received_packets": received,
                    "uav_communication_energy_j": 3., "uav_tx_energy_j": 2., "uav_rx_energy_j": 1.,
                    "gcs_tx_energy_j": .1, "gcs_rx_energy_j": .2,
                    "diagnostics": {"data_attempts": samples, "data_phy_successes": samples,
                        "action_slots": 8, "action_receiver_eligible": 2, "action_silent": 4,
                        "data_tx_power_w": {"count": samples, "sum": .2 * samples,
                            "mean": .2 if samples else None, "min": .2 if samples else None,
                            "max": .2 if samples else None},
                        "data_tx_power_histogram": [0, samples, 0, 0, 0, 0],
                        "delivery_hop_histogram": [deliveries, 0, 0, 0, 0, 0, 0, 0]}},
        "report_diagnostics": {"delivery_count": deliveries,
            "delivery_hops_mean": 1. if deliveries else None,
            "delivered_report_age_mean_s": 2. if deliveries else None,
            "first_discovery_to_delivery_mean_s": 3. if deliveries else None,
            "discovered_undelivered": 1, "discovered_undelivered_with_custody": 1,
            "discovered_undelivered_without_custody": 0,
            "buffer_peak_bytes": [1024, 2048], "oldest_report_age_s": [None, 3.]},
    }


def test_episode_emission_counts_and_per_uav_values_keep_null_conditionals():
    r = outcome(samples=0, deliveries=0)
    metrics = episode_metrics(r, 2)
    assert metrics["network/diagnostics/data_tx_power_w/count"] == 0
    assert metrics["network/diagnostics/data_tx_power_w/mean"] is None
    assert metrics["energy/uav_1/total_energy_j"] == 62.
    assert metrics["energy/uav_0/battery_j"] == 80.
    assert metrics["reports/uav_0/oldest_report_age_s"] is None
    assert metrics["reports/delivery_count"] == 0
    assert metrics["energy/j_per_gcs_delivery"] is None
    assert metrics["network/diagnostics/action_receiver_eligible_fraction"] == .25
    assert metrics["network/diagnostics/data_tx_power_histogram/bin_1"] == 0
    legacy = deepcopy(r)
    legacy["network"].pop("diagnostics")
    legacy.pop("report_diagnostics")
    assert episode_metrics(legacy, 2)["network/diagnostics/available"] == 0


def test_v3_radio_energy_pool_uses_raw_joules_and_delivery_denominator():
    from uav_network.experiments.telemetry import OutcomePool
    pool = OutcomePool(); pool.add([outcome(deliveries=1), outcome(deliveries=2)])
    values = pool.metrics()
    assert values["energy/uav_tx_radio_energy_j"] == 4.
    assert values["energy/uav_rx_radio_energy_j"] == 2.
    assert values["energy/uav_rf_energy_j"] == 6.
    assert values["energy/radio_j_per_gcs_delivery"] == 2.
    episode = episode_metrics(outcome(deliveries=1), 2)
    assert episode["energy/uav_tx_radio_energy_j"] == 2.
    assert episode["energy/uav_rx_radio_energy_j"] == 1.
    assert episode["energy/gcs_tx_radio_energy_j"] == .1
    assert episode["energy/gcs_rx_radio_energy_j"] == .2
    assert episode["energy/uav_1/tx_radio_energy_j"] == 1.3
    assert episode["energy/uav_1/rx_radio_energy_j"] == .7


def test_time_windows_pool_raw_counts_and_finish_entire_collection_before_comparing():
    results = []
    for reverse in (False, True):
        with TemporaryDirectory() as directory:
            c = Config()
            c.runtime.num_envs, c.world.max_steps = 2, 4
            c.tracking.diagnostic_window = 2
            d = TrainingDiagnostics(c, directory)
            records = [outcome(value=v, generated=g, received=p, deliveries=t, samples=n)
                       for v, g, p, t, n in ((10., 1, 1, 1, 1), (-10., 9, 0, 0, 3))]
            if reverse:
                records.reverse()
            values = d.batch(records, 8)
            results.append(values)
            assert values["cohort/episodes"] == 2
            assert values["cohort/source_start_step"] == 0
            assert values["cohort/source_end_step"] == 8
            assert values["cohort/partial"] == 0
            assert values["cohort/network/packet_pdr_fraction"] == .1
            assert values["cohort/network/generated_packets"] == 10
            assert values["cohort/network/diagnostics/data_tx_power_w/count"] == 4
            assert values["cohort/network/diagnostics/data_tx_power_histogram/bin_1"] == 4
            assert values["cohort/energy/j_per_gcs_delivery"] == 206.
            assert values["diagnostics/return_drop_warning"] == 0
            assert not any(json.loads(line)["code"] == "return_drop"
                           for line in d.path.read_text().splitlines())
    assert results[0] == results[1]


def test_partial_and_empty_windows_are_labelled_and_return_drop_needs_distinct_time():
    with TemporaryDirectory() as directory:
        c = Config()
        c.runtime.num_envs, c.world.max_steps = 2, 4
        d = TrainingDiagnostics(c, directory, initial_step=3)
        first = d.batch([outcome(step=8, value=100.)], 8)
        assert first["cohort/partial"] == 1
        empty = d.batch([], 16)
        assert empty["cohort/episodes"] == 0
        assert empty["cohort/return_mean"] is None
        assert empty["cohort/energy/j_per_gcs_delivery"] is None
        values = d.batch([outcome(step=18, value=-100.)], 18, final=True)
        assert values["cohort/partial"] == 1
        assert values["cohort/source_end_step"] == 18
        assert values["diagnostics/return_drop_warning"] == 0


def test_periodic_profile_keeps_four_cases_matched_and_final_test_bank_disjoint():
    old = load_config("configs/comparison-25k.yaml")
    c = load_config("configs/comparison-25k-diagnostics.yaml")
    for algorithm in ("masac", "matd3"):
        for pql in (False, True):
            c.runtime.algorithm, c.pql.enabled = algorithm, pql
            c.validate()
            assert c.world == old.world
            assert c.algorithm_config == getattr(old, algorithm)
            assert c.runtime.episodes == 25000
            assert c.runtime.num_envs == 4096
            assert c.runtime.seed == 44
            assert c.tracking.evaluation_interval_steps == 7372800
            assert c.tracking.periodic_evaluation_episodes == 5
            assert c.tracking.periodic_evaluation_seed == 1000000044
            assert (c.runtime.eval_seed, c.runtime.eval_episodes) == (2000000044, 30)
    assert c.tracking.group == "uav-network-base-pql-25k-seed44-t35-o25-diagnostics-v1"
    assert {"diagnostics-v1", "eval5", "finaleval30"} <= set(c.tracking.tags)
    assert (c.tracking.entity, c.tracking.project) == ("truongdaoanhduy-uit", "uav_search_target")
    protocol = evaluation_protocol(c, list(range(2000000044, 2000000074)),
                                   {"next_seed": 50000, "global_step": 7372800, "episodes": 0})
    assert protocol["validation_seed_overlap"] is False
    exposed = evaluation_protocol(c, [1000000044], {"next_seed": 50000, "global_step": 7372800, "episodes": 0})
    assert exposed["validation_seed_overlap"] is True
    assert exposed["validation_seed_intervals_current"] == [[1000000044, 1000000048]]


@pytest.mark.parametrize("field,value", [
    ("evaluation_interval_steps", -1), ("evaluation_interval_steps", 1.5),
    ("evaluation_interval_steps", True), ("periodic_evaluation_episodes", 0),
    ("periodic_evaluation_episodes", 1.5), ("periodic_evaluation_seed", -1),
    ("periodic_evaluation_seed", True), ("evaluation_timeout_seconds", float("nan")),
])
def test_periodic_evaluation_config_rejects_invalid_units(field, value):
    c = Config()
    setattr(c.tracking, field, value)
    with pytest.raises(ValueError, match="evaluation"):
        c.validate()


def test_pooled_power_means_use_attempt_counts_and_closed_windows_can_warn():
    with TemporaryDirectory() as directory:
        c = Config()
        c.runtime.num_envs, c.world.max_steps = 2, 4
        c.tracking.diagnostic_window = 2
        d = TrainingDiagnostics(c, directory)
        a, b = outcome(value=100.), outcome(value=100., samples=3)
        a["network"]["diagnostics"]["data_tx_power_w"] = dict(count=1, sum=.1, mean=.1, min=.1, max=.1)
        b["network"]["diagnostics"]["data_tx_power_w"] = dict(count=3, sum=1.5, mean=.5, min=.5, max=.5)
        first = d.batch([a, b], 8)
        assert first["cohort/network/diagnostics/data_tx_power_w/mean"] == .4
        assert first["cohort/network/diagnostics/data_tx_power_w/min"] == .1
        assert first["cohort/network/diagnostics/data_tx_power_w/max"] == .5
        second = d.batch([outcome(step=16, value=0.), outcome(step=16, value=0.)], 16)
        assert second["diagnostics/return_drop_warning"] == 1
        events = [json.loads(line) for line in d.path.read_text().splitlines()]
        assert events[-1]["code"] == "return_drop"
        assert events[-1]["root_cause_confirmed"] is False
        windows = [json.loads(line) for line in d.cohorts_path.read_text().splitlines()]
        assert [window["cohort/source_end_step"] for window in windows] == [8, 16]


def test_legacy_rolling_return_change_stays_descriptive_without_regression_warning():
    with TemporaryDirectory() as directory:
        c = Config()
        c.tracking.diagnostic_window = 2
        d = TrainingDiagnostics(c, directory)
        for value in (100., 100., 0., 0.):
            values = d.episode(outcome(step=8, value=value))
        assert values["rolling/return_change"] == -100.
        assert not any(json.loads(line)["code"] == "return_drop"
                       for line in d.path.read_text().splitlines())


def test_v2_pooled_fractions_use_raw_attempt_and_relay_denominators():
    from uav_network.experiments.telemetry import OutcomePool
    a, b = outcome(samples=1), outcome(samples=3)
    for record, attempts, below, peers, toward in ((a,1,1,1,1),(b,3,0,3,1)):
        d = record["network"]["diagnostics"]
        d.update(data_attempts=attempts, data_below_required_power_attempts=below,
                 custody_peer_transfers=peers, relay_toward_gcs_transfers=toward,
                 relay_away_from_gcs_transfers=peers-toward)
    pool = OutcomePool(); pool.add([a,b]); values = pool.metrics()
    assert values["network/diagnostics/data_below_required_power_fraction"] == .25
    assert values["network/diagnostics/relay_toward_gcs_fraction"] == .5


def test_scenario_v2_profile_is_new_and_v1_remains_immutable():
    import hashlib
    from uav_network.environment.observations import observation_dim
    from uav_network.communication.link_budget import power_from_action
    root = Path(__file__).resolve().parents[1]
    v1_path = root / "configs/comparison-25k-diagnostics.yaml"
    assert hashlib.sha256(v1_path.read_bytes()).hexdigest() == "6253e377e72d2caf709c017dfdcd016b45b4b895605a4c257a704c364dd3d48b"
    v1 = load_config(v1_path)
    v2 = load_config(root / "configs/comparison-25k-scenario-v2.yaml")
    assert (v2.world.num_uavs, v2.world.num_targets, v2.world.num_obstacles, v2.world.max_steps) == (6,35,25,1800)
    assert (v2.network.peer_range_m, v2.network.gcs_range_m) == (500.0,500.0)
    assert (v2.network.report_bytes, v2.network.buffer_bytes, v2.network.report_ttl_s, v2.network.max_retries) == (1024,3072,300.0,5)
    assert (v2.network.tx_power_min_w, v2.network.tx_power_max_w, v2.network.tx_power_mapping) == (0.001,0.4,"log")
    assert v2.network.power_limited_admission and v2.network.link_state_observation
    assert (v2.reward.communication_efficiency, v2.reward.shield_correction) == (1.0,0.1)
    assert observation_dim(v2) == observation_dim(v1) + 3 * v2.world.num_uavs
    assert power_from_action(-1, v1.network) == 0.1
    assert power_from_action(-1, v2.network) == 0.001
    p1 = evaluation_protocol(v1, [2000000044], {"next_seed":50000})
    p2 = evaluation_protocol(v2, [2000000044], {"next_seed":50000})
    assert p1["scenario_hash"] != p2["scenario_hash"]
    assert (v2.runtime.eval_seed, v2.runtime.eval_episodes) == (2000000044,30)
    assert "scenario-v2" in v2.tracking.tags
    assert "scenario-v2" in v2.tracking.group


def test_outcome_pool_ignores_undefined_derived_diagnostic_scalars():
    from uav_network.experiments.telemetry import OutcomePool
    r = outcome(samples=0)
    r["network"]["diagnostics"].update(
        data_below_required_power_attempts=0,
        data_below_required_power_fraction=None,
        custody_peer_transfers=0,
        relay_toward_gcs_transfers=0,
        relay_toward_gcs_fraction=None,
    )
    pool = OutcomePool()
    pool.add([r])
    values = pool.metrics()
    assert values["network/diagnostics/data_below_required_power_fraction"] is None
    assert values["network/diagnostics/relay_toward_gcs_fraction"] is None
