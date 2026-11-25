"""Replay a real four-transition packet simulation into each learner."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=spec_from_file_location("cpu_actor_learning_gate",ROOT/"scripts/v61_cpu_actor_learning_gate.py")
assert SPEC and SPEC.loader
MODULE=module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("algorithm",["matd3","masac"])
def test_four_real_packet_steps_feed_one_real_optimizer_update(algorithm):
    result=MODULE.integration_gate(algorithm,seed=44,transitions=4)
    assert result["transitions"]==4
    assert result["gradient_update_count"]==1
    assert result["finite_report"]
    assert (result["actor_obs_dim"],result["joint_critic_dim"])==(522,3132)
    if algorithm=="matd3":
        assert result["actor_weight_unchanged_expected_if_delayed"]
    else:
        assert not result["actor_weight_unchanged_expected_if_delayed"]
