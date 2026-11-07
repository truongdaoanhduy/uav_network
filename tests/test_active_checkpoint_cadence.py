"""Keep active 20k baseline profiles on the requested 256-episode cadence."""
import pytest

from uav_network.configuration import load_config


@pytest.mark.parametrize('profile', [
    'comparison-reference-masac-matd3-xyz-sensing.yaml',
    'comparison-independent-baseline-xyz-sensing.yaml',
    'comparison-independent-35t25o-h512-xyz-sensing.yaml',
])
def test_20k_training_checkpoints_every_256_completed_episodes(profile):
    c = load_config('configs/' + profile)
    assert c.runtime.episodes == 20000
    assert c.runtime.checkpoint_interval == 256
    assert c.tracking.local_checkpoints_to_keep == 2
    assert c.tracking.evaluation_interval_steps == 3686400
    assert c.tracking.periodic_evaluation_episodes == 5
    assert c.runtime.eval_episodes == 30
