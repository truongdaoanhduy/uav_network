"""Notebook-source preflight: exact paper-core config, no accidental old run rewrite."""
from pathlib import Path

import pytest

from uav_network.cloud.ui_notebooks import build_independent_notebook, PAPER_CORE_PROFILE
from uav_network.configuration import load_config

STAGE_START='if SPEC["stage"] in {"visualize", "archive"}:'


@pytest.mark.parametrize("algorithm",["matd3","masac"])
def test_paper_core_notebook_resolves_exact_config_no_training(algorithm):
    nb=build_independent_notebook(
        algorithm,b"dummy-source-archive","frozen-test-commit",
        seed=44,scaled=True,paper_core=True)
    assert nb.metadata.uav_baseline.baseline_class=="PAPER_CORE_CTDE_HYBRID"
    assert nb.metadata.uav_baseline.profile == PAPER_CORE_PROFILE
    code=nb.cells[-1].source
    for seed in (44,45):
        scope={"NOTEBOOK_REF":"owner/paper-core-35t25o","TRAIN_SEED":seed}
        exec(code[:code.index(STAGE_START)],scope)
        spec=scope["SPEC"]
        assert spec["config"]==PAPER_CORE_PROFILE
        assert spec["stage"]=="train" and spec["training_only"] is True
        assert spec["auto_continue"] is False
        assert not spec["warm_start_artifacts"]
        assert spec["requested_episodes"]==20000
        assert "PAPERCORE-INDEPENDENT" in spec["runs"][algorithm]["name"]
        assert f"-S{seed}-" in spec["runs"][algorithm]["name"]
        c=load_config(spec["config"],spec["overrides"])
        assert c.runtime.algorithm==algorithm
        assert c.runtime.seed==seed
        assert c.tracking.group=="uav-paper-core-35t25o-h512-u8-1500"
        assert c.world.num_targets==35 and c.world.num_obstacles==25
        assert c.world.max_steps==1500
        assert c.runtime.num_envs==4096 and c.runtime.critic_chunk_size==0
        assert c.matd3.hidden_sizes==[512,512] and c.masac.hidden_sizes==[512,512]
        assert c.matd3.updates_per_step==8 and c.masac.updates_per_step==8
        assert not c.matd3.share_actor_parameters and not c.matd3.share_critic_parameters
        assert not c.masac.share_actor_parameters and not c.masac.share_critic_parameters
        assert not c.matd3.actor_layer_norm and not c.masac.actor_layer_norm
        assert not c.pql.enabled
        assert c.masac.temperature_loss=="alpha"
        assert c.masac.mask_aware_discrete_entropy_target
        assert c.matd3.initialization=="default"
        assert c.matd3.output_normalization is False


def test_paper_core_is_explicit_not_a_silent_override():
    with pytest.raises(ValueError,match="requires scaled"):
        build_independent_notebook("matd3",b"bundle","sha",scaled=False,paper_core=True)
    normal=build_independent_notebook("matd3",b"bundle","sha",scaled=True)
    assert normal.metadata.uav_baseline.profile != PAPER_CORE_PROFILE
    assert "PAPERCORE" not in normal.cells[-1].source
