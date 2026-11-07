"""Validate all five MASAC BASE UI recipes and actual runtime architecture."""
from dataclasses import asdict

import pytest

from uav_network.cloud.matd3_ui_notebooks import CASES
from uav_network.cloud.masac_ui_notebooks import build_masac_ablation_notebook
from uav_network.cloud.ui_notebooks import SCALED_PROFILE, STAGE_START
from uav_network.configuration import load_config


@pytest.mark.parametrize('case_id', list(CASES))
def test_masac_ui_case_resolves_every_requested_setting_and_new_identity(case_id):
    shared, width, updates, layernorm = CASES[case_id]
    notebook=build_masac_ablation_notebook(case_id,b'frozen', 'frozen-source')
    ids=[]
    before=load_config(SCALED_PROFILE)
    for seed in (44,45):
        scope={'TRAIN_SEED':seed,'NOTEBOOK_REF':'owner/masac-case'}
        code=notebook.cells[-1].source
        exec(code[:code.index(STAGE_START)],scope)
        spec=scope['SPEC']
        c=load_config(spec['config'],spec['overrides'])
        assert spec['config']==SCALED_PROFILE and spec['commit']=='frozen-source'
        assert set(spec['runs'])=={'masac'} and spec['training_mode']=='base'
        assert spec['training_only'] and not spec['cpu_visualize']
        assert not spec['warm_start_artifacts'] and not spec['auto_continue']
        assert spec['requested_episodes']==20000 and spec['gpus_per_run']==2
        assert c.runtime.algorithm=='masac' and not c.pql.enabled
        assert c.runtime.seed==seed and c.runtime.episodes==20000
        assert c.runtime.checkpoint_interval==256
        assert c.runtime.num_envs==4096
        assert not c.runtime.vectorize_critics
        assert c.runtime.critic_chunk_size==0
        assert not c.runtime.compile_models and c.runtime.amp_mode=='off'
        assert not c.runtime.fused_optimizers and c.runtime.deterministic
        assert c.runtime.max_train_seconds==41400
        assert c.tracking.evaluation_interval_steps==1843200
        assert c.tracking.periodic_evaluation_episodes==5
        assert c.masac.share_actor_parameters is shared
        assert c.masac.share_critic_parameters is shared
        assert c.masac.hidden_sizes==[width,width]
        assert c.masac.actor_layer_norm is layernorm
        assert c.masac.updates_per_step==updates
        assert c.masac.mask_aware_discrete_entropy_target is True
        assert c.masac.temperature_loss=='alpha'
        assert c.matd3==before.matd3
        for key in ('world','sensor','network','energy','reward','pql'):
            assert asdict(getattr(c,key))==asdict(getattr(before,key))
        changed={'share_actor_parameters','share_critic_parameters','hidden_sizes','actor_layer_norm','updates_per_step'}
        bs,act=asdict(before.masac),asdict(c.masac)
        for key in changed:bs.pop(key);act.pop(key)
        assert bs==act
        run=spec['runs']['masac']; mode='SH' if shared else 'IND'
        norm='ALN' if layernorm else 'NOLN'
        assert f'-{mode}-' in run['name'] and f'-{norm}-' in run['name']
        assert f'-H{width}-U{updates}-' in run['name']
        assert f'-S{seed}-' in run['name']
        assert run['id'] in run['name'] and c.runtime.output_dir.endswith(run['id'])
        assert f'updates{updates}' in c.tracking.tags and f'seed{seed}' in c.tracking.tags
        assert ('shared-actors' if shared else 'independent-actors') in c.tracking.tags
        assert ('actor-layernorm' if layernorm else 'actor-no-layernorm') in c.tracking.tags
        assert 'critic-no-layernorm' in c.tracking.tags
        assert notebook.metadata.uav_ablation.hidden_sizes==[width,width]
        assert notebook.metadata.uav_ablation.filename.startswith(f'MASAC_{case_id}_')
        assert notebook.metadata.uav_ablation.source_commit=='frozen-source'
        ids.append(run['id'])
    assert ids[0]!=ids[1]


@pytest.mark.parametrize('case_id,seed',[('06',44),('01',-1),('01',True)])
def test_invalid_case_and_seed_are_rejected(case_id,seed):
    with pytest.raises(ValueError):
        build_masac_ablation_notebook(case_id,b'frozen','source',seed=seed)


@pytest.mark.parametrize('case_id', list(CASES))
def test_masac_requested_architecture_real_cpu_forward(case_id):
    """Instantiate actual MASAC layers from *emitted* overrides, not from labels."""
    import torch
    from torch import nn
    from uav_network.algorithms.masac import MASAC
    from uav_network.environment.observations import observation_dim,state_dim

    shared,width,updates,layernorm=CASES[case_id]
    nb=build_masac_ablation_notebook(case_id,b'frozen','source')
    scope={'TRAIN_SEED':44,'NOTEBOOK_REF':'owner/case'}
    code=nb.cells[-1].source
    exec(code[:code.index(STAGE_START)],scope)
    c=load_config(SCALED_PROFILE,scope['SPEC']['overrides']+['runtime.device=cpu','runtime.num_envs=2'])
    torch.set_num_threads(1)
    model=MASAC(c,observation_dim(c),state_dim(c))
    assert len(model.actors)==len(model.critics)==(1 if shared else 6)
    assert c.masac.updates_per_step==updates
    for actor in model.actors:
        linear=[layer for layer in actor.encoder.modules() if isinstance(layer,nn.Linear)]
        assert linear[0].in_features==(234 if shared else 228)
        assert [x.out_features for x in linear]==[width,width]
        assert sum(isinstance(layer,nn.LayerNorm) for layer in actor.modules())==(2 if layernorm else 0)
    assert not any(isinstance(layer,nn.LayerNorm) for layer in model.critics.modules())
    assert model.critics[0].q1[0].in_features==896


def test_masac_emitted_16u_independent_actor_critic_updates_are_finite():
    """Small true actor+critic update smoke without running a Kaggle job."""
    import numpy as np
    import torch
    from uav_network.algorithms.masac import MASAC
    from uav_network.algorithms.replay import TensorReplayBuffer
    from uav_network.environment.observations import observation_dim,state_dim
    from uav_network.environment.gpu.vector import WarpVectorEnv

    torch.set_num_threads(1)
    nb=build_masac_ablation_notebook('03',b'frozen','source')
    scope={'TRAIN_SEED':44,'NOTEBOOK_REF':'owner/case'}
    code=nb.cells[-1].source
    exec(code[:code.index(STAGE_START)],scope)
    c=load_config(SCALED_PROFILE,scope['SPEC']['overrides']+['runtime.device=cpu','runtime.num_envs=2'])
    learner=MASAC(c,observation_dim(c),state_dim(c))
    replay=TensorReplayBuffer(8,(6,228),830,continuous_dim=4)
    vector=WarpVectorEnv(c,device='cpu')
    try:
        obs,_=vector.reset(seed=44)
        for _ in range(4):
            bef={key:value.clone() if hasattr(value,'clone') else value.copy() for key,value in obs.items()}
            action=learner.act_batch(bef['observation'],return_tensors=True)
            obs,reward,done,_,_=vector.step({'continuous':action.continuous,'destination':action.destination})
            replay.add_batch(bef['observation'],bef['state'],action,reward,
                             obs['observation'],obs['state'],done.float())
        for _ in range(16):
            metrics=learner.update(replay.sample(8))
            assert np.isfinite(list(metrics.values())).all()
        assert learner.updates==learner.actor_updates==16
        assert all(torch.isfinite(p).all() for p in learner.parameters())
    finally:
        vector.close()
