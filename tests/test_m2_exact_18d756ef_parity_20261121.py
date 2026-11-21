"""Exact 18d756ef W&B run model/train/eval/test settings for future M2 Warp rollout.

Passing configuration tests never establishes a finished Warp M2 simulator.
"""
from pathlib import Path
import hashlib
import numpy as np
import pytest
import torch

from uav_network.configuration import load_config
from uav_network.algorithms.matd3 import MATD3
from uav_network.algorithms.masac import MASAC
from uav_network.algorithms.replay import transition_shapes
from uav_network.experiments.pql_replay import make_replay
from uav_network.environment.vector import make_vector

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/m2-target2d-exact-18d756ef-20261121.yaml'
# Verified source extracted from the real Kaggle run therwed/notebook5051433b29.
MODEL_SHA='c1b65c7d5e64574404f3f0394d8377de92d886e7ab126576153ad82a981b2f8f'
MATD3_SHA='e106277d2695936d983e2b558ad5cefac7a1015b24cca3b8032c727563bfeefa'
MASAC_SHA='7e5fd9cb00c664bbea88321dcfdd3d0b75527c992b041a1a42434162a92796ce'


def test_exact_18d756ef_model_source_fingerprint():
    assert hashlib.sha256((ROOT/'uav_network/algorithms/models.py').read_bytes()).hexdigest()==MODEL_SHA
    assert hashlib.sha256((ROOT/'uav_network/algorithms/matd3.py').read_bytes()).hexdigest()==MATD3_SHA
    assert hashlib.sha256((ROOT/'uav_network/algorithms/masac.py').read_bytes()).hexdigest()==MASAC_SHA


def test_every_legacy_training_config_value_preserved_except_declared_m2_fields():
    import yaml
    original=yaml.safe_load((ROOT/'configs/reference-18d756ef-wandb-resolved-20261121.yaml').read_text())
    migrated=yaml.safe_load(CONFIG.read_text())
    original['runtime']['m2_enabled']=True
    original['runtime']['max_train_seconds']=0
    original['runtime']['output_dir']='outputs/m2-2d-18d756ef-warp-matd3'
    original['tracking']['group']='M2-TARGET2D-18d756ef-20261121'
    original['tracking']['tags']=['m2-target-belief-2d','exact-18d756ef-training-protocol',
                                  'old-model-actorbn-plaincritic','unported-full-warp-m2']
    assert migrated==original


def test_m2_preserves_exact_18d756ef_training_validation_and_final_test_profile():
    c=load_config(CONFIG)
    assert c.runtime.m2_enabled and c.runtime.environment_backend=='warp_packet'
    assert c.runtime.num_envs==4096 and c.runtime.episodes==20000
    assert c.runtime.max_train_seconds==0
    assert c.runtime.seed==44 and c.runtime.eval_seed==2000000044
    assert c.runtime.eval_episodes==4000
    assert c.runtime.replay_cpu_fraction==.75 and c.runtime.replay_device=='cpu'
    assert c.world.map_size_m==1000 and c.world.max_steps==1500
    assert c.world.num_uavs==6 and c.world.num_targets==35 and c.world.num_obstacles==25
    assert c.sensor.grid_cell_m==25
    assert c.matd3.hidden_sizes==[256,512] and c.masac.hidden_sizes==[256,512]
    assert c.matd3.actor_batch_norm is True and c.matd3.critic_batch_norm is False
    assert c.runtime.collector_overlap is True
    assert c.matd3.share_actor_parameters and c.matd3.share_critic_parameters
    assert c.matd3.actor_layer_norm and c.matd3.initialization=='default'
    assert c.matd3.replay_capacity==1000000
    assert c.matd3.batch_size==8192 and c.matd3.updates_per_step==4
    assert c.matd3.learning_starts==131072 and c.matd3.actor_output_regularization==0
    assert c.tracking.periodic_evaluation_episodes==100
    assert c.tracking.deferred_gpu_eval_collection_stride==100
    assert c.tracking.deferred_gpu_eval_enabled
    assert c.runtime.checkpoint_interval==256
    assert c.continuous_dim==5 and c.motion_dim==3
    assert c.world.motion_safety=='reject'


def test_old_hybrid_replay_real_75_cpu_25_hot_layout_with_m2_5channel_actions():
    c=load_config(CONFIG)
    # CPU hot device is used only for the CI test to avoid requiring CUDA.
    # The actual GPU deployment must have `hot_device=cuda` and the same split.
    c.matd3.replay_capacity=20
    c.matd3.batch_size=4
    replay=make_replay(c,(6,255),6*255,'cpu')
    assert replay.cpu_capacity==15 and replay.hot_capacity==5
    assert replay.continuous_dim==5
    rng=np.random.default_rng(44)
    for k in range(5):
        from uav_network.environment.env import Action
        obs=rng.normal(size=(4,6,255)).astype('f')
        state=obs.reshape(4,-1)
        action=Action(rng.uniform(-1,1,(4,6,5)).astype('f'),rng.integers(0,7,(4,6)))
        replay.add_batch(obs,state,action,rng.normal(size=4).astype('f'),obs,state,np.zeros(4,'f'))
    assert len(replay)==20
    batch=replay.sample(4)
    assert batch['continuous'].shape==(4,6,5)
    assert batch['state'].shape==(4,6*255)
    assert batch['destination'].dtype==torch.int64


def test_m2_warp_train_is_blocked_until_real_packet_and_belief_parity():
    c=load_config(CONFIG)
    c.runtime.num_envs=1
    c.runtime.device='cpu'
    with pytest.raises(NotImplementedError,match='M2'):
        make_vector(c,device='cpu')


def test_old_matd3_5channel_forward_and_gradients_for_m2():
    c=load_config(CONFIG)
    torch.manual_seed(44)
    actor=MATD3(c,255,1530,device='cpu')
    observations=torch.rand(4,6,255)
    # Established architecture, with only 255D observation and 5D action ABI.
    act=actor.act_batch(observations,deterministic=True)
    assert tuple(act.continuous.shape)==(4,6,5)
    assert tuple(act.destination.shape)==(4,6)
    assert np.isfinite(act.continuous).all()
    # Original author-reference output penalty must actually be executable.
    terms=actor._actor_output_regularization(observations)
    assert torch.isfinite(terms).all()
    assert float(terms.mean())>=0


def test_old_masac_5channel_original_policy_model_compatible_with_m2():
    c=load_config(CONFIG)
    torch.manual_seed(44)
    learner=MASAC(c,255,1530,device='cpu')
    observed=torch.rand(2,6,255)
    action=learner.act_batch(observed,deterministic=True)
    assert action.continuous.shape==(2,6,5)
    assert action.destination.shape==(2,6)
    assert np.isfinite(action.continuous).all()


def test_direct_legacy_warp_vector_constructor_must_refuse_m2_before_allocation():
    # Direct use of the class must not bypass make_vector's M2 guard and
    # silently run the old true-target-id radio/old Bayesian observation.
    from uav_network.environment.gpu.vector import WarpVectorEnv
    c=load_config(CONFIG)
    c.runtime.num_envs=1
    c.runtime.device='cpu'
    with pytest.raises(NotImplementedError,match='M2'):
        vector=WarpVectorEnv(c,device='cpu')
        vector.close()


@pytest.mark.parametrize('algorithm',["matd3","masac"])
def test_pinned_legacy_matd3_masac_learners_update_on_m2_hybrid_action_batch(algorithm):
    c=load_config(CONFIG)
    c.runtime.algorithm=algorithm
    # Keep model architecture and regularization exactly the pinned source.
    torch.manual_seed(44)
    learner=(MATD3 if algorithm=='matd3' else MASAC)(c,255,1530,device='cpu')
    rng=np.random.default_rng(44)
    obs=rng.normal(size=(5,6,255)).astype('float32')
    next_obs=rng.normal(size=(5,6,255)).astype('float32')
    # These are only local action masks, not privileged receiver feasibility.
    obs[:,:,-7:]=1
    next_obs[:,:,-7:]=1
    batch={'observation':obs,'next_observation':next_obs,
           'state':obs.reshape(5,-1),'next_state':next_obs.reshape(5,-1),
           'continuous':rng.uniform(-1,1,size=(5,6,5)).astype('float32'),
           'destination':rng.integers(0,7,size=(5,6)),
           'reward':rng.normal(size=5).astype('float32'),
           'terminated':np.zeros(5,dtype='float32')}
    first_params={name:value.detach().clone() for name,value in learner.named_parameters() if value.requires_grad}
    for _ in range(2):
        output=learner.update(batch,collect_metrics=True)
    assert learner.updates==2
    assert all(torch.isfinite(parameter).all() for parameter in learner.parameters())
    assert all(np.isfinite(float(x)) for x in output.values() if isinstance(x,(float,int,np.number)))
    assert any(not torch.equal(initial,dict(learner.named_parameters())[name])
               for name,initial in first_params.items())
    assert learner.actor_updates>0


def test_full_old_training_entry_rejects_unported_m2_warp_before_writing_run(tmp_path):
    from uav_network.experiments.training import train
    c=load_config(CONFIG)
    c.runtime.num_envs=1;c.runtime.device='cpu'
    c.runtime.output_dir=str(tmp_path/'should_not_be_created')
    c.tracking.enabled=False
    with pytest.raises(NotImplementedError,match='M2'):
        train(c)
    assert not (tmp_path/'should_not_be_created').exists()


def test_exact_run_actor_uses_rollout_batchnorm_not_layernorm_and_critic_is_plain():
    from uav_network.algorithms.models import RolloutBatchNorm1d
    c=load_config(CONFIG)
    learner=MATD3(c,255,1530,device='cpu')
    actor=learner.actors[0]
    assert any(isinstance(module,RolloutBatchNorm1d) for module in actor.encoder.modules())
    assert not any(isinstance(module,torch.nn.LayerNorm) for module in actor.encoder.modules())
    assert not any(isinstance(module,torch.nn.BatchNorm1d) for module in learner.critics[0].modules())
    obs=torch.rand(6,255)
    actor.train()
    bn=next(module for module in actor.encoder.modules() if isinstance(module,RolloutBatchNorm1d))
    before_mean=bn.running_mean.clone();before_count=bn.num_batches_tracked.clone()
    with torch.no_grad():
        actions,_=actor(torch.cat((obs,torch.nn.functional.one_hot(torch.arange(6),6).float()),-1),
                        torch.ones((6,7),dtype=torch.bool))
    assert actions.shape==(6,5)
    assert torch.equal(bn.running_mean,before_mean)
    assert torch.equal(bn.num_batches_tracked,before_count)
    assert all(not m.training for a in learner.target_actors for m in a.modules() if isinstance(m,RolloutBatchNorm1d))
    # A genuine Actor optimization forward does update BN statistics.
    obs=torch.rand(6,255)
    _=actor(torch.cat((obs,torch.nn.functional.one_hot(torch.arange(6),6).float()),-1),
            torch.ones((6,7),dtype=torch.bool))
    assert int(bn.num_batches_tracked)>int(before_count)
    learner.update_targets()
    target_bn=next(m for m in learner.target_actors[0].encoder.modules()
                   if isinstance(m,RolloutBatchNorm1d))
    torch.testing.assert_close(bn.running_mean,target_bn.running_mean)


def test_frozen_original_runner_archive_matches_sha_manifest():
    import json, zipfile
    folder=ROOT/'docs/reference_runs/18d756ef'
    manifest=json.loads((folder/'manifest.json').read_text())
    with zipfile.ZipFile(folder/'frozen_runner_reference.zip') as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest['files'])
        for name,digest in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest()==digest


def test_exact_18d756ef_auxiliary_algorithm_source_fingerprints():
    fingerprints = {
        "optimization.py": "1a092d1964246d55fe81a969e875206aae99e90c6c3b99de82ff53797e3193f6",
        "hybrid_replay.py": "64eb37f04e93a7877ee229a09650ccc668aaa1cacb9ce1d5dddc0f01122fd5b2",
        "layerwise_gradient_probe.py": "699b1ce84570db21f25efa8f9a945f4e56b48a532f09b31bc9a9e23e186f2ce6",
        "factorized_candidate.py": "6030cb509892499281493ff14d72f318203adea700021fe7506145116cfa6984",
    }
    for name,digest in fingerprints.items():
        assert hashlib.sha256((ROOT/"uav_network/algorithms"/name).read_bytes()).hexdigest()==digest
