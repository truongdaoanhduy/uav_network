"""Production execution flags enter real updates and survive complete resume."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

import torch

from test_baseline_resume import small_config, assert_nested_equal
from uav_network.configuration import Config, load_config
from uav_network.algorithms.checkpoints import make_learner
from uav_network.algorithms.optimization import compile_models
from uav_network.environment.observations import observation_dim, state_dim
from uav_network.experiments.training import train
from uav_network.experiments.continuation import restore_learner


def batch(config, device="cpu"):
    b,n=4,config.world.num_uavs
    obs=torch.randn(b,n,observation_dim(config),device=device)*.1
    obs[:,:,9]=1
    obs[:,:,-(n+1):]=1
    return {"observation":obs,"next_observation":obs.clone(),
            "state":torch.randn(b,state_dim(config),device=device)*.1,
            "next_state":torch.randn(b,state_dim(config),device=device)*.1,
            "continuous":torch.zeros(b,n,3,device=device),
            "destination":torch.zeros(b,n,dtype=torch.long,device=device),
            "reward":torch.ones(b,device=device)*.01,
            "terminated":torch.zeros(b,device=device)}


class ProductionExecutionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_full_requested_execution_flags_are_serialized_and_valid_for_both_schedules(self):
        for pql in (False,True):
            c=Config()
            c.runtime.amp_mode="fp16"
            c.runtime.compile_models=c.runtime.vectorize_critics=c.runtime.fused_optimizers=True
            c.pql.enabled=pql
            self.assertEqual(c.to_dict()["runtime"].get("amp_mode"),"fp16")
            self.assertIs(c.validate(),c)
        c.runtime.amp_mode="unknown"
        with self.assertRaisesRegex(ValueError,"amp_mode"):
            c.validate()

    def test_omitted_amp_runs_fp32_without_scalers_for_both_learners(self):
        for algorithm in ("masac", "matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                c = small_config(Path(directory), algorithm)
                learner = make_learner(c, observation_dim(c), state_dim(c))
                seen = []
                hook = learner.actors[0].encoder.register_forward_hook(
                    lambda mod, args, result: seen.append(result.dtype))
                try:
                    for _ in range(4):
                        metrics = learner.update(batch(c))
                finally:
                    hook.remove()
                self.assertEqual(c.runtime.amp_mode, "off")
                self.assertFalse(learner.precision.enabled)
                self.assertEqual(learner.precision.scalers, {})
                self.assertTrue(seen)
                self.assertEqual(set(seen), {torch.float32})
                for parameter in learner.parameters():
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(parameter).all())
                    if parameter.grad is not None:
                        self.assertEqual(parameter.grad.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertTrue(all(torch.isfinite(torch.as_tensor(value)).all()
                                    for value in metrics.values()))

    def test_real_both_learners_autocast_updates_and_keep_attempt_schedule(self):
        for algorithm in ("masac","matd3"):
            with self.subTest(algorithm=algorithm), TemporaryDirectory() as directory:
                c=small_config(Path(directory),algorithm)
                c.runtime.amp_mode="fp16"
                c.runtime.vectorize_critics=True
                learner=make_learner(c,observation_dim(c),state_dim(c))
                self.assertTrue(hasattr(learner,"precision"),"production learner must own AMP state")
                seen=[]
                hook=learner.actors[0].encoder.register_forward_hook(
                    lambda mod,args,result:seen.append((threading.current_thread().name,result.dtype)))
                try:
                    values=batch(c)
                    for _ in range(4):
                        learner.update(values)
                finally:
                    hook.remove()
                self.assertTrue(seen)
                self.assertTrue(all(dtype==torch.float16 for _,dtype in seen))
                state=learner.precision.state_dict()
                self.assertEqual(learner.updates,4)
                self.assertEqual(learner.actor_updates,4 if algorithm=="masac" else 2)
                self.assertEqual(state["counts"]["critic"]["attempted"],4)
                self.assertEqual(state["counts"]["actor"]["attempted"],learner.actor_updates)
                self.assertTrue(all(torch.isfinite(p).all() for p in learner.parameters()))

    def test_fixed_training_checkpoint_restores_scalers_and_rejects_missing_state(self):
        for algorithm in ("masac","matd3"):
            with self.subTest(algorithm=algorithm),TemporaryDirectory() as directory:
                c=small_config(Path(directory)/"first",algorithm)
                c.runtime.amp_mode="fp16"
                c.runtime.vectorize_critics=True
                result=train(c)
                saved=torch.load(result["resume_checkpoint"],weights_only=True)
                self.assertIn("precision",saved["training"],"resume must preserve AMP scale clocks")
                c.runtime.episodes=3
                c.runtime.warm_start_checkpoint=result["resume_checkpoint"]
                restored,_=restore_learner(c,"cpu")
                assert_nested_equal(self,restored.precision.state_dict(),saved["training"]["precision"])
                changed=deepcopy(saved);del changed["training"]["precision"]
                corrupt=Path(directory)/"missing.pt";torch.save(changed,corrupt)
                c.runtime.warm_start_checkpoint=str(corrupt)
                with self.assertRaisesRegex(ValueError,"(?i)precision|amp"):
                    restore_learner(c,"cpu")

    def test_outer_vectorized_compile_preserves_real_update_and_parameter_names(self):
        real_compile=torch.compile
        for algorithm in ("masac","matd3"):
            with self.subTest(algorithm=algorithm),TemporaryDirectory() as directory:
                c=small_config(Path(directory),algorithm)
                c.runtime.vectorize_critics=True
                reference=make_learner(c,observation_dim(c),state_dim(c))
                actual=make_learner(c,observation_dim(c),state_dim(c))
                actual.load_state_dict(reference.state_dict())
                keys=set(actual.state_dict())
                graphs=[]
                def backend(gm,example_inputs):
                    graphs.append(gm)
                    return gm.forward
                def compile_test(function=None,**kwargs):
                    kwargs["backend"]=backend
                    return real_compile(function,**kwargs)
                with patch("torch.compile",side_effect=compile_test):
                    compile_models(actual)
                    data=batch(c)
                    for step in range(2):
                        torch.manual_seed(44+step);reference.update(data)
                        torch.manual_seed(44+step);actual.update(data)
                self.assertTrue(graphs,"actual TorchDynamo graphs must execute")
                self.assertEqual(set(actual.state_dict()),keys)
                for name,value in reference.state_dict().items():
                    torch.testing.assert_close(actual.state_dict()[name],value,rtol=1e-5,atol=1e-6)


    def test_async_pql_worker_autocast_and_scaler_checkpoint_ownership(self):
        from test_pql_system import pql_config, transition
        from test_split_updates import replay_batch
        from uav_network.experiments.pql_system import PQLSystem
        for algorithm in ("masac","matd3"):
            with self.subTest(algorithm=algorithm),TemporaryDirectory() as directory:
                c=pql_config(algorithm,"async")
                c.runtime.amp_mode="fp16"
                c.runtime.vectorize_critics=True
                c.pql.checkpoint_replay=True
                engine=PQLSystem(c)
                seen=[];hooks=[]
                try:
                    self.assertTrue(hasattr(engine.critic,"precision"),"PQL must own worker-local AMP")
                    for learner in (engine.critic,engine.policy):
                        actors = learner.target_actors if algorithm == "matd3" and learner is engine.critic else learner.actors
                        hooks.append(actors[0].encoder.register_forward_hook(
                            lambda mod,args,result:seen.append((threading.current_thread().name,result.dtype))))
                    values=replay_batch(engine.critic)
                    for _ in range(3):engine.submit(transition(values))
                    engine.flush()
                    for name in ("pql-critic","pql-policy"):
                        rows=[dtype for thread,dtype in seen if thread==name]
                        self.assertTrue(rows,name)
                        self.assertTrue(all(dtype==torch.float16 for dtype in rows),name)
                    path=Path(directory)/"pql.pt"
                    progress={"episodes":0,"global_step":12,"collection_steps":3,"next_seed":44,"records":[]}
                    engine.save(path,progress)
                    saved=torch.load(path,weights_only=True)
                    self.assertIn("precision",saved["training"])
                    states=deepcopy(saved["training"]["precision"])
                finally:
                    for hook in hooks:hook.remove()
                    engine.close()
                resumed=PQLSystem(c,checkpoint=path)
                try:
                    assert_nested_equal(self,resumed.critic.precision.state_dict(),states["critic"])
                    assert_nested_equal(self,resumed.policy.precision.state_dict(),states["policy"])
                finally:resumed.close()
                del saved["training"]["precision"]
                torch.save(saved,path)
                with self.assertRaisesRegex(ValueError,"(?i)precision|amp"):
                    PQLSystem(c,checkpoint=path)

    def test_legacy_policy_warm_start_uses_requested_amp_mode(self):
        from uav_network.algorithms.checkpoints import save_policy
        with TemporaryDirectory() as directory:
            c=small_config(Path(directory),"matd3")
            learner=make_learner(c,observation_dim(c),state_dim(c))
            path=Path(directory)/"policy.pt"
            save_policy(learner,path,dict(episodes=0,global_step=0,collection_steps=0,next_seed=44,records=[]))
            c.runtime.amp_mode="fp16"
            c.runtime.warm_start_checkpoint=str(path)
            restored,progress=restore_learner(c,"cpu")
            self.assertEqual(restored.precision.mode,"fp16")
            self.assertEqual(progress["state_restore"],"policy_warm_start")
            assert_nested_equal(self,restored.state_dict(),learner.state_dict())

    def test_stage_metrics_do_not_overwrite_other_pql_optimizer_groups(self):
        with TemporaryDirectory() as directory:
            c=small_config(Path(directory),"masac")
            c.runtime.amp_mode="fp16"
            learner=make_learner(c,observation_dim(c),state_dim(c))
            data=batch(c)
            critic=learner.update_critic(data)
            actor=learner.update_actor(data)
            self.assertIn("amp/critic/attempted",critic)
            self.assertNotIn("amp/actor/attempted",critic)
            self.assertNotIn("amp/alpha/attempted",critic)
            self.assertNotIn("amp/critic/attempted",actor)
            self.assertEqual({**critic,**actor}["amp/critic/attempted"],1)
