"""A real CPU evaluation is visible online and closes its run on failure."""
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from uav_network.configuration import Config
from uav_network.experiments.training import train
from uav_network.cloud.visualize import run_visualization


class VisualizationTrackingTests(unittest.TestCase):
    def test_cpu_job_logs_online_in_main_project_and_finishes_failures(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);c=Config();c.runtime.output_dir=str(root/'train')
            c.runtime.algorithm='matd3';c.runtime.device=c.runtime.replay_device='cpu'
            c.runtime.episodes=1;c.world.max_steps=3;c.world.num_targets=1;c.world.num_obstacles=0
            c.matd3.hidden_sizes=[8,8];c.matd3.batch_size=2;c.matd3.replay_capacity=8;c.matd3.learning_starts=2
            trained=train(c);checkpoint=trained['checkpoint']
            for failure in (False,True):
                logs=[];finished=[];initializations=[]
                run=SimpleNamespace(offline=False,id='cpu1',entity='owner',project='main',url='https://wandb.ai/owner/main/runs/cpu1',
                    summary={},save=lambda *a,**k:None,define_metric=lambda *a,**k:None,
                    use_artifact=lambda *a,**k:None,log=lambda v:logs.append(v),finish=lambda exit_code:finished.append(exit_code))
                def init(**kwargs):initializations.append(kwargs);return run
                sdk=SimpleNamespace(init=init,Settings=lambda **kw:kw)
                spec={'entity':'owner','project':'main','config':str(root/'train/resolved_config.yaml'),'overrides':[],
                    'kernel_ref':'owner/cpu','parent_kernel_ref':'owner/gpu','training_mode':'base',
                    'runs':{'matd3':{'id':'cpu1','name':'MATD3-CPU-visualize'}},
                    'artifacts':{'matd3':'owner/main/model:v1'},'training_status':{'matd3':{'episodes':1}},
                    'cpu_eval':{'episodes':1,'seed':10044,'frame_stride':2,'fps':5}}
                with patch.dict(sys.modules,{'wandb':sdk}), \
                     patch('uav_network.experiments.tracking._authentication',return_value=('owner','main')), \
                     patch('uav_network.experiments.dashboard.ensure_project_workspace',return_value='https://wandb.ai/owner/main/workspace'), \
                     patch('uav_network.cloud.remote.prepare_checkpoints',return_value={'matd3':checkpoint}), \
                     patch('uav_network.cloud.visualize.publish_visualization_media',return_value={}):
                    if failure:
                        with patch('uav_network.cloud.visualize.evaluate',side_effect=ValueError('render failed')):
                            with self.assertRaises(RuntimeError):run_visualization(spec,root/'failed')
                    else:
                        result=run_visualization(spec,root/'success')
                        self.assertEqual(result['matd3']['device'],'cpu')
                self.assertEqual(len(initializations),1,'CPU jobs must open an online run')
                self.assertEqual(initializations[0]['mode'],'online')
                self.assertEqual(initializations[0]['project'],'main')
                self.assertEqual(finished,[1 if failure else 0])
                if not failure:self.assertTrue(any(v.get('evaluation/episodes_completed')==1 for v in logs))
