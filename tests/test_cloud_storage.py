"""External storage credentials follow both GPU and CPU jobs without public secrets."""
import unittest
from uav_network.cloud.kernel import kernel_metadata,kernel_script
from uav_network.cloud.visualize import build_visualization_spec


class CloudStorageTests(unittest.TestCase):
    def test_private_storage_dataset_is_attached_to_gpu_and_cpu_bootstraps(self):
        base={'stage':'train','entity':'owner','project':'main','kernel_ref':'owner/job','commit':'a'*40,
              'credential_dataset':'owner/wandb','storage_credential_dataset':'owner/hf-private',
              'checkpoint_repo':'owner/checkpoints','account':'one','runs':{'matd3':{'id':'run1','name':'MATD3'}}}
        for stage in ('train','visualize'):
            spec=dict(base,stage=stage)
            self.assertIn('owner/hf-private',kernel_metadata(spec)['dataset_sources'])
            self.assertIn('HF_TOKEN',kernel_script(spec))
            self.assertEqual(kernel_metadata(spec)['enable_gpu'],stage=='train')
        result={'matd3':{'tracking_status':'finished','checkpoint_artifact':'owner/main/model:v1','episodes':10}}
        cpu=build_visualization_spec(base,result)
        self.assertEqual(cpu.get('storage_credential_dataset'),'owner/hf-private')
        self.assertEqual(cpu.get('checkpoint_repo'),'owner/checkpoints')
        self.assertIn('matd3',cpu.get('runs',{}))
        self.assertIn('CPU',cpu['runs']['matd3']['name'])
