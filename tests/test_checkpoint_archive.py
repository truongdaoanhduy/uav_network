"""Deletion is gated on full backup verification and active-dependency checks."""
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from uav_network.configuration import Config


class CheckpointArchiveTests(unittest.TestCase):
    def test_completion_proof_is_bound_to_owned_pipeline_and_actual_cpu_output(self):
        from uav_network.cloud.archive import verify_completed_source
        gpu = 'owner/training'; cpu = 'owner/training-cpu'
        ref = 'owner/main/uav-network-matd3-12345678:v3'
        proof = {'gpu_kernel_ref': gpu, 'root_kernel_ref': gpu,
                 'cpu_kernel_ref': cpu, 'artifact_ref': ref}
        spec = {'entity':'owner', 'project':'main', 'kernel_ref':'owner/archive',
                'completed_sources':{'12345678':proof}}
        source = SimpleNamespace(state='finished', config={'kaggle_kernel':gpu},
                                 summary={'episode_budget_complete':True})
        api = SimpleNamespace(flush=lambda:None, run=lambda path:source)
        job = {'stage':'visualize', 'parent_kernel_ref':gpu, 'artifacts':{'matd3':ref}}
        result = {'matd3':{'device':'cpu', 'episodes':2, 'checkpoint_artifact':ref}}
        def download(args, account, timeout):
            folder = Path(args[args.index('-p')+1])
            for name, data in [('job.json',job), ('visualization-result.json',result),
                               ('hardware.json',{'cuda_available':False,'cuda_devices':[]})]:
                (folder/name).write_text(json.dumps(data))
        with patch('uav_network.cloud.client.kernel_status', return_value='complete') as status, \
             patch('uav_network.cloud.client.run_kaggle', side_effect=download):
            verify_completed_source(api, spec, '12345678', object(), verify_outputs=True)
            self.assertEqual({c.args[0]for c in status.call_args_list},{gpu,cpu})
            for key, value in [('gpu_kernel_ref','other/training'),
                               ('root_kernel_ref','owner/unrelated'),
                               ('cpu_kernel_ref','owner/unrelated-cpu'),
                               ('artifact_ref','owner/main/uav-network-matd3-87654321:v3')]:
                original = proof[key]; proof[key] = value
                with self.assertRaisesRegex(RuntimeError, 'proof'):
                    verify_completed_source(api, spec, '12345678', object())
                proof[key] = original
            result['matd3']['checkpoint_artifact'] = 'owner/main/unrelated:v1'
            with self.assertRaisesRegex(RuntimeError, 'outputs'):
                verify_completed_source(api, spec, '12345678', object(), verify_outputs=True)
        with patch('uav_network.cloud.client.kernel_status', return_value='pending'):
            with self.assertRaisesRegex(RuntimeError, 'not completed'):
                verify_completed_source(api, spec, '12345678', object())

    def test_archive_requires_bound_completion_proof_before_online_side_effects(self):
        from uav_network.cloud import archive as module
        spec = {'entity': 'owner', 'project': 'main', 'checkpoint_repo': 'owner/private',
                'account': 'one', 'kernel_ref': 'owner/archive',
                'runs': {'archive': {'id': 'archive1', 'name': 'archive'}},
                'archive_candidates': ['owner/main/uav-network-matd3-12345678:v0']}
        with TemporaryDirectory() as d, patch.object(module, 'OnlineTracker') as tracker:
            with self.assertRaisesRegex(RuntimeError, 'completion proof'):
                module.run_archive(spec, Path(d))
            tracker.assert_not_called()

    def test_producer_restarted_during_backup_is_refreshed_and_never_deleted(self):
        from uav_network.cloud.archive import archive_artifact
        from uav_network.experiments.checkpoint_store import file_sha256
        with TemporaryDirectory() as d:
            root = Path(d); source = root / 'checkpoint.pt'; source.write_bytes(b'checkpoint')
            actual_state = ['finished']; cache = {}; events = []
            def run(path):
                if path not in cache:
                    cache[path] = SimpleNamespace(state=actual_state[0], summary={}, update=lambda:None)
                return cache[path]
            artifact = SimpleNamespace(manifest=SimpleNamespace(entries={'checkpoint.pt': {}}),
                metadata={}, aliases=[], used_by=lambda:[], download=lambda **kw:d,
                delete=lambda **kw:events.append('deleted'))
            api = SimpleNamespace(run=run, flush=lambda:cache.clear(), artifact=lambda *a, **kw:artifact)
            descriptor = {'schema':1, 'backend':'huggingface', 'repo_id':'owner/private',
                'revision':'a'*40, 'files':{'checkpoint.pt':{'path':'archive/checkpoint.pt',
                'size':source.stat().st_size, 'sha256':file_sha256(source)}}}
            def upload(*args, **kwargs):
                actual_state[0] = 'running'
                return descriptor
            saved = SimpleNamespace(wait=lambda **kw:None,
                qualified_name='owner/main/uav-network-matd3-12345678:v9')
            tracker = SimpleNamespace(run=SimpleNamespace(log_artifact=lambda *a, **kw:saved))
            sdk = SimpleNamespace(Artifact=lambda *a, **kw:SimpleNamespace(add_file=lambda *a, **kw:None))
            with patch.dict(sys.modules, {'wandb':sdk}), \
                 patch('uav_network.cloud.archive.upload_checkpoint_files', side_effect=upload), \
                 patch('uav_network.cloud.archive.download_manifest_file', return_value=str(source)):
                with self.assertRaisesRegex(RuntimeError, 'active|finished'):
                    archive_artifact(api, tracker, Config(),
                        'owner/main/uav-network-matd3-12345678:v0', root, set(), lambda:None)
            self.assertEqual(events, [])

    def test_failed_backup_or_running_dependency_never_deletes_source(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_network.cloud.archive'),'verified archive migration is missing')
        from uav_network.cloud.archive import archive_artifact
        with TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.pt';p.write_bytes(b'checkpoint')
            deleted=[];published=[]
            run=SimpleNamespace(state='finished',summary={},update=lambda:None)
            artifact=SimpleNamespace(id='source-id',manifest=SimpleNamespace(entries={'checkpoint.pt':{}}),
                metadata={},aliases=['latest','final'],used_by=lambda:[],
                download=lambda **kw:d,delete=lambda **kw:deleted.append(True))
            api=SimpleNamespace(artifact=lambda *a,**kw:artifact,run=lambda *a:run,flush=lambda:None)
            tracker=SimpleNamespace(run=SimpleNamespace(log_artifact=lambda *a,**kw:published.append(True)))
            with patch('uav_network.cloud.archive.upload_checkpoint_files',side_effect=RuntimeError('backup failed')):
                with self.assertRaises(RuntimeError):archive_artifact(api,tracker,Config(),'owner/main/uav-network-matd3-12345678:v0',Path(d),set(),lambda:None)
            self.assertEqual(deleted,[]);self.assertEqual(published,[])
            artifact.used_by=lambda:[SimpleNamespace(state='running')]
            with self.assertRaisesRegex(RuntimeError,'active'):
                archive_artifact(api,tracker,Config(),'owner/main/uav-network-matd3-12345678:v0',Path(d),set(),lambda:None)
            self.assertEqual(deleted,[])
            with self.assertRaisesRegex(RuntimeError,'protected'):
                archive_artifact(api,tracker,Config(),'owner/main/uav-network-matd3-12345678:v0',Path(d),{'12345678'},lambda:None)

    def test_verified_fresh_restore_publishes_mapping_before_source_delete(self):
        from uav_network.cloud.archive import archive_artifact
        from uav_network.experiments.checkpoint_store import file_sha256
        with TemporaryDirectory() as d:
            root=Path(d); source=root/'checkpoint.pt'; source.write_bytes(b'checkpoint-bytes')
            events=[]
            source_run=SimpleNamespace(state='finished',summary={},update=lambda:events.append('run-update'))
            artifact=SimpleNamespace(manifest=SimpleNamespace(entries={'checkpoint.pt':{}}),metadata={},
                aliases=['latest','final'],used_by=lambda:[],download=lambda **kw:d,
                delete=lambda **kw:events.append('delete'))
            api=SimpleNamespace(artifact=lambda *a,**kw:artifact,run=lambda *a:source_run,flush=lambda:None)
            descriptor={'schema':1,'backend':'huggingface','repo_id':'owner/private','revision':'a'*40,
                'files':{'checkpoint.pt':{'path':'archive/checkpoint.pt','size':source.stat().st_size,
                                         'sha256':file_sha256(source)}}}
            saved=SimpleNamespace(qualified_name='owner/main/uav-network-matd3-12345678:v9',
                                  wait=lambda timeout:events.append('published'))
            tracker=SimpleNamespace(run=SimpleNamespace(log_artifact=lambda *a,**kw:saved))
            replacement=SimpleNamespace(add_file=lambda *a,**kw:None)
            sdk=SimpleNamespace(Artifact=lambda *a,**kw:replacement)
            with patch.dict(sys.modules,{'wandb':sdk}), \
                 patch('uav_network.cloud.archive.upload_checkpoint_files',side_effect=lambda *a,**kw:(events.append('upload') or descriptor)), \
                 patch('uav_network.cloud.archive.download_manifest_file',side_effect=lambda *a,**kw:(events.append('fresh-download') or str(source))):
                result=archive_artifact(api,tracker,Config(),'owner/main/uav-network-matd3-12345678:v0',root,set(),lambda:None)
            self.assertEqual(result['status'],'archived')
            self.assertEqual(source_run.summary['checkpoint_archives']['owner/main/uav-network-matd3-12345678:v0'],saved.qualified_name)
            self.assertLess(events.index('fresh-download'),events.index('published'))
            self.assertLess(events.index('published'),events.index('delete'))

    def test_inventory_marks_only_finished_unprotected_unused_legacy_versions_as_candidates(self):
        from uav_network.cloud.archive import inventory_legacy_artifacts
        def artifact(ref, entries, consumers=()):
            return SimpleNamespace(qualified_name=ref, manifest=SimpleNamespace(entries={n:{} for n in entries}),
                                   used_by=lambda:list(consumers))
        eligible=artifact('owner/main/uav-network-matd3-12345678:v0',['checkpoint.pt','resume.pt'])
        active=artifact('owner/main/uav-network-masac-87654321:v2',['checkpoint.pt'],[SimpleNamespace(state='running')])
        external=artifact('owner/main/uav-network-matd3-aaaaaaaa:v3',['checkpoint.json'])
        protected=artifact('owner/main/uav-network-masac-bbbbbbbb:v1',['checkpoint.pt'])
        runs=[SimpleNamespace(id='12345678',state='finished',logged_artifacts=lambda:[eligible]),
              SimpleNamespace(id='87654321',state='finished',logged_artifacts=lambda:[active]),
              SimpleNamespace(id='aaaaaaaa',state='finished',logged_artifacts=lambda:[external]),
              SimpleNamespace(id='bbbbbbbb',state='finished',logged_artifacts=lambda:[protected])]
        api=SimpleNamespace(runs=lambda path:runs)
        records=inventory_legacy_artifacts(api,'owner','main',{'bbbbbbbb'})
        by_ref={r['ref']:r for r in records}
        self.assertTrue(by_ref[eligible.qualified_name]['candidate'])
        self.assertFalse(by_ref[active.qualified_name]['candidate'])
        self.assertEqual(by_ref[external.qualified_name]['status'],'already_external')
        self.assertEqual(by_ref[protected.qualified_name]['status'],'protected')

    def test_archive_runner_refuses_requested_ref_not_safe_in_fresh_inventory(self):
        from uav_network.cloud import archive as module
        ref='owner/main/uav-network-matd3-12345678:v0'
        artifact=SimpleNamespace(qualified_name=ref,manifest=SimpleNamespace(entries={'checkpoint.pt':{}}),
                                 used_by=lambda:[SimpleNamespace(state='running')])
        source=SimpleNamespace(id='12345678',state='finished',logged_artifacts=lambda:[artifact])
        api=SimpleNamespace(runs=lambda path:[source])
        run=SimpleNamespace(summary={},log=lambda *a,**k:None,save=lambda *a,**k:None)
        tracker=SimpleNamespace(run=run,close=lambda failed=False:None)
        spec={'entity':'owner','project':'main','checkpoint_repo':'owner/private','account':'one',
              'runs':{'archive':{'id':'archive1','name':'archive'}},'archive_candidates':[ref]}
        with TemporaryDirectory() as d, patch.dict(sys.modules,{'wandb':SimpleNamespace(Api=lambda:api)}), \
             patch.object(module,'OnlineTracker',return_value=tracker), \
             patch.object(module,'archive_artifact',side_effect=AssertionError('unsafe ref reached migration')):
            with self.assertRaisesRegex(RuntimeError,'completion proof'):
                module.run_archive(spec,Path(d))
