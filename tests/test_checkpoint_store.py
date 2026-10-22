"""Real checkpoint bytes/checksums with only Hub transport replaced."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from uav_network.configuration import Config


class CheckpointStoreTests(unittest.TestCase):
    def test_explicit_env_credentials_do_not_parse_unneeded_malformed_key_file(self):
        from uav_network.experiments import checkpoint_store as store
        with TemporaryDirectory() as directory:
            broken = Path(directory) / "key.txt"
            broken.write_text('{"huggingface": {"token": "fallback",},}')
            config = Config()
            config.tracking.credentials_file = str(broken)
            config.tracking.checkpoint_repo = "owner/explicit"
            with patch.dict(os.environ, {"HF_TOKEN": "env-token", "HF_REPO_ID": "owner/env"}, clear=False):
                token, repo = store.storage_credentials(config)
                self.assertEqual((token, repo), ("env-token", "owner/explicit"))
                token_only, _ = store.storage_credentials(None, require_repo=False)
                self.assertEqual(token_only, "env-token")

    def test_public_checkpoint_store_is_allowed_when_explicitly_configured(self):
        from uav_network.experiments import checkpoint_store as store
        created = []
        class Hub:
            def __init__(self, **kwargs): pass
            def create_repo(self, **kwargs): created.append(kwargs)
            def repo_info(self, *args, **kwargs): return SimpleNamespace(private=False)
        sdk = SimpleNamespace(HfApi=Hub)
        config = Config()
        config.tracking.checkpoint_repo = 'owner/public-checkpoints'
        config.tracking.checkpoint_repo_private = False
        with patch.dict(sys.modules, {'huggingface_hub': sdk}), \
             patch.dict(os.environ, {'HF_TOKEN': 'secret'}, clear=False):
            _, repo = store.ensure_private_store(config)
        self.assertEqual(repo, 'owner/public-checkpoints')
        self.assertEqual(created[0]['private'], False)

    def test_identical_private_snapshot_is_reused_without_a_new_commit(self):
        from uav_network.experiments import checkpoint_store as store
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'checkpoint.pt'; source.write_bytes(b'policy')
            remote = SimpleNamespace(private=True, sha='b'*40, siblings=[SimpleNamespace(
                rfilename='archive/run/v1/checkpoint.pt', size=source.stat().st_size,
                lfs=SimpleNamespace(sha256=store.file_sha256(source)))])
            hub = SimpleNamespace(repo_info=Mock(return_value=remote),
                                  create_commit=Mock(side_effect=AssertionError('duplicate commit')))
            config = Config()
            config.tracking.checkpoint_repo_private = True
            with patch.object(store, 'ensure_private_store', return_value=(hub, 'owner/private')):
                manifest = store.upload_checkpoint_files(config, {'checkpoint.pt':source}, 'archive/run/v1')
            self.assertEqual(manifest['revision'], 'b'*40)
            hub.create_commit.assert_not_called()

    def test_same_size_changed_content_requires_a_verified_new_commit(self):
        from uav_network.experiments import checkpoint_store as store
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'checkpoint.pt'; source.write_bytes(b'new')
            def snapshot(digest, revision):
                return SimpleNamespace(private=True, sha=revision, siblings=[SimpleNamespace(
                    rfilename='archive/run/v1/checkpoint.pt', size=3,
                    lfs=SimpleNamespace(sha256=digest))])
            hub = SimpleNamespace(repo_info=Mock(side_effect=[snapshot('0'*64, 'b'*40),
                snapshot(store.file_sha256(source), 'c'*40)]),
                create_commit=Mock(return_value=SimpleNamespace(oid='c'*40)))
            config = Config()
            config.tracking.checkpoint_repo_private = True
            with patch.object(store, 'ensure_private_store', return_value=(hub, 'owner/private')):
                manifest = store.upload_checkpoint_files(config, {'checkpoint.pt':source}, 'archive/run/v1')
            self.assertEqual(manifest['revision'], 'c'*40)
            hub.create_commit.assert_called_once()

    def test_private_immutable_roundtrip_and_corruption_rejection(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_network.experiments.checkpoint_store'),
                             'external checkpoint transport is missing')
        store = importlib.import_module('uav_network.experiments.checkpoint_store')
        with TemporaryDirectory() as directory:
            root = Path(directory); source = root/'policy.pt'; source.write_bytes(b'actual-policy-state')
            resume = root/'resume.pt'; resume.write_bytes(b'optimizer-replay-and-rng')
            remote = {}; private = [True]; corrupt = [False]
            class Hub:
                def create_repo(self, **kwargs):
                    if not kwargs['private']: raise AssertionError('must stay private')
                def create_commit(self, **kwargs):
                    for op in kwargs['operations']: remote[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
                    return SimpleNamespace(oid='a'*40)
                def repo_info(self, *args, **kwargs):
                    return SimpleNamespace(private=private[0], siblings=[SimpleNamespace(rfilename=k,size=len(v),
                        lfs=SimpleNamespace(sha256=hashlib.sha256(v).hexdigest())) for k,v in remote.items()])
            def download(repo_id, filename, revision, local_dir, **kwargs):
                self.assertEqual(revision,'a'*40)
                p=Path(local_dir)/filename;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(remote[filename]+(b'corrupt'if corrupt[0]else b''));return str(p)
            sdk=SimpleNamespace(HfApi=lambda **kwargs:Hub(),CommitOperationAdd=lambda **kwargs:SimpleNamespace(**kwargs),hf_hub_download=download)
            c=Config();c.tracking.credentials_file=str(root/'absent.txt')
            c.tracking.checkpoint_repo_private = True
            with patch.dict(sys.modules,{'huggingface_hub':sdk}), patch.dict(os.environ,{'HF_TOKEN':'not-logged-secret','HF_REPO_ID':'owner/private-checkpoints'}):
                manifest=store.upload_checkpoint_files(c,{'checkpoint.pt':source,'resume.pt':resume},'run-1/step-10')
                self.assertEqual(manifest['revision'],'a'*40)
                self.assertNotIn('not-logged-secret',json.dumps(manifest))
                m=root/'checkpoint.json';m.write_text(json.dumps(manifest))
                class Artifact:
                    manifest=SimpleNamespace(entries={'checkpoint.json':{}})
                    def get_entry(self,name):
                        assert name=='checkpoint.json'
                        return SimpleNamespace(download=lambda root:str(m))
                path=store.download_checkpoint_file(Artifact(),'resume.pt',root/'restore')
                self.assertEqual(Path(path).read_bytes(),resume.read_bytes())
                corrupt[0]=True
                with self.assertRaisesRegex(ValueError,'checksum|size'):
                    store.download_checkpoint_file(Artifact(),'checkpoint.pt',root/'corrupt')
                corrupt[0]=False;private[0]=False
                with self.assertRaisesRegex(RuntimeError,'private'):
                    store.upload_checkpoint_files(c,{'checkpoint.pt':source},'run-2')
                for key,value in [('revision','main'),('path','../secret.pt')]:
                    bad=copy.deepcopy(manifest)
                    if key=='revision':bad[key]=value
                    else:bad['files']['checkpoint.pt']['path']=value
                    m.write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):store.download_checkpoint_file(Artifact(),'checkpoint.pt',root/'unsafe')

    def test_legacy_binary_artifact_still_downloads_without_hf(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_network.experiments.checkpoint_store'),
                             'external checkpoint transport is missing')
        store=importlib.import_module('uav_network.experiments.checkpoint_store')
        with TemporaryDirectory() as d:
            p=Path(d)/'resume.pt';p.write_bytes(b'legacy-full-state')
            artifact=SimpleNamespace(manifest=SimpleNamespace(entries={'resume.pt':{}}),
                get_entry=lambda name:SimpleNamespace(download=lambda root:str(p)))
            self.assertEqual(Path(store.download_checkpoint_file(artifact,'resume.pt',Path(d)/'inputs')).read_bytes(),b'legacy-full-state')

    def test_config_rejects_unsafe_explicit_checkpoint_repo(self):
        config = Config()
        config.tracking.checkpoint_repo = '../other/repo'
        with self.assertRaisesRegex(ValueError, 'checkpoint_repo'):
            config.validate()

    def test_archived_legacy_reference_resolves_through_original_run(self):
        store=importlib.import_module('uav_network.experiments.checkpoint_store')
        self.assertTrue(callable(getattr(store,'resolve_checkpoint_artifact',None)), 'archived references need a reader')
        old='owner/project/uav-network-matd3-12345678:v0';new='owner/project/uav-network-matd3-12345678:v7'
        saved=SimpleNamespace(qualified_name=new)
        def artifact(ref, **kwargs):
            if ref==old:raise ValueError('legacy version archived')
            self.assertEqual(ref,new);return saved
        key = 'checkpoint_archive_uav-network-matd3-12345678_v0'
        for summary in ({key:new}, {'checkpoint_archives.'+old:new}, {'checkpoint_archives':{old:new}}):
            api=SimpleNamespace(artifact=artifact,run=lambda ref:SimpleNamespace(summary=summary),flush=lambda:None)
            self.assertIs(store.resolve_checkpoint_artifact(api,old),saved)
