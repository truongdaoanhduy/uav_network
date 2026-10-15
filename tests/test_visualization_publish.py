"""Visualization media is published to public Hub storage and W&B independently."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from uav_network.configuration import Config


class VisualizationPublishTests(unittest.TestCase):
    def module(self):
        from uav_network.cloud import visualize
        self.assertTrue(callable(getattr(visualize, 'publish_visualization_assets', None)),
                        'public visualization asset publisher is missing')
        self.assertTrue(callable(getattr(visualize, 'log_visualization_media', None)),
                        'W&B media logger is missing')
        self.assertTrue(callable(getattr(visualize, 'publish_visualization_media', None)),
                        'independent visualization publication coordinator is missing')
        return visualize

    def fixture(self, root):
        folder = Path(root) / 'visualize' / 'masac'
        folder.mkdir(parents=True)
        png = folder / 'final_episode.png'; png.write_bytes(b'png-bytes')
        video = folder / 'final_episode.mp4'; video.write_bytes(b'video-bytes')
        metrics = folder / 'metrics.json'; metrics.write_text(json.dumps({'mean_return': 12.3}))
        result = {'algorithm': 'masac', 'episodes': 30, 'render_path': str(png),
                  'animation_path': str(video), 'mean_return': 12.3}
        return result, png, video, metrics

    def test_public_hf_publisher_uploads_and_freshly_verifies_video_png_metrics(self):
        module = self.module()
        with TemporaryDirectory() as directory:
            result, png, video, metrics = self.fixture(directory)
            remote = {}
            class Hub:
                def __init__(self, **kwargs): pass
                def create_repo(self, **kwargs):
                    self.created = kwargs
                def repo_info(self, *args, **kwargs):
                    return SimpleNamespace(private=False)
                def create_commit(self, **kwargs):
                    for op in kwargs['operations']:
                        remote[op.path_in_repo] = Path(op.path_or_fileobj).read_bytes()
                    return SimpleNamespace(oid='a' * 40)
            def download(repo_id, filename, revision, local_dir, **kwargs):
                self.assertEqual(repo_id, 'owner/public-media')
                self.assertEqual(revision, 'a' * 40)
                destination = Path(local_dir) / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(remote[filename])
                return str(destination)
            sdk = SimpleNamespace(HfApi=Hub,
                CommitOperationAdd=lambda **kwargs: SimpleNamespace(**kwargs),
                hf_hub_download=download)
            config = Config()
            config.tracking.checkpoint_repo = 'owner/public-media'
            config.tracking.checkpoint_repo_private = False
            config.tracking.credentials_file = str(Path(directory) / 'absent.json')
            spec = {'checkpoint_repo': 'owner/public-media', 'training_mode': 'base'}
            with patch.dict(sys.modules, {'huggingface_hub': sdk}), \
                 patch.dict(os.environ, {'HF_TOKEN': 'secret-token'}, clear=False):
                published = module.publish_visualization_assets(
                    config, spec, 'masac', 'run12345', result)
            self.assertEqual(published['repo_id'], 'owner/public-media')
            self.assertEqual(published['revision'], 'a' * 40)
            self.assertEqual(set(published['files']), {'video', 'image', 'metrics'})
            self.assertTrue(all(item['path'].startswith('visualizations/base/masac/run12345/')
                                for item in published['files'].values()))
            for item in published['files'].values():
                self.assertEqual(item['sha256'], hashlib.sha256(remote[item['path']]).hexdigest())
                self.assertIn('/resolve/' + 'a' * 40 + '/', item['url'])

    def test_wandb_media_logger_logs_video_image_and_public_hf_links(self):
        module = self.module()
        with TemporaryDirectory() as directory:
            result, png, video, _ = self.fixture(directory)
            logged = []
            run = SimpleNamespace(log=lambda payload: logged.append(payload), summary={})
            fake_wandb = SimpleNamespace(
                Video=lambda path, fps, format: ('video', path, fps, format),
                Image=lambda path, caption=None: ('image', path, caption))
            published = {'repo_id': 'owner/public-media', 'revision': 'b' * 40,
                'files': {
                    'video': {'url': 'https://hf/video'},
                    'image': {'url': 'https://hf/image'},
                    'metrics': {'url': 'https://hf/metrics'},
                }}
            with patch.dict(sys.modules, {'wandb': fake_wandb}):
                module.log_visualization_media(run, result, published, fps=10)
            self.assertEqual(len(logged), 1)
            self.assertIn('visualization/video', logged[0])
            self.assertIn('visualization/final_episode', logged[0])
            self.assertEqual(logged[0]['global_step'], 30)
            self.assertEqual(run.summary['visualization/hf_video_url'], 'https://hf/video')
            self.assertEqual(run.summary['visualization/hf_image_url'], 'https://hf/image')
            self.assertEqual(run.summary['visualization/hf_revision'], 'b' * 40)

    def test_publication_failures_are_isolated_between_hf_and_wandb(self):
        module = self.module()
        with TemporaryDirectory() as directory:
            result, *_ = self.fixture(directory)
            config = Config(); spec = {'training_mode': 'pql'}
            run = SimpleNamespace(summary={})
            with patch.object(module, 'publish_visualization_assets', side_effect=RuntimeError('hub down')), \
                 patch.object(module, 'log_visualization_media') as log_media:
                publication = module.publish_visualization_media(
                    config, spec, 'masac', 'cpu12345', result, run, fps=10)
            self.assertEqual(publication['huggingface']['status'], 'failed')
            self.assertEqual(publication['huggingface']['error_type'], 'RuntimeError')
            self.assertEqual(publication['wandb']['status'], 'ok')
            log_media.assert_called_once()


    def test_posthoc_publisher_remaps_kaggle_paths_and_resumes_existing_cpu_run(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_network.cloud.media_publish'),
                             'post-hoc publisher for already-pinned CPU jobs is missing')
        module = importlib.import_module('uav_network.cloud.media_publish')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'visualize' / 'masac'; folder.mkdir(parents=True)
            (folder / 'final_episode.png').write_bytes(b'png')
            (folder / 'final_episode.mp4').write_bytes(b'video')
            (folder / 'metrics.json').write_text('{}')
            spec = {'config': 'configs/comparison-20k.yaml', 'overrides': ['pql.enabled=false'],
                    'entity': 'owner', 'project': 'project', 'training_mode': 'base',
                    'checkpoint_repo': 'owner/public-media', 'cpu_eval': {'fps': 10}}
            (root / 'job.json').write_text(json.dumps(spec))
            (root / 'visualization-result.json').write_text(json.dumps({'masac': {
                'algorithm': 'masac', 'episodes': 30, 'wandb_run_id': 'cpu12345',
                'render_path': '/kaggle/working/visualize/masac/final_episode.png',
                'animation_path': '/kaggle/working/visualize/masac/final_episode.mp4'}}))
            key = root / 'key.txt'
            key.write_text(json.dumps({'kaggle': {'accounts': [{'name': 'one', 'token': 'k'}]},
                'wandb': {'api_key': 'w', 'entity': 'owner', 'project': 'project'},
                'huggingface': {'token': 'h', 'repo_id': 'owner/public-media'}}))
            calls = []
            class Run:
                summary = {}
                def __enter__(self): return self
                def __exit__(self, *args): return False
            fake_wandb = SimpleNamespace(init=lambda **kwargs: calls.append(kwargs) or Run())
            config = Config(); config.tracking.checkpoint_repo = 'owner/public-media'; config.tracking.checkpoint_repo_private = False
            captured = {}
            def publish(cfg, child, algorithm, run_id, result, run, fps):
                captured.update(result=result.copy(), algorithm=algorithm, run_id=run_id, fps=fps)
                return {'huggingface': {'status': 'ok'}, 'wandb': {'status': 'ok'}}
            with patch.dict(sys.modules, {'wandb': fake_wandb}), \
                 patch.object(module, 'load_config', return_value=config), \
                 patch.object(module, 'publish_visualization_media', side_effect=publish):
                published = module.publish_downloaded_visualizations(root, key)
            self.assertEqual(calls[0]['id'], 'cpu12345')
            self.assertEqual(calls[0]['resume'], 'must')
            self.assertEqual(Path(captured['result']['render_path']), folder / 'final_episode.png')
            self.assertEqual(Path(captured['result']['animation_path']), folder / 'final_episode.mp4')
            self.assertEqual(captured['fps'], 10)
            self.assertEqual(published['masac']['wandb']['status'], 'ok')
            self.assertTrue((root / 'visualization-publication.json').is_file())


    def test_cpu_visualization_invokes_media_publisher_after_evaluation(self):
        module = self.module()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {'entity': 'owner', 'project': 'project', 'kernel_ref': 'owner/cpu-run',
                    'config': 'configs/comparison-20k.yaml', 'overrides': ['pql.enabled=false'],
                    'artifacts': {'masac': 'owner/project/model:v1'},
                    'runs': {'masac': {'id': 'cpu99999', 'name': 'MASAC-CPU'}},
                    'cpu_eval': {'episodes': 1, 'seed': 10044, 'frame_stride': 5, 'fps': 10},
                    'training_status': {'masac': {'episodes': 20000}}}
            class Run:
                def __init__(self): self.summary = {}
                def log(self, *args, **kwargs): pass
                def save(self, *args, **kwargs): pass
                def use_artifact(self, *args, **kwargs): pass
            run = Run()
            tracker = SimpleNamespace(run=run, identity=lambda: {
                'wandb_run_id': 'cpu99999', 'wandb_url': 'https://wandb/run/cpu99999',
                'wandb_entity': 'owner', 'wandb_project': 'project',
                'wandb_workspace_url': 'https://wandb/workspace'}, close=lambda failed=False: None)
            def evaluate(*args, output_dir=None, **kwargs):
                folder = Path(output_dir); folder.mkdir(parents=True)
                image = folder / 'final_episode.png'; image.write_bytes(b'png')
                video = folder / 'final_episode.mp4'; video.write_bytes(b'video')
                (folder / 'metrics.json').write_text('{}')
                return {'algorithm': 'masac', 'episodes': 1, 'device': 'cpu',
                        'render_path': str(image), 'animation_path': str(video),
                        'mean_return': 1.0, 'success_rate': 0.0,
                        'mean_search_rate_percent': 50.0, 'mean_delivery_rate_percent': 40.0}
            with patch.object(module.torch.cuda, 'is_available', return_value=False), \
                 patch('uav_network.cloud.remote.prepare_checkpoints', return_value={'masac': str(root / 'policy.pt')}), \
                 patch('uav_network.experiments.tracking.start_tracking', return_value=tracker), \
                 patch.object(module, 'evaluate', side_effect=evaluate), \
                 patch.object(module, 'publish_visualization_media', return_value={
                     'huggingface': {'status': 'ok'}, 'wandb': {'status': 'ok'}}) as publish:
                result = module.run_visualization(spec, root / 'cpu')
            publish.assert_called_once()
            self.assertEqual(result['masac']['publication']['huggingface']['status'], 'ok')


    def test_case_publisher_downloads_completed_cpu_kernel_via_api_before_publishing(self):
        module = importlib.import_module('uav_network.cloud.media_publish')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'launch.json'
            manifest.write_text(json.dumps({'account': 'account_01',
                'kernel_ref': 'owner/uavnet-train-example', 'root_kernel_ref': 'owner/uavnet-train-example'}))
            account = SimpleNamespace(name='account_01')
            credentials = SimpleNamespace(accounts=[account])
            downloaded = root / 'downloaded'
            published = {'masac': {'huggingface': {'status': 'ok'}, 'wandb': {'status': 'ok'}}}
            with patch.object(module, 'load_credentials', return_value=credentials), \
                 patch.object(module, 'kernel_status', return_value='complete'), \
                 patch.object(module, 'run_kaggle') as transport, \
                 patch.object(module, 'publish_downloaded_visualizations', return_value=published) as publish:
                result = module.download_and_publish_case(manifest, downloaded, root / 'key.txt')
            self.assertEqual(result['status'], 'published')
            self.assertTrue(transport.called)
            args = transport.call_args.args[0]
            self.assertEqual(args[:2], ['kernels', 'output'])
            self.assertTrue(args[2].endswith('-cpu'))
            publish.assert_called_once_with(downloaded, root / 'key.txt')


    def test_case_publisher_treats_not_yet_created_cpu_kernel_as_pending(self):
        module = importlib.import_module('uav_network.cloud.media_publish')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'launch.json'
            manifest.write_text(json.dumps({'account': 'account_01',
                'kernel_ref': 'owner/uavnet-train-example', 'root_kernel_ref': 'owner/uavnet-train-example'}))
            account = SimpleNamespace(name='account_01')
            credentials = SimpleNamespace(accounts=[account])
            with patch.object(module, 'load_credentials', return_value=credentials), \
                 patch.object(module, 'kernel_status', side_effect=RuntimeError('not found')), \
                 patch.object(module, 'run_kaggle') as transport:
                result = module.download_and_publish_case(manifest, root / 'downloaded', root / 'key.txt')
            self.assertEqual(result['status'], 'pending')
            self.assertEqual(result['error_type'], 'RuntimeError')
            transport.assert_not_called()


if __name__ == '__main__':
    unittest.main()
