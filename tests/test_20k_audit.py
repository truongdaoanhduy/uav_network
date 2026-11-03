"""Regression checks from the 20k production audit."""
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from uav_network.algorithms.matd3 import MATD3
from uav_network.configuration import load_config
from uav_network.environment.observations import observation_dim, state_dim

ROOT = Path(__file__).resolve().parents[1]


class Production20kAuditTests(unittest.TestCase):
    def test_matd3_keeps_source_categorical_sampling_during_training(self):
        torch.set_num_threads(1)
        torch.manual_seed(44)
        config = load_config(ROOT / 'configs/comparison-20k.yaml')
        # Zero logits make deterministic evaluation choose destination 0. During
        # training, Ackermann-style Gumbel categorical sampling remains stochastic
        # rather than following the removed epsilon-greedy schedule.
        config.matd3.hidden_sizes = [16, 16]
        learner = MATD3(config, observation_dim(config), state_dim(config))
        with torch.no_grad():
            for parameter in learner.actors.parameters():
                parameter.zero_()
        obs = np.zeros((4096, config.world.num_uavs, observation_dim(config)), dtype=np.float32)
        obs[:, :, 9] = 1
        obs[:, :, -(config.world.num_uavs + 1):] = 1
        for step in (config.matd3.learning_starts,
                     config.pql.warmup_collection_steps * config.runtime.num_envs,
                     config.matd3.epsilon_decay_steps):
            with self.subTest(step=step):
                sampled = learner.act_batch(obs, step=step).destination
                self.assertGreater(float((sampled != 0).mean()), .7)
        deterministic = learner.act_batch(obs, deterministic=True, step=0).destination
        self.assertTrue(np.all(deterministic == 0))

    def test_media_download_excludes_checkpoint_and_cache_outputs(self):
        from uav_network.cloud import media_publish
        names = ['job.json', 'hardware.json', 'visualization-result.json',
                 'visualize/masac/metrics.json', 'visualize/masac/final_episode.png',
                 'visualize/masac/final_episode.mp4', 'visualize/masac/final_episode.gif',
                 'checkpoints/masac/checkpoint.pt', 'cache/checkpoint.pt.metadata',
                 'tracking/masac/output.log']
        selected = []
        def transport(args, account, timeout):
            pattern = args[args.index('--file-pattern') + 1] if '--file-pattern' in args else '.*'
            selected.extend(name for name in names if re.search(pattern, name))
            return ''
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'launch.json'
            manifest.write_text(json.dumps({'account': 'one', 'kernel_ref': 'owner/training'}))
            credentials = SimpleNamespace(accounts=[SimpleNamespace(name='one')])
            with patch.object(media_publish, 'load_credentials', return_value=credentials), \
                 patch.object(media_publish, 'kernel_status', return_value='complete'), \
                 patch.object(media_publish, 'run_kaggle', side_effect=transport), \
                 patch.object(media_publish, 'publish_downloaded_visualizations', return_value={}):
                media_publish.download_and_publish_case(manifest, root / 'download', root / 'key.txt')
        self.assertEqual(set(selected), set(names[:7]),
                         'Publishing media must not download checkpoint/replay or cache files')


if __name__ == '__main__':
    unittest.main()
