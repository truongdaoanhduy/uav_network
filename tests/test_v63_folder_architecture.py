"""Architecture contract: algorithm variants are real packages, not a flat drawer."""
from pathlib import Path
import importlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALG = ROOT / 'uav_network' / 'algorithms'


def test_matd3_masac_have_own_packages_and_shared_core():
    for name in ('matd3', 'masac', 'baselines', 'core', 'configuration'):
        assert (ALG / name / '__init__.py').is_file(), name
    assert (ALG / 'matd3' / 'learner.py').is_file()
    assert (ALG / 'masac' / 'learner.py').is_file()
    assert not (ALG / 'matd3.py').exists()
    assert not (ALG / 'masac.py').exists()
    assert not (ALG / 'vanilla_baselines.py').exists()
    assert not (ALG / 'models.py').exists()
    assert not (ALG / 'replay.py').exists()


def test_legacy_import_paths_remain_valid_without_flat_files():
    from uav_network.algorithms.matd3 import MATD3, td3_target
    from uav_network.algorithms.masac import MASAC, soft_bellman_target
    from uav_network.algorithms.vanilla_baselines import VanillaMATD3, VanillaMASAC
    from uav_network.algorithms.configuration import LearnerConfig
    from uav_network.algorithms.checkpoints import load_policy
    for item in (MATD3, MASAC, td3_target, soft_bellman_target,
                 VanillaMATD3, VanillaMASAC, LearnerConfig, load_policy):
        assert callable(item)


def test_folder_policy_all_source_files_at_most_500_lines():
    from scripts.maintenance.source_inventory import source_paths
    for name in source_paths(ROOT):
        f = ROOT / name
        if f.suffix != '.py':
            continue
        assert len(f.read_text().splitlines()) <= 500, f.relative_to(ROOT)


def test_gpu_compression_not_falsely_reported_as_active():
    from zhang_xue.gpu.runtime import WarpSevenRadioNetwork
    from zhang_xue.gpu.layout import backing_bytes_per_environment
    assert WarpSevenRadioNetwork.active_soa_bytes_per_environment() == backing_bytes_per_environment()
    assert WarpSevenRadioNetwork.active_layout_id() == 'v61_uncompressed'
