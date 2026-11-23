"""Production TX5 deep forensic profile is enabled only for explicit training runs."""
import os
from pathlib import Path
import pytest
from uav_network.configuration import load_config
from scripts.launch_nooracle_map800_tx5_training_with_hf_forensics_20261123 import (
    CONFIG, forensic_environment, training_command, validate_production)


def test_production_hf_profile_preserves_critical_logs_and_public_upload():
    c=load_config(CONFIG)
    settings=forensic_environment({'PATH':os.environ['PATH']})
    assert c.tracking.enabled
    assert settings['UAV_DIAGNOSTIC_STORE']=='huggingface_public'
    assert settings['UAV_DIAGNOSTIC_HF_REPO']=='duy95/uav-search-diagnostics'
    assert settings['UAV_HF_VERIFY_PUBLIC_READBACK']=='true'
    assert settings['UAV_FORENSIC_ENABLED']=='true'
    assert settings['UAV_ROOT_CAUSE_AUDIT']=='true'
    assert settings['UAV_FORENSIC_WORLD_IDS']=='0,1'
    assert 1<=int(settings['UAV_FORENSIC_STRIDE'])<=32
    assert int(settings['UAV_FORENSIC_MAX_RAW_MIB'])>=384
    assert settings['UAV_FORENSIC_FULL_BELIEF']=='true'
    assert settings['UAV_DEFERRED_EXPORT_HF']=='true'
    assert settings['UAV_TRAINING_ONLY']=='true'
    assert settings['WANDB_MODE']=='online'
    assert settings['UAV_FAST_TRAINING']=='false'
    assert not any(k in settings for k in ('WANDB_API_KEY','HF_TOKEN','HF_API_KEY'))
    assert training_command()[-2:]==['--config',CONFIG]


def test_training_launcher_does_not_silently_run_gpu_on_cpu_or_overwrite(tmp_path):
    c=load_config(CONFIG)
    with pytest.raises(RuntimeError,match='two CUDA GPUs'):
        validate_production(c,cuda_count=0)
    with pytest.raises(RuntimeError,match='two CUDA GPUs'):
        validate_production(c,cuda_count=1)
    c.runtime.output_dir=str(tmp_path/'existing')
    (tmp_path/'existing').mkdir()
    with pytest.raises(FileExistsError):validate_production(c,cuda_count=2)
    c.runtime.output_dir=str(tmp_path/'new')
    validate_production(c,cuda_count=2)
