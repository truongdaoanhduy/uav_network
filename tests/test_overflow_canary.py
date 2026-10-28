"""A CUDA diagnostic must not silently allocate a CPU production replay."""
import pytest
from unittest.mock import patch
from scripts.probe_overflow_canary import run_case


def test_canary_rejects_cpu_before_any_output_or_large_allocation(tmp_path):
    with patch('torch.cuda.is_available', return_value=False):
        with pytest.raises(RuntimeError, match='no CPU fallback'):
            run_case('masac', False, tmp_path / 'run')
    assert not (tmp_path / 'run').exists()
