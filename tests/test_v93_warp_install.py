"""CUDA12 driver must select the pinned CUDA12-compatible Warp wheel."""
import pytest


def test_observed_vast_driver_570_selects_cu12_without_upgrading_driver():
    from scripts.maintenance.install_warp_compat import requirement_for_driver
    assert requirement_for_driver("570.195.03") == (
        "https://github.com/NVIDIA/warp/releases/download/v1.18.0/"
        "warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl")


def test_cuda13_compatible_driver_can_use_the_pinned_default_wheel():
    from scripts.maintenance.install_warp_compat import requirement_for_driver
    assert requirement_for_driver("580.95.05") == "warp-lang==1.18.0"


def test_invalid_or_pre_cuda12_driver_does_not_silently_select_wrong_binary():
    from scripts.maintenance.install_warp_compat import requirement_for_driver
    for value in ("", "unknown", "510.60.00"):
        with pytest.raises(ValueError):
            requirement_for_driver(value)

