"""v6.3 repo-wide package, setup, reproducible source data and governance."""
from pathlib import Path
import json, os, subprocess, sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_root_setuptools_finds_nested_algorithm_packages():
    import setuptools
    cfg=tomllib.loads((ROOT/'pyproject.toml').read_text())
    include=cfg['tool']['setuptools']['packages']['find']['include']
    assert 'uav_network*' in include
    found=setuptools.find_packages(where=str(ROOT),include=include)
    for package in ('uav_network.algorithms.matd3','uav_network.algorithms.masac',
                    'uav_network.algorithms.baselines','uav_network.algorithms.core',
                    'uav_network.algorithms.configuration'):
        assert package in found
    assert any('PyYAML' in item for item in cfg['project']['dependencies'])


def test_research_wheel_has_all_new_kernel_packages():
    import setuptools
    cfg=tomllib.loads((ROOT/'research/zhang_xue/pyproject.toml').read_text())
    source=ROOT/'research/zhang_xue/src'
    packages=setuptools.find_packages(where=str(source),include=['zhang_xue*'])
    for package in ('zhang_xue.gpu.kernels','zhang_xue.learners','zhang_xue.compat',
                    'zhang_xue.communication','zhang_xue.environment'):
        assert package in packages
    assert not list(source.glob('*.py'))


def test_setup_has_real_repo_wide_commands_not_just_scenario():
    s=(ROOT/'setup.sh').read_text()
    for flag in ('--prepare-data','--check-only','--cpu-gate','--test','--gpu-gate','--dry-run'):
        assert flag in s
    assert 'scripts/data/prepare_data.py' in s
    assert (ROOT/'configs/data/sources.yaml').is_file()
    assert (ROOT/'configs/learners/matd3_cpu_gate.yaml').is_file()
    assert (ROOT/'configs/learners/masac_cpu_gate.yaml').is_file()


def test_synthetic_source_preparation_is_reproducible_and_local(tmp_path):
    script=ROOT/'scripts/data/prepare_data.py'
    manifest=ROOT/'configs/data/sources.yaml'
    one=tmp_path/'first';two=tmp_path/'second'
    for directory in (one,two):
        subprocess.run([sys.executable,str(script),'--manifest',str(manifest),
                        '--output-dir',str(directory),'--seed', '44', '--episodes', '2'],
                       check=True,env={**os.environ,'PYTHONPATH':str(ROOT)+os.pathsep+str(ROOT/'research/zhang_xue/src')},
                       capture_output=True,text=True)
    def files(path):
        return {p.relative_to(path).as_posix():p.read_bytes() for p in path.rglob('*') if p.is_file()}
    assert files(one)==files(two)
    meta=json.loads((one/'metadata.json').read_text())
    assert meta['kind']=='generated-synthetic-no-external-dataset'
    assert len(meta['episodes'])==2
    assert all(len(ep['target_cells'])==15 for ep in meta['episodes'])


def test_external_download_requires_https_and_sha256(tmp_path):
    from scripts.data.prepare_data import validate_manifest
    for bad in ({'kind':'download','url':'http://example.com/a','sha256':'a'*64},
                {'kind':'download','url':'https://example.com/a'},
                {'kind':'download','url':'file:///tmp/sensitive','sha256':'a'*64}):
        with pytest.raises(ValueError):
            validate_manifest({'sources':[bad]})
