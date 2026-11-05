"""Proof that Kaggle receives the *actual* local source in its one code_file."""
import ast
import base64
import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from uav_network.cloud.kernel import kernel_script
from uav_network.cloud.source_bundle import build_source_archive


PROJECT = Path(__file__).resolve().parents[1]


def test_api_kernel_code_contains_all_runtime_source_without_remote_checkout():
    source = kernel_script({
        "stage": "train", "kernel_ref": "owner/test-1500", "entity": "owner",
        "project": "project", "commit": "a" * 40,
        "training_only": True,
    })
    assert "git clone" not in source
    assert "git checkout" not in source
    assert "https://github.com/" not in source
    assert "https://gitlab.com/" not in source
    assert "scripts/requirements-network.txt" in source
    assert "base64.b85decode(" in source
    assert "SHA256" in source.upper()
    compile(source, "kernel.py", "exec")

    bundle = build_source_archive(PROJECT)
    encoded = base64.b85encode(bundle).decode()
    assert encoded in source
    assert hashlib.sha256(bundle).hexdigest() in source
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        names = {m.name for m in archive.getmembers()}
        assert "uav_network/algorithms/masac.py" in names
        assert "uav_network/algorithms/matd3.py" in names
        assert "configs/comparison-reference-masac-matd3-xyz-sensing.yaml" in names
        assert "third_party/UavNetSim/simulator/simulator.py" in names
        assert "third_party/UavNetSim/phy/channel.py" in names
        assert "scripts/requirements-network.txt" in names
        assert "pyproject.toml" in names
        assert len(names) > 200
        assert all(".git" not in n.split("/") for n in names)
        assert all("key.txt" not in n and ".env" not in n and
                   "outputs/" not in n and "tests/" not in n for n in names)


def test_bundle_refuses_unpinned_upstream():
    with patch("uav_network.cloud.source_bundle._revision", return_value="f" * 40):
        with pytest.raises(ValueError, match="pinned upstream"):
            build_source_archive(PROJECT)


def test_actual_remote_bootstrap_unpacks_embedded_source_without_any_git_call(tmp_path):
    source = kernel_script({
        "stage": "train", "kernel_ref": "owner/test-1500", "entity": "owner",
        "project": "project", "commit": "b" * 40,
        "training_only": True,
    })
    nodes = ast.parse(source).body
    start = next(i for i, x in enumerate(nodes) if isinstance(x, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "payload" for t in x.targets))
    end = next(i for i in range(start + 1, len(nodes)) if isinstance(nodes[i], ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "python" for t in nodes[i].targets))
    block = nodes[start:end]
    for x in block:
        if (isinstance(x, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == "root" for t in x.targets)):
            x.value.args[0].value = str(tmp_path / "unpacked")

    imports = [x for x in nodes if isinstance(x, (ast.Import, ast.ImportFrom))]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[*imports, *block],
                type_ignores=[])), "kernel.py", "exec"), namespace)
    root = tmp_path / "unpacked"
    assert (root / "uav_network/algorithms/masac.py").is_file()
    assert (root / "uav_network/algorithms/matd3.py").is_file()
    assert (root / "third_party/UavNetSim/simulator/simulator.py").is_file()
    assert (root / "configs/comparison-reference-masac-matd3-xyz-sensing.yaml").is_file()
    assert not (root / ".git").exists()
    assert not (root / "key.txt").exists()
    assert not (root / "setup.sh").exists()


def test_future_five_case_configs_use_1500_steps_and_same_pql_replica():
    import json
    import hashlib
    from uav_network.configuration import load_config

    matrix = [(1, "masac", False), (2, "masac", True),
              (3, "matd3", False), (4, "matd3", True),
              (5, "masac", True)]
    checksums = {}
    for case, algo, pql in matrix:
        overrides = [
            "world.max_steps=1500", "runtime.episodes=20000", "runtime.seed=44",
            "runtime.deterministic=true", "runtime.collector_overlap=false",
            "runtime.compile_models=false", "runtime.fused_optimizers=false",
            "runtime.vectorize_critics=false", "runtime.critic_chunk_size=0",
            "tracking.resumable_checkpoint_interval_seconds=0",
            f"runtime.algorithm={algo}", f"pql.enabled={str(pql).lower()}",
            f"{algo}.share_actor_parameters=true", f"{algo}.share_critic_parameters=true",
            f"runtime.output_dir=outputs/pending-case-{case:02d}",
        ]
        config = load_config(PROJECT / "configs/comparison-reference-masac-matd3-xyz-sensing.yaml",
                             overrides)
        assert config.world.max_steps == 1500
        assert config.runtime.episodes == 20000 and config.runtime.seed == 44
        assert config.runtime.deterministic and config.pql.enabled == pql
        assert config.algorithm_config.share_actor_parameters
        assert config.algorithm_config.share_critic_parameters
        assert config.tracking.checkpoint_store == "huggingface"
        assert config.tracking.evaluation_interval_steps > 0
        data = config.to_dict()
        data["runtime"].pop("output_dir", None)
        checksums[case] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    assert checksums[2] == checksums[5]
