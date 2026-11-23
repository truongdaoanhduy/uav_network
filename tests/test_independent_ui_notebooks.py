"""The exported notebooks must launch the declared recipe, not an old SH profile."""
import ast
import base64
import os
import sys
import types

import pytest

from uav_network.cloud import kernel
from uav_network.configuration import load_config

PROFILE = "configs/comparison-independent-baseline-xyz-sensing.yaml"
STAGE_START = 'if SPEC["stage"] in {"visualize", "archive"}:'


def credentials_code(script):
    begin = script.index("from kaggle_secrets import UserSecretsClient")
    end = script.index('spec_file = Path("/kaggle/working/job.json")', begin)
    return script[begin:end]


def test_explicit_source_archive_is_embedded_without_rebuilding(monkeypatch):
    def unexpected_rebuild(root):
        pytest.fail("A supplied archive must remain the exact frozen input")

    monkeypatch.setattr(kernel, "build_source_archive", unexpected_rebuild)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    frozen = b"frozen audit archive"
    spec = {"stage": "train", "entity": "entity", "project": "project",
            "commit": "source", "kernel_ref": "owner/slug"}
    source = kernel.kernel_script(spec, source_archive=frozen)
    tree = ast.parse(source)
    encoded = next(n.args[0].value for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "b85decode")
    assert base64.b85decode(encoded) == frozen
    # A corrupt payload must fail the real generated checksum before extraction.
    prefix = source[:source.index('root = Path("/tmp/uav_network")')]
    tree = ast.parse(prefix)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "b85decode":
            node.args[0] = ast.Constant(base64.b85encode(b"corrupt").decode())
    with pytest.raises(RuntimeError, match="checksum"):
        exec(compile(ast.fix_missing_locations(tree), "<corrupt notebook>", "exec"), {})


@pytest.mark.parametrize("secret", ["wandb-secret-fixture", "", None])
def test_kaggle_secrets_bootstrap_validates_and_uses_current_users_secrets(monkeypatch, capsys, secret):
    script = kernel.kernel_script({}, source_archive=b"frozen", credential_source="kaggle_secrets")
    module = types.ModuleType("kaggle_secrets")
    class Secrets:
        def get_secret(self, name):
            return secret if name == "WANDB_API_KEY" else "hf-secret-fixture"
    module.UserSecretsClient = Secrets
    monkeypatch.setitem(sys.modules, "kaggle_secrets", module)
    environment = {}
    scope = {"os": types.SimpleNamespace(environ=environment),
             "SPEC": {"checkpoint_repo": "owner/checkpoints"}}
    if secret:
        exec(credentials_code(script), scope)
        assert environment == {"WANDB_API_KEY": secret, "HF_TOKEN": "hf-secret-fixture",
                               "HF_REPO_ID": "owner/checkpoints"}
    else:
        with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
            exec(credentials_code(script), scope)
    captured = capsys.readouterr()
    assert "wandb-secret-fixture" not in captured.out + captured.err
    assert "hf-secret-fixture" not in captured.out + captured.err


def test_dataset_credentials_remain_the_default(tmp_path):
    (tmp_path / "wandb_api_key.txt").write_text("wandb-fixture")
    (tmp_path / "hf_token.txt").write_text("hf-fixture")
    script = kernel.kernel_script({}, source_archive=b"frozen")
    begin = script.index('inputs = Path("/kaggle/input")')
    end = script.index('spec_file = Path("/kaggle/working/job.json")', begin)
    env = {}
    exec(script[begin:end], {"Path": lambda name: tmp_path,
         "os": types.SimpleNamespace(environ=env),
         "SPEC": {"storage_credential_dataset": "owner/private", "checkpoint_repo": "owner/best"}})
    assert env["WANDB_API_KEY"] == "wandb-fixture"
    assert env["HF_TOKEN"] == "hf-fixture"
    assert env["HF_REPO_ID"] == "owner/best"


def test_unknown_credential_source_is_rejected():
    with pytest.raises(ValueError, match="credential_source"):
        kernel.kernel_script({}, source_archive=b"frozen", credential_source="other")


@pytest.mark.parametrize("algorithm", ["masac", "matd3"])
def test_independent_notebook_launches_the_effective_recipe_and_edited_seed(algorithm):
    from uav_network.cloud.ui_notebooks import build_independent_notebook
    import nbformat

    nb = build_independent_notebook(algorithm, b"frozen", "pinned-source", seed=44)
    nbformat.validate(nb)
    code = nb.cells[-1].source
    compile(nb.cells[1].source, "<preflight>", "exec")
    compile(code, "<notebook launch>", "exec")
    identities = []
    for seed in (44, 45):
        scope = {"NOTEBOOK_REF": "owner/independent-test", "TRAIN_SEED": seed}
        exec(code[:code.index(STAGE_START)], scope)
        spec = scope["SPEC"]
        effective = load_config(spec["config"], spec["overrides"])
        original = load_config(PROFILE)
        assert spec["config"] == PROFILE
        assert spec["commit"] == "pinned-source"
        assert set(spec["runs"]) == {algorithm}
        assert spec["training_only"] and not spec["cpu_visualize"]
        assert not spec.get("auto_continue") and not spec["warm_start_artifacts"]
        assert effective.runtime.algorithm == algorithm
        assert effective.runtime.seed == seed
        assert effective.runtime.critic_chunk_size == 0
        assert not effective.pql.enabled
        assert effective.masac.temperature_loss == "alpha"
        assert effective.masac.mask_aware_discrete_entropy_target
        assert effective.matd3.initialization == "default"
        for name in ("world", "sensor", "network", "reward", "energy"):
            assert getattr(effective, name) == getattr(original, name)
        for name in ("masac", "matd3"):
            options = getattr(effective, name)
            assert not options.share_actor_parameters
            assert not options.share_critic_parameters
            assert not options.actor_layer_norm
        assert effective.tracking.periodic_evaluation_seed == 1000000044
        assert f"seed{seed}" in effective.tracking.tags
        run = spec["runs"][algorithm]
        assert f"-INDEPENDENT-NOCHUNK-" in run["name"]
        assert f"-S{seed}-" in run["name"]
        assert run["id"] in run["name"]
        assert effective.runtime.output_dir.endswith(run["id"])
        identities.append(run["id"])
    assert identities[0] != identities[1]


@pytest.mark.parametrize("algorithm,seed", [("other", 44), ("masac", -1), ("masac", True)])
def test_notebook_rejects_unsupported_algorithm_and_seed(algorithm, seed):
    from uav_network.cloud.ui_notebooks import build_independent_notebook
    with pytest.raises(ValueError):
        build_independent_notebook(algorithm, b"frozen", "source", seed=seed)
