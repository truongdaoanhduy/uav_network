"""Contract for the new NoOracle V9 experiment, matching 18d756ef Eval/Test mechanics."""
import ast
import base64
import hashlib
import io
import json
import lzma
import re
import tarfile

import nbformat
from uav_network.configuration import load_config
from uav_network.experiments.gpu_eval_layout import plan_test, plan_validation
from scripts.build_matd3_v9_ct095_h256x256_test1000_ui_20261123 import (
    build_notebook, CONFIG_PATH,
)


def test_new_science_config_changes_only_requested_fields():
    base = load_config('configs/matd3-18d756ef-no-oracle-cell-reports-20261122.yaml')
    new = load_config(CONFIG_PATH)
    assert base.sensor.confirmation_threshold == 0.99
    assert new.sensor.confirmation_threshold == 0.95
    assert new.matd3.hidden_sizes == [256, 256]
    assert new.runtime.eval_episodes == 1000
    assert new.tracking.periodic_evaluation_episodes == 100
    assert new.runtime.num_envs == 4096
    assert new.matd3.updates_per_step == 4
    assert new.matd3.batch_size == 8192
    assert new.runtime.seed == 44
    assert new.runtime.eval_seed == base.runtime.eval_seed == 2000000044
    assert new.network.enabled and new.sensor.report_cell_hypotheses
    assert new.world.num_uavs == 6 and new.world.num_targets == 35
    assert new.world.num_obstacles == 25 and new.world.max_steps == 1500
    # Keep the NoOracle science, communication, energy, reward, and training knobs.
    assert new.network == base.network
    assert new.reward == base.reward
    assert new.energy == base.energy
    assert new.world == base.world


def test_two_gpu_eval_and_1000_seed_final_test_layout_exactly_once():
    plan0 = plan_validation(6, 100, 200, gpu_index=0, gpu_count=2)
    plan1 = plan_validation(6, 100, 200, gpu_index=1, gpu_count=2)
    assert [i for batch in plan0 for i in batch.checkpoint_indices] == [0, 2, 4]
    assert [i for batch in plan1 for i in batch.checkpoint_indices] == [1, 3, 5]
    assert all(batch.worlds <= 200 for batch in plan0 + plan1)
    assert plan_test(1000, 2, 500) == (((0, 500),), ((500, 1000),))


def test_notebook_has_new_config_eval_test_and_pinned_source():
    n = build_notebook()
    nbformat.validate(n)
    assert len(nbformat.writes(n).encode()) < 1_000_000
    assert n.metadata['protocol']['validation_episodes'] == 100
    assert n.metadata['protocol']['test_episodes'] == 1000
    assert n.metadata['protocol']['report_identity'] == 'observed_cell_id'
    assert n.metadata['reference_run'].endswith('18d756ef')
    assert '0.95' in n.cells[0].source and '256' in n.cells[0].source
    assert 'VALIDATION_EPISODES = 100' in n.cells[1].source
    assert 'MAX_WARP_TEST_WORLDS_PER_GPU = 500' in n.cells[1].source
    assert 'CHECKPOINTS_PER_GPU_BATCH = 2' in n.cells[1].source
    c = n.cells[2].source
    ast.parse(c)
    assert f'SPEC["config"] = "{CONFIG_PATH}"' in c
    assert 'SPEC["dense_eval_schedule"]["final_test_scenarios"] = 1000' in c
    assert 'H256x256' in c and 'CT095' in c
    assert 'runtime.episodes=20000' in c
    assert '--max-test-worlds-per-gpu' in c
    assert re.search(r'--test-batch-size.{0,20}0', c)  # zero selects per-GPU cap
    m = re.search(r'payload\s*=\s*base64\.b85decode\("([^"\n]+)"\)', c)
    assert m
    bundle = base64.b85decode(m.group(1))
    assert hashlib.sha256(bundle).hexdigest() == n.metadata['source_sha256']
    with tarfile.open(fileobj=io.BytesIO(lzma.decompress(bundle))) as tf:
        member = next(x for x in tf.getmembers() if x.name == CONFIG_PATH)
        text = tf.extractfile(member).read().decode()
        assert 'confirmation_threshold: 0.95' in text
        assert 'eval_episodes: 1000' in text


def test_runtime_effective_spec_and_eval_actor_architecture():
    """Execute only the injected, side-effect-free spec customization."""
    from scripts.build_matd3_nooracle_one_session_ui import make_spec
    from uav_network.configuration import load_config
    from uav_network.environment.observations import observation_dim
    from uav_network.experiments.deferred_gpu_evaluation import GPUActorOnlyPolicy
    from uav_network.algorithms.matd3 import MATD3
    import torch

    n = build_notebook()
    code = n.cells[2].source
    runtime = code.split('# CT095-H256x256-TEST1000: explicit values AFTER reference-style setup.')[1]
    runtime = runtime.split('SPEC["gpu_eval_candidate_limit"] = 512')[0]
    cfg = load_config(CONFIG_PATH)
    spec = make_spec('source-commit')
    scope = {
        'SPEC': spec, 'ALGORITHM': 'matd3', 'TRAIN_SEED': 44,
        'UPDATES_PER_VECTOR_STEP': 4, '_rid': 'test1234',
        'MAX_WARP_TEST_WORLDS_PER_GPU': 500,
        'CHECKPOINTS_PER_GPU_BATCH': 2,
    }
    exec(runtime, scope)
    assert spec['config'] == CONFIG_PATH
    assert spec['dense_eval_schedule']['final_test_scenarios'] == 1000
    assert len(spec['overrides']) == 4
    assert any(x.startswith('tracking.group=MATD3-NOORACLE-V9-CT095') for x in spec['overrides'])
    assert 'CT095' in spec['runs']['matd3']['name']
    resolved = load_config(spec['config'], spec['overrides'])
    assert resolved.sensor.confirmation_threshold == 0.95
    assert resolved.matd3.hidden_sizes == [256,256]
    assert resolved.runtime.eval_episodes == 1000
    # The real Val/Test Actor-only loader must reconstruct the new H256x256 Actor.
    policy = GPUActorOnlyPolicy(resolved, 'cpu')
    assert len(policy.actors) == 1
    hidden = [m.out_features for m in policy.actors[0].modules()
              if isinstance(m, torch.nn.Linear)]
    assert 256 in hidden
    actor_state = policy.actors.state_dict()
    assert actor_state
